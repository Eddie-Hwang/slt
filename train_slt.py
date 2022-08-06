import os
from argparse import ArgumentParser
from collections import OrderedDict
from typing import Counter

import pytorch_lightning as pl
import torch
import yaml
from einops import rearrange
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torchtext.vocab import vocab 

from data import load_data, load_json
from signjoey.loss import XentLoss
from signjoey.metrics import bleu, chrf, rouge
from signjoey.model import build_model
from signjoey.search import greedy
from tokenizer import (HugTokenizer, SimpleTokenizer, build_vocab_from_phoenix,
                       white_space_tokenizer)


def load_config(path="configs/default.yaml") -> dict:
    """
    Loads and parses a YAML configuration file.

    :param path: path to YAML configuration file
    :return: configuration dictionary
    """
    with open(path, "r", encoding="utf-8") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    return cfg


def set_whitespace_tokenizer(train_path, mode, **kwargs):
    _vocab = build_vocab_from_phoenix(train_path, white_space_tokenizer, mode = mode)
    _tokenizer = SimpleTokenizer(white_space_tokenizer, _vocab)

    return {'tokenizer': _tokenizer, 'vocab': _vocab}


def set_hug_tokenizer(tokenizer_fpath, **kwargs):
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
            
    txt_json = load_json(tokenizer_fpath)
    txt_json = txt_json['model']['vocab']
    
    # torchtext cannot merge with Tokenizers of HuggingFace
    specials = ['<pad>', '<bos>', '<eos>', '<unk>'] # IMPORTANT: order is strict
    
    vocab_list = []
    for k in txt_json.keys():
        if k not in specials:
            vocab_list.append(k)
    vocab_counter = Counter(vocab_list)
    
    # the vocab size is 1000 with phoenix
    _vocab = vocab(OrderedDict(vocab_counter), specials = specials)
    
    _tokenizer = HugTokenizer(tokenizer_fpath)
    
    return {'tokenizer': _tokenizer, 'vocab': _vocab}


def set_tokenizer_dict():
    return {
        'whitespace': set_whitespace_tokenizer,
        'bpe': set_hug_tokenizer,
        'wordpiece': set_hug_tokenizer
    }


class SignLanguageTranslatorModule(pl.LightningModule):
    def __init__(
        self,
        cfg_file,
        num_save,
        dataset_type,
        train_path,
        valid_path,
        test_path,
        num_workers,
        lr,
        tokenizer_type,
        tokenizer_fpath,
        **kwargs
    ):
        super().__init__()

        self.save_hyperparameters()

        # load predefined slt config
        config = load_config(cfg_file)
        
        # load dataset
        trainset, validset, testset = load_data(
            dataset_type = dataset_type,
            train_trans_path = train_path,
            valid_trans_path = valid_path,
            test_trans_path = test_path,
            min_seq_len = 32,
            seq_len = 512, # model capacity cannot handle full sequence length of dataset.
        )

        tokenizer_dict = set_tokenizer_dict()
        
        _tokenizer = tokenizer_dict[tokenizer_type](
            train_path = train_path,
            tokenizer_fpath = tokenizer_fpath, 
            mode = 'text'
        )
        
        # define tokenzier
        self.text_tokenizer = _tokenizer['tokenizer']

        train_config = config["training"]
        
        self.batch_type = train_config.get("batch_type", "sentence")
        self.batch_size = train_config["batch_size"]
        
        model = build_model(
            cfg = config["model"],
            gls_vocab = None,
            txt_vocab = _tokenizer['vocab'],
            sgn_dim = sum(config["data"]["feature_size"])
            if isinstance(config["data"]["feature_size"], list)
            else config["data"]["feature_size"],
            do_recognition = False, # Note that we do not use gloss supervision of this model
            do_translation = True,
        )
        
        self.model = model
        self.txt_pad_index = self.model.txt_pad_index
        self.txt_bos_index = self.model.txt_bos_index
        self.txt_eos_index = self.model.txt_eos_index

        self.feature_size = config["data"]["feature_size"]

        self._get_translation_params(train_config = train_config)

        self.train_path = train_path
        self.valid_path = valid_path
        self.test_path = test_path

        self.num_display = num_save
        self.num_worker = num_workers

        self.lr = lr

        self.trainset = trainset
        self.validset = validset
        self.testset = testset

        self.eval_batch_size = train_config.get("eval_batch_size", self.batch_size)
        self.eval_batch_type = train_config.get("eval_batch_type", self.batch_type)
        self.dataset_version = config["data"].get("version", "phoenix_2014_trans")

        self.translation_max_output_length = train_config.get(
            "translation_max_output_length", None
        )

    def _get_translation_params(self, train_config) -> None:
        self.label_smoothing = train_config.get("label_smoothing", 0.0)
        self.translation_loss_function = XentLoss(
            pad_index=self.txt_pad_index, smoothing=self.label_smoothing
        )
        self.translation_normalization_mode = train_config.get(
            "translation_normalization", "batch"
        )
        if self.translation_normalization_mode not in ["batch", "tokens"]:
            raise ValueError(
                "Invalid normalization {}.".format(self.translation_normalization_mode)
            )
        self.translation_loss_weight = train_config.get("translation_loss_weight", 1.0)
        self.eval_translation_beam_size = train_config.get(
            "eval_translation_beam_size", 1
        )
        self.eval_translation_beam_alpha = train_config.get(
            "eval_translation_beam_alpha", -1
        )
        self.translation_max_output_length = train_config.get(
            "translation_max_output_length", None
        )

    def _common_step(self, batch, stage):
        text, joints = batch['text'], batch['joints']
        
        joint_inputs = pad_sequence(joints, batch_first = True, padding_value = 0.)
        joint_inputs = rearrange(joint_inputs, 'b t v c -> b t (v c)')
        
        joint_lengths = torch.tensor([len(j) for j in joints])
        joint_pad_mask = [torch.ones(j.size(0)) for j in joints]
        joint_pad_mask = pad_sequence(joint_pad_mask, batch_first = True)
        joint_pad_mask = joint_pad_mask.to(self.device)
        
        # tokenizing
        text_input_ids, text_pad_mask = self.text_tokenizer.encode(
            text, 
            padding = True, 
            add_special_tokens = True,
            device = self.device
        )
        
        joint_pad_mask = rearrange(joint_pad_mask, 'b t -> b () t')
        joint_pad_mask = joint_pad_mask.bool()
        
        text_pad_mask = rearrange(text_pad_mask, 'b t -> b () t')
        text_pad_mask = text_pad_mask.bool()
        
        decoder_outputs, _ = self.model(
            sgn = joint_inputs,
            sgn_mask = joint_pad_mask,
            sgn_lengths = joint_lengths,
            txt_input = text_input_ids,
            txt_mask = text_pad_mask,
        )
        
        word_outputs, _, _, _ = decoder_outputs
        
        # Calculate Translation Loss: It does not seems a right-shifted loss
        # txt_log_probs = F.log_softmax(word_outputs, dim=-1)
        # loss = self.translation_loss_function(txt_log_probs, text_input_ids)
        # loss /= self.batch_size
        
        word_outputs = rearrange(word_outputs, 'b n c -> b c n')
        loss = F.cross_entropy(word_outputs[:, :, :-1], text_input_ids[:, 1:], ignore_index = self.txt_pad_index)
        
        self.log(f'{stage}/loss', loss, batch_size = self.batch_size)

        if stage == 'tr':
            return loss

        encoder_output, encoder_hidden = self.model.encode(
            sgn = joint_inputs,
            sgn_mask = joint_pad_mask,
            sgn_length = joint_lengths
        )

        # we only use greedy here
        generated, _ = greedy(
            encoder_hidden = encoder_hidden,
            encoder_output = encoder_output,
            src_mask = joint_pad_mask,
            embed = self.model.txt_embed,
            bos_index = self.txt_bos_index,
            eos_index = self.txt_eos_index,
            decoder = self.model.decoder,
            max_output_length = self.translation_max_output_length,
        )
        
        return {
            'loss': loss,
            'generated': generated,
            'answer': text_input_ids,
            'text': text
        }

    def _common_epoch_end(self, outputs, stage):
        ans_text, gen_text, real_text = [], [], []
        for output in outputs:
            generated_outs = output['generated']
            generated_outs = torch.tensor(generated_outs)
            answers = output['answer']
            text = output['text']
            
            batch_gen_text = self.text_tokenizer.decode(
                generated_outs, 
                special_tokens = [self.txt_bos_index, self.txt_eos_index, self.txt_pad_index]
            )
            batch_ans_text = self.text_tokenizer.decode(
                answers,
                special_tokens = [self.txt_bos_index, self.txt_eos_index, self.txt_pad_index]
            )
            
            gen_text += batch_gen_text
            ans_text += batch_ans_text
            real_text += text
        
        txt_bleu = bleu(references = real_text, hypotheses = gen_text)
        txt_chrf = chrf(references = real_text, hypotheses = gen_text)
        txt_rouge = rouge(references = real_text, hypotheses = gen_text)
        
        # logging
        for k in txt_bleu.keys():
            self.log(f'{stage}/{k}', txt_bleu[k])
        self.log(f'{stage}/chrf', txt_chrf)
        self.log(f'{stage}/rouge', txt_rouge)

        # display results
        print('============================================================')
        print(f'[INFO] Sample token outputs at epoch {self.current_epoch}')
        print(f'\nReal text: {real_text[:self.num_display]}')
        print(f'\nAnswer decoded token: {ans_text[:self.num_display]}')
        print(f'\nGenerated decoded token: {gen_text[:self.num_display]}')
        print(f'\nScores: {txt_bleu}')
        print('============================================================')
        
    def training_step(self, batch, batch_idx):
        return self._common_step(batch, 'tr')

    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, 'val')

    def test_step(self, batch, batch_idx):
        return self._common_step(batch, 'tst')

    def validation_epoch_end(self, outputs):
        return self._common_epoch_end(outputs, 'val')

    def test_epoch_end(self, outputs):
        self._common_epoch_end(outputs, 'tst')

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr = self.lr)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer = optim,
            mode = 'min',
            factor = 0.5,
            patience = 10,
            cooldown = 10,
            min_lr = 1e-8,
            verbose = True
        )
        
        return {
            'optimizer': optim,
            'lr_scheduler': {
                'scheduler': sched,
                'monitor': 'val/bleu4',
                'frequency': self.trainer.check_val_every_n_epoch
            },
        }

    def train_dataloader(self):
        return DataLoader(
            self.trainset, 
            batch_size = self.batch_size, 
            shuffle = True, 
            num_workers = self.num_worker, 
            collate_fn = self._collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            self.validset, 
            batch_size = self.batch_size, 
            shuffle = False, 
            num_workers = self.num_worker, 
            collate_fn = self._collate_fn
        )

    def test_dataloader(self):
        return DataLoader(
            self.testset, 
            batch_size = self.batch_size, 
            shuffle = False, 
            num_workers = self.num_worker, 
            collate_fn = self._collate_fn
        )

    def _collate_fn(self, batch):
        id_list, text_list, gloss_list, joint_list = [], [], [], []
        id_list, text_list, joint_list = [], [], []
        
        sorted_batch = sorted(batch, key = lambda x: x['frame_len'], reverse = True)
        for data in sorted_batch:
            id_list.append(data['id'])
            text_list.append(data['text'])
            # gloss_list.append(data['gloss'])
            joint_list.append(data['joint_feats'])

        return {
            'id': id_list,
            'text': text_list,
            # 'gloss': gloss_list,
            'joints': joint_list
        }

    def get_callback_fn(self, monitor = 'val/loss', patience = 50):
        early_stopping_callback = EarlyStopping(
            monitor = monitor, 
            patience = patience, 
            mode = 'min', 
            verbose = True
        )
        ckpt_callback = ModelCheckpoint(
            filename = 'epoch={epoch}-val_loss={val/loss:.2f}', 
            monitor = monitor, 
            save_last = True, 
            save_top_k = 1, 
            mode = 'min', 
            verbose = True,
            auto_insert_metric_name = False
        )
        return early_stopping_callback, ckpt_callback

    def get_logger(self, type, name):
        if type == 'tensorboard':
            logger = TensorBoardLogger("slt_logs", name = name)
        else:
            raise NotImplementedError
        return logger


def main(hparams):
    pl.seed_everything(hparams.seed)
    
    module = SignLanguageTranslatorModule(**vars(hparams))
    
    early_stopping, ckpt = module.get_callback_fn('val/bleu4', patience = 50)
    
    callbacks_list = [ckpt]

    if hparams.use_early_stopping:
        callbacks_list.append(early_stopping)
    
    logger = module.get_logger('tensorboard', name = hparams.dataset_type)
    hparams.logger = logger
    
    trainer = pl.Trainer.from_argparse_args(hparams, callbacks = callbacks_list)
    
    if not hparams.test:
        trainer.fit(module, ckpt_path = hparams.ckpt if hparams.ckpt != None else None)
        if not hparams.fast_dev_run:
            trainer.test(module)
    else:
        assert hparams.ckpt != None, 'Trained checkpoint must be provided.'
        trainer.test(module, ckpt_path = hparams.ckpt)


if __name__=='__main__':
    parser = ArgumentParser(add_help = False)
    parser.add_argument('--seed', type = int, default = 42)
    parser.add_argument('--cfg_file', type = str, default = 'configs/sign.yaml')
    parser.add_argument("--fast_dev_run", action = "store_true")
    parser.add_argument('--dataset_type', default = 'phoenix')
    parser.add_argument('--train_path', default = '/home/ejhwang/projects/phoenix14t/data/phoenix14t.pose.train')
    parser.add_argument('--valid_path', default = '/home/ejhwang/projects/phoenix14t/data/phoenix14t.pose.dev')
    parser.add_argument('--test_path', default = '/home/ejhwang/projects/phoenix14t/data/phoenix14t.pose.test')
    parser.add_argument("--num_workers", type = int, default = 0)
    parser.add_argument("--max_epochs", type = int, default = 500)
    parser.add_argument('--check_val_every_n_epoch', type = int, default = 5)
    parser.add_argument('--accelerator', default = 'cpu')
    parser.add_argument('--devices', nargs = '+', type = int, default = None)
    parser.add_argument('--strategy', default = None)
    parser.add_argument('--num_save', type = int, default = 3)
    parser.add_argument('--use_early_stopping', action = 'store_true')
    parser.add_argument('--gradient_clip_val', type = float, default = 0.0)
    parser.add_argument('--lr', type = float, default = 1e-4)
    parser.add_argument('--test', action = 'store_true')
    parser.add_argument('--ckpt', default = None)
    parser.add_argument('--tokenizer_type', type = str, default = 'wordpiece', help="[bpe, wordpiece, whitespace]")
    parser.add_argument('--tokenizer_fpath', default = './wordpiece/phoenix/phoenix-wordpiece-512-vocab.json')

    hparams = parser.parse_args()

    # main(hparams)
    
    datast_type = hparams.dataset_type
    tokenizer_type = hparams.tokenizer_type
    
    tokenizer_fpath_list \
        = [f'./{tokenizer_type}/{datast_type}/{datast_type}-{tokenizer_type}-{2**i}-vocab.json' for i in range(14, 16)]
    
    for tokenizer_fpath in tokenizer_fpath_list:
        print(tokenizer_fpath)
        hparams.tokenizer_fpath = tokenizer_fpath
        main(hparams)


'''
(training on phoenix)
python train_slt.py \
    --accelerator gpu --devices 0 --num_workers 8 --use_early_stopping

(training on how2sign)
python train_slt.py \
   --accelerator gpu --devices 0 --num_workers 8 --use_early_stopping \
    --dataset_type how2sign \
    --train_path /home/ejhwang/projects/how2sign/how2sign_realigned_train.csv \
    --valid_path /home/ejhwang/projects/how2sign/how2sign_realigned_val.csv \
    --test_path /home/ejhwang/projects/how2sign/how2sign_realigned_test.csv
'''
