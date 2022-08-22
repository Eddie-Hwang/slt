from argparse import ArgumentParser

import torch
from einops import rearrange
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from signjoey.metrics import bleu, chrf, rouge
from signjoey.search import greedy
from train_slt import SignLanguageTranslatorModule


def load_model(ckpt, device = 'cpu'):
    module = SignLanguageTranslatorModule.load_from_checkpoint(ckpt)
    
    module.eval()
    module.freeze()
    
    tokenizer = module.text_tokenizer
    model = module.model
    model = model.to(device)
    
    print(f'[INFO] Pretrained model and tokenizer are loaded from {ckpt}.')
    
    return model, tokenizer


def main(args):
    # load model
    slt, tokenizer = load_model(args.ckpt, device = args.device)
    
    # load inputs
    inputs = torch.load(args.input_fpath)
    
    input_ids = inputs['ids']
    input_texts = inputs['texts']
    input_feats = inputs['outputs']
    generated_texts = []

    chunked_feats = [input_feats[i:i+args.batch_size] for i in range(0, len(input_feats), args.batch_size)]

    for joint_feat in tqdm(chunked_feats):
        joint_lengths = torch.tensor([j.size(0) for j in joint_feat], device = args.device)
        
        joint_pad_mask = [torch.ones(j.size(0)) for j in joint_feat]
        joint_pad_mask = pad_sequence(joint_pad_mask, batch_first = True, padding_value = 0.)
        joint_pad_mask = joint_pad_mask.to(args.device)
        joint_pad_mask = joint_pad_mask.bool()
        joint_pad_mask = rearrange(joint_pad_mask, 'b t -> b () t')
        
        joint_inputs = pad_sequence(joint_feat, batch_first = True, padding_value = 0.)
        
        if len(joint_inputs.size()) != 3:
            joint_inputs = rearrange(joint_inputs, 'b t v c -> b t (v c)')
        
        joint_inputs = joint_inputs.to(args.device)
        
        encoder_output, encoder_hidden = slt.encode(
            sgn = joint_inputs,
            sgn_mask = joint_pad_mask,
            sgn_length = joint_lengths
        )

        generated, _ = greedy(
            encoder_hidden = encoder_hidden,
            encoder_output = encoder_output,
            src_mask = joint_pad_mask,
            embed = slt.txt_embed,
            bos_index = args.txt_bos_index,
            eos_index = args.txt_eos_index,
            decoder = slt.decoder,
            max_output_length = args.translation_max_output_length,
        )

        gen_text = tokenizer.decode(
            torch.tensor(generated), 
            special_tokens = [args.txt_bos_index, args.txt_eos_index, args.txt_pad_index]
        )
        generated_texts += gen_text
    
    txt_bleu = bleu(references = input_texts, hypotheses = generated_texts)
    txt_chrf = chrf(references = input_texts, hypotheses = generated_texts)
    txt_rouge = rouge(references = input_texts, hypotheses = generated_texts)

    print('============================================================')
    print(f'BLEU: {txt_bleu}')
    print(f'\nCHRF: {txt_chrf}')
    print(f'\nROUGE: {txt_rouge}')
    print('============================================================')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--seed', type = int, default = 42)
    parser.add_argument('--ckpt', default = './slt_logs/phoenix/4096/checkpoints/last.ckpt')
    # parser.add_argument('--input_fpath', default = '/home/ejhwang/projects/NSLP-G/slp_logs/tfae/phoenix/valid_outputs/outputs.pt')
    parser.add_argument('--input_fpath', default = '/home/ejhwang/projects/ugloss/slp_logs/dVAE/phoenix/8192/test_outputs/outputs.pt')
    parser.add_argument('--translation_max_output_length', type = int, default = 30)
    parser.add_argument('--txt_pad_index', type = int, default = 0)
    parser.add_argument('--txt_bos_index', type = int, default = 1)
    parser.add_argument('--txt_eos_index', type = int, default = 2)
    parser.add_argument('--batch_size', type = int, default = 32)
    parser.add_argument('--device', type = str, default = 'cpu')

    args = parser.parse_args()

    main(args)

