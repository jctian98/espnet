# convert the ESPnet-SLM model checkpoint into a Transformers checkpoint
# Currently only for LLaMA modeling file. Mainly for textLLM test

import torch
import yaml
import sys
from pathlib import Path

input_ckpt = sys.argv[1]
output_ckpt = input_ckpt.replace(".pth", "_lm_eval.pth")

# ESPnet changes the eos token for these LLMs. When convert back to the original model, we replace the
# new eos token to keep proper behavior.
tgt_eos_token = 100257 # olmo
cur_eos_token = 5

def main():

    # (1) load config file and checkpoint state dict
    old_ckpt = torch.load(input_ckpt, map_location='cpu')
    if 'model' in old_ckpt:
        old_ckpt = old_ckpt['model']
    if 'module' in old_ckpt:
        old_ckpt = old_ckpt['module']
    config_yaml = Path(input_ckpt).parent / 'config.yaml'
    config = yaml.safe_load(open(config_yaml, 'r'))

    # (2) make the new state dict
    new_ckpt = dict()

    # (2.1) transformer body
    for k, v in old_ckpt.items():
        if k.startswith('corelm.decoders.model'):
            k = k.replace("corelm.decoders.model.", "model.")
            new_ckpt[k] = v
        
    
    # (2.2) get / output embeddings
    text_start, text_end = config['token_bias']['text_bpe']
    input_emb = old_ckpt['corelm.emb.weight'][text_start: text_end]
    output_emb = old_ckpt['corelm.lm_head.weight'][text_start: text_end]

    input_emb[tgt_eos_token] = old_ckpt['corelm.emb.weight'][cur_eos_token]
    output_emb[tgt_eos_token] = old_ckpt['corelm.lm_head.weight'][cur_eos_token]

    new_ckpt['model.embed_tokens.weight'] = input_emb
    new_ckpt['lm_head.weight'] = output_emb

    # (3) save the checkpoint
    torch.save(new_ckpt, output_ckpt)
    print(f'save {output_ckpt}')

if __name__ == "__main__":
    main()


