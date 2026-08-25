"""
Sample text from a trained tttGPT checkpoint.

    python sample.py --out_dir=out --start="First Citizen:" --num_samples=3
"""

import os
import pickle
import argparse

import torch
from model import TTTGPTConfig, TTTGPT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=str, default='out')
    parser.add_argument('--start', type=str, default="\n")
    parser.add_argument('--num_samples', type=int, default=5)
    parser.add_argument('--max_new_tokens', type=int, default=500)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top_k', type=int, default=200)
    parser.add_argument('--seed', type=int, default=1337)
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    device = args.device
    checkpoint = torch.load(os.path.join(args.out_dir, 'ckpt.pt'),
                            map_location=device)

    # keep only keys the config knows (tolerates schema evolution)
    cfg_kwargs = {k: v for k, v in checkpoint['model_args'].items()
                  if k in TTTGPTConfig.__dataclass_fields__}
    model = TTTGPT(TTTGPTConfig(**cfg_kwargs))

    state_dict = checkpoint['model']
    for k in list(state_dict):
        if k.startswith('_orig_mod.'):
            state_dict[k[len('_orig_mod.'):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.to(device).eval()

    print(f"loaded '{cfg_kwargs.get('ttt_type', '?')}' model "
          f"({sum(p.numel() for p in model.parameters())/1e6:.1f}M params) "
          f"from {args.out_dir}")

    # char-level tokenizer from the dataset meta (matches prepare.py)
    with open(os.path.join('data', 'shakespeare_char', 'meta.pkl'), 'rb') as f:
        meta = pickle.load(f)
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s if c in stoi]   # skip unknown chars
    decode = lambda l: ''.join(itos[i] for i in l)

    torch.manual_seed(args.seed)
    if device == 'cuda':
        torch.cuda.manual_seed(args.seed)

    ids = encode(args.start)
    if not ids:
        raise ValueError("prompt has no characters present in the vocab")
    x = torch.tensor([ids], dtype=torch.long, device=device)

    for i in range(args.num_samples):
        y = model.generate(x, args.max_new_tokens,
                           temperature=args.temperature, top_k=args.top_k)
        print(f"\n--- sample {i+1} " + "-" * 45)
        print(decode(y[0].tolist()))


if __name__ == '__main__':
    main()
