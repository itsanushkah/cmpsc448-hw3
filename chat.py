"""Step 4: prompt the trained model interactively (use this for the TA demo).

Usage:
    python chat.py
    python chat.py --length 400 --temperature 0.7
"""
import argparse

import torch

from model import GPT, GPTConfig
from train import pick_device


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="out/ckpt.pt")
    p.add_argument("--length", type=int, default=300, help="characters to generate")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=40)
    p.add_argument("--device", default=pick_device())
    args = p.parse_args()

    ck = torch.load(args.ckpt, map_location=args.device)
    model = GPT(GPTConfig(**ck["config"])).to(args.device)
    model.load_state_dict(ck["model"])
    model.eval()
    chars = ck["chars"]
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for i, c in enumerate(chars)}

    print(f"Loaded model: {model.num_params():,} params, trained {ck['iter']} steps, "
          f"val loss {ck['val_loss']:.3f}")
    print("Type a prompt and press Enter. Commands: /temp 0.5, /len 500, /quit\n")

    while True:
        try:
            prompt = input("prompt> ")
        except (EOFError, KeyboardInterrupt):
            break
        if prompt.strip() == "/quit":
            break
        if prompt.startswith("/temp"):
            args.temperature = float(prompt.split()[1]); print(f"temperature = {args.temperature}"); continue
        if prompt.startswith("/len"):
            args.length = int(prompt.split()[1]); print(f"length = {args.length}"); continue

        unknown = sorted({c for c in prompt if c not in stoi})
        if unknown:
            print(f"(skipping characters not in training vocab: {unknown})")
        ids = [stoi[c] for c in prompt if c in stoi] or [stoi.get("\n", 0)]
        x = torch.tensor([ids], dtype=torch.long, device=args.device)
        out = model.generate(x, args.length, temperature=args.temperature, top_k=args.top_k)
        print("\n" + "".join(itos[i] for i in out[0].tolist()) + "\n" + "-" * 60)


if __name__ == "__main__":
    main()
