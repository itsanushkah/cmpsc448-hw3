"""Step 2-3: pretrain the tiny GPT from scratch with next-token prediction.

Usage:
    python train.py                 # default ~0.8M params, 3000 steps
    python train.py --max_iters 1000 --n_layer 2   # faster / smaller
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch

from model import GPT, GPTConfig


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"  # Apple Silicon
    return "cpu"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data")
    p.add_argument("--out", default="out")
    p.add_argument("--block_size", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--n_layer", type=int, default=4)
    p.add_argument("--n_head", type=int, default=4)
    p.add_argument("--n_embd", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--max_iters", type=int, default=3000)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--eval_every", type=int, default=250)
    p.add_argument("--eval_iters", type=int, default=50)
    p.add_argument("--device", default=pick_device())
    args = p.parse_args()

    torch.manual_seed(1337)
    os.makedirs(args.out, exist_ok=True)
    meta = json.load(open(os.path.join(args.data, "meta.json")))
    chars = meta["chars"]
    itos = {i: c for i, c in enumerate(chars)}
    train = np.fromfile(os.path.join(args.data, "train.bin"), dtype=np.uint16)
    val = np.fromfile(os.path.join(args.data, "val.bin"), dtype=np.uint16)

    def get_batch(split):
        d = train if split == "train" else val
        ix = np.random.randint(0, len(d) - args.block_size - 1, args.batch_size)
        x = torch.stack([torch.from_numpy(d[i:i + args.block_size].astype(np.int64)) for i in ix])
        # targets are the inputs shifted one position to the right -> predict the NEXT token
        y = torch.stack([torch.from_numpy(d[i + 1:i + 1 + args.block_size].astype(np.int64)) for i in ix])
        return x.to(args.device), y.to(args.device)

    cfg = GPTConfig(vocab_size=len(chars), block_size=args.block_size, n_layer=args.n_layer,
                    n_head=args.n_head, n_embd=args.n_embd, dropout=args.dropout)
    model = GPT(cfg).to(args.device)
    print(f"Device: {args.device} | Parameters: {model.num_params():,} (randomly initialized)")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)

    def lr_at(it):  # linear warmup, then cosine decay
        if it < args.warmup:
            return args.lr * (it + 1) / args.warmup
        prog = (it - args.warmup) / max(1, args.max_iters - args.warmup)
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * prog)))

    @torch.no_grad()
    def estimate_loss():
        model.eval()
        out = {}
        for split in ("train", "val"):
            losses = [model(*get_batch(split))[1].item() for _ in range(args.eval_iters)]
            out[split] = sum(losses) / len(losses)
        model.train()
        return out

    def sample(n=200):
        model.eval()
        start = torch.zeros((1, 1), dtype=torch.long, device=args.device)
        s = "".join(itos[i] for i in model.generate(start, n)[0].tolist())
        model.train()
        return s

    print(f"Sample BEFORE training (random weights):\n{sample()}\n" + "-" * 60)

    best_val, log = float("inf"), []
    t0 = time.time()
    for it in range(args.max_iters + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(it)
        if it % args.eval_every == 0 or it == args.max_iters:
            l = estimate_loss()
            log.append({"iter": it, **l})
            print(f"step {it:5d} | train loss {l['train']:.3f} | val loss {l['val']:.3f} "
                  f"| {time.time() - t0:.0f}s")
            if l["val"] < best_val:
                best_val = l["val"]
                torch.save({"model": model.state_dict(), "config": cfg.__dict__, "chars": chars,
                            "iter": it, "val_loss": best_val}, os.path.join(args.out, "ckpt.pt"))
        if it == args.max_iters:
            break
        x, y = get_batch("train")
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    json.dump(log, open(os.path.join(args.out, "loss_log.json"), "w"), indent=1)
    print("-" * 60 + f"\nBest val loss {best_val:.3f}. Checkpoint saved to {args.out}/ckpt.pt")
    print(f"Sample AFTER training:\n{sample(400)}")


if __name__ == "__main__":
    main()
