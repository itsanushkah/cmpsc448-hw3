# CMPSC 448 HW3 — Pretraining a Small Language Model from Scratch

A tiny character-level GPT (~0.8M parameters, random init, no pretrained weights),
trained with next-token prediction on a downloaded text dataset.

## Files
| File | What it does |
|---|---|
| `prepare.py` | Downloads a text dataset, builds a character vocabulary, writes `data/train.bin`, `data/val.bin`, `data/meta.json` |
| `model.py` | Decoder-only transformer (token + position embeddings, 4 causal self-attention blocks, LayerNorm, MLP, tied output head) |
| `train.py` | Pretrains from scratch with cross-entropy on the next token; AdamW, warmup + cosine LR; saves best checkpoint to `out/ckpt.pt` |
| `chat.py` | Interactive prompt loop for the demo |

## Setup
```bash
pip install torch numpy
```

## Run
```bash
python prepare.py                 # Tiny Shakespeare (~1.1M characters)
# or: python prepare.py --dataset sherlock   (also: austen, alice, or --url <any .txt>)
python train.py                   # ~5-15 min on a laptop CPU, faster on Apple Silicon / GPU
python chat.py                    # type prompts, see generated text
```

Faster run if short on time: `python train.py --max_iters 1500`
Bigger/better run (GPU or patience): `python train.py --n_layer 6 --n_embd 192 --max_iters 5000`

In `chat.py`: `/temp 0.5` (more conservative) or `/temp 1.2` (more random), `/len 500`, `/quit`.

## What to point out in the demo
- **Before training** the model prints random gibberish (random weights).
- **During training** train/val loss falls from ~4.2 (≈ ln(vocab size), i.e. uniform guessing) to ~1.5.
- **After training** it produces text with correct spelling of common words, character names,
  line breaks and dialogue format — all learned only from predicting the next character.
- This is the same objective used to pretrain large LLMs; they just scale up the data, model and tokenizer.
