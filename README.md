# CMPSC 448 HW3 — Pretraining a Small Language Model from Scratch

A tiny character-level GPT (~0.8M parameters, random init, no pretrained weights),
pretrained with next-token prediction on Bollywood song lyrics.

## Dataset
**Bollyrics** — about 10,000 Bollywood songs (1934–2019) as romanized Hindi lyrics,
from Gupta et al., *"Bollyrics: Automatic Lyrics Generator for Romanised Hindi"* (2020),
[arXiv:2007.12916](https://arxiv.org/abs/2007.12916), [github.com/lingo-iitgn/Bollyrics](https://github.com/lingo-iitgn/Bollyrics) (Apache-2.0).
`prepare.py` downloads the repo as a zip, combines all songs (shuffled across decades),
and keeps the first 5M characters. Vocabulary: 26 lowercase letters + space + newline.

## Files
| File | What it does |
|---|---|
| `prepare.py` | Downloads the dataset, builds a character vocabulary, writes `data/train.bin`, `data/val.bin`, `data/meta.json` (90/10 split) |
| `model.py` | Decoder-only transformer (token + position embeddings, 4 causal self-attention blocks, LayerNorm, MLP, tied output head) |
| `train.py` | Pretrains from scratch with cross-entropy on the next character; AdamW, warmup + cosine LR; saves best checkpoint to `out/ckpt.pt` |
| `chat.py` | Interactive prompt loop for the demo |

## Run
```bash
pip install torch numpy
python prepare.py      # Bollywood lyrics (default); also --dataset tinystories / shakespeare / sherlock, or --url <any .txt>
python train.py        # ~5 min on an Apple Silicon Mac
python chat.py         # type the start of a lyric, see the model continue it
```
In `chat.py`: `/temp 0.5` (safer, more repetitive), `/temp 1.2` (more random), `/len 500`, `/quit`.

Example prompts: `dil`, `tere bina`, `pyaar`, `tum hi ho`, `main tera`.

## What it shows
- **Before training** the model outputs random letters (random weights).
- **During training** loss falls from ~3.3 (= ln 28, uniform guessing over 28 characters) as it learns.
- **After training** it writes lyric-like lines: real Hindi words (dil, pyaar, zindagi), short lines,
  stanza breaks and repeated choruses — all learned only from predicting the next character.
- Real LLMs use the same next-token objective, scaled to far more data and parameters.
