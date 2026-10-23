"""Step 1: download a text dataset and turn it into character-level token IDs.

Usage:
    python prepare.py                      # Tiny Shakespeare (default)
    python prepare.py --dataset sherlock   # Sherlock Holmes (Project Gutenberg)
    python prepare.py --url <any .txt URL>
"""
import argparse
import json
import os
import urllib.request

import numpy as np

DATASETS = {
    "shakespeare": "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
    "sherlock": "https://www.gutenberg.org/cache/epub/1661/pg1661.txt",   # Adventures of Sherlock Holmes
    "austen": "https://www.gutenberg.org/cache/epub/1342/pg1342.txt",     # Pride and Prejudice
    "alice": "https://www.gutenberg.org/cache/epub/11/pg11.txt",          # Alice in Wonderland
}


def strip_gutenberg(text):
    """Remove Project Gutenberg license header/footer if present."""
    start = text.find("*** START OF")
    end = text.find("*** END OF")
    if start != -1 and end != -1:
        text = text[text.find("\n", start) + 1:end]
    return text.strip() + "\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="shakespeare", choices=DATASETS.keys())
    p.add_argument("--url", default=None, help="override with any plain-text URL")
    p.add_argument("--out", default="data")
    args = p.parse_args()

    url = args.url or DATASETS[args.dataset]
    os.makedirs(args.out, exist_ok=True)
    print(f"Downloading {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    text = urllib.request.urlopen(req).read().decode("utf-8", errors="ignore")
    text = strip_gutenberg(text).replace("\r\n", "\n")

    with open(os.path.join(args.out, "input.txt"), "w", encoding="utf-8") as f:
        f.write(text)

    # Character-level tokenizer: every unique character is one token
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    ids = np.array([stoi[c] for c in text], dtype=np.uint16)

    n = int(0.9 * len(ids))  # 90% train / 10% validation
    ids[:n].tofile(os.path.join(args.out, "train.bin"))
    ids[n:].tofile(os.path.join(args.out, "val.bin"))
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump({"chars": chars, "source": url}, f)

    print(f"Characters in dataset: {len(text):,}")
    print(f"Vocab size: {len(chars)}")
    print(f"Train tokens: {n:,} | Val tokens: {len(ids) - n:,}")
    print(f"Saved to {args.out}/")


if __name__ == "__main__":
    main()
