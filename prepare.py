"""Step 1: download a text dataset and turn it into character-level token IDs.

Usage:
    python prepare.py                        # Bollywood song lyrics (default)
    python prepare.py --dataset tinystories  # simple children's stories
    python prepare.py --dataset shakespeare  # Tiny Shakespeare
    python prepare.py --url <any .txt URL>
"""
import argparse
import io
import json
import os
import random
import re
import urllib.request
import zipfile

import numpy as np

# Bollyrics (Gupta et al., 2020, arXiv:2007.12916; Apache-2.0): ~10K Bollywood songs from
# 1934-2019 as romanized Hindi lyrics, one .txt file per song. Downloaded as one zip of the repo.
BOLLYWOOD_ZIP = "https://github.com/lingo-iitgn/Bollyrics/archive/refs/heads/master.zip"


def load_bollywood(zip_bytes, max_chars, seed=0):
    """Combine every song in the zip into one text, songs separated by a blank line."""
    songs = []
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        for name in sorted(z.namelist()):
            if "/yearwise_dataset/" in name and name.endswith(".txt"):
                t = z.read(name).decode("utf-8", errors="ignore").lower()
                lines = [re.sub(r" +", " ", ln).strip() for ln in t.splitlines()]
                song = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
                if song:
                    songs.append(song)
    random.Random(seed).shuffle(songs)  # mix decades so train and val both cover all years
    print(f"Songs found: {len(songs):,}")
    text = "\n\n".join(songs) + "\n"
    text = "".join(c for c in text if c == "\n" or 32 <= ord(c) < 127)
    if max_chars and len(text) > max_chars:
        text = text[:text.rfind("\n\n", 0, max_chars)] + "\n"  # cut at a song boundary
    return text


DATASETS = {
    "bollywood": BOLLYWOOD_ZIP,
    "shakespeare": "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
    "sherlock": "https://www.gutenberg.org/cache/epub/1661/pg1661.txt",   # Adventures of Sherlock Holmes
    "austen": "https://www.gutenberg.org/cache/epub/1342/pg1342.txt",     # Pride and Prejudice
    "alice": "https://www.gutenberg.org/cache/epub/11/pg11.txt",          # Alice in Wonderland
    # TinyStories (Eldan & Li, 2023): short stories with simple vocabulary, designed so that
    # very small models can learn to write coherent English. We use the 22 MB validation file.
    "tinystories": "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt",
}


def clean_tinystories(text, max_chars):
    """Turn story separators into blank lines, keep plain ASCII only, cap the size."""
    stories = [s.strip() for s in text.split("<|endoftext|>") if s.strip()]
    text = "\n\n".join(stories) + "\n"
    for fancy, plain in {"“": '"', "”": '"', "‘": "'", "’": "'",
                         "—": "-", "–": "-", "…": "..."}.items():
        text = text.replace(fancy, plain)
    text = "".join(c for c in text if c == "\n" or 32 <= ord(c) < 127)
    if max_chars and len(text) > max_chars:
        text = text[:text.rfind("\n\n", 0, max_chars)] + "\n"  # cut at a story boundary
    return text


def strip_gutenberg(text):
    """Remove Project Gutenberg license header/footer if present."""
    start = text.find("*** START OF")
    end = text.find("*** END OF")
    if start != -1 and end != -1:
        text = text[text.find("\n", start) + 1:end]
    return text.strip() + "\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="bollywood", choices=DATASETS.keys())
    p.add_argument("--url", default=None, help="override with any plain-text URL")
    p.add_argument("--out", default="data")
    p.add_argument("--max_chars", type=int, default=5_000_000,
                   help="cap dataset size for Bollywood/TinyStories so training stays fast")
    args = p.parse_args()

    url = args.url or DATASETS[args.dataset]
    os.makedirs(args.out, exist_ok=True)
    print(f"Downloading {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    raw = urllib.request.urlopen(req).read()
    if url == BOLLYWOOD_ZIP:
        text = load_bollywood(raw, args.max_chars)
    else:
        text = raw.decode("utf-8", errors="ignore")
        text = strip_gutenberg(text).replace("\r\n", "\n")
        if "<|endoftext|>" in text:
            text = clean_tinystories(text, args.max_chars)

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
