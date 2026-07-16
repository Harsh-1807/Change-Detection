"""
Cache frozen-encoder features to disk, once.

This is the single biggest speedup available to you on CPU. Right now,
every epoch re-runs the full ViT forward pass on every tile — but while
the encoder is frozen, that forward pass produces the exact same output
every time. There's no reason to pay for it more than once.

After running this script, decoder training reads small cached tensors
from disk instead of running the ViT at all. In practice this turns
"N epochs x full ViT forward per tile" into "1x full ViT forward per tile,
ever" + cheap decoder-only epochs afterward.

Run this once per split (train/val) whenever the encoder or input tiles
change. If you later unfreeze the encoder for fine-tuning, the cache is
no longer valid for that phase (see note in train_decoder_cpu.py).
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from model_dinov3 import DinoV3Vegetation
from dataset import VegetationDataset  # your existing dataset class

DATA_ROOT = Path(r"C:\Users\13519\data\train\vegetation\vegetation_dino")
CACHE_ROOT = Path(r"C:\Users\13519\data\train\vegetation\feature_cache")

BATCH_SIZE = 4
NUM_WORKERS = 2


@torch.no_grad()
def cache_split(split: str, model: DinoV3Vegetation):
    ds = VegetationDataset(DATA_ROOT / split)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    out_dir = CACHE_ROOT / split
    out_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    idx = 0
    n = len(ds)
    for images, masks in loader:
        feats = model.encode(images)  # (B, 384, h, w) — the expensive part, done once
        for b in range(images.shape[0]):
            torch.save(
                {"feat": feats[b].clone(), "mask": masks[b].clone(), "hw": images.shape[-2:]},
                out_dir / f"{idx:06d}.pt",
            )
            idx += 1
        print(f"[{split}] {idx}/{n} cached", end="\r")
    print(f"[{split}] done: {idx} tiles cached to {out_dir}")


def main():
    model = DinoV3Vegetation(freeze_encoder=True)
    cache_split("train", model)
    cache_split("val", model)


if __name__ == "__main__":
    main()
