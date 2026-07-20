"""
Cache frozen-encoder features to disk, once.
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from model_dinov3 import DinoV3Vegetation
from dataset import VegetationDataset

DATA_ROOT = Path(r"C:\Users\13519\data\train\vegetation\vegetation_dino")
CACHE_ROOT = Path(r"C:\Users\13519\data\train\vegetation\feature_cache")

BATCH_SIZE = 4
NUM_WORKERS = 0


@torch.no_grad()
def cache_split(split: str, model: DinoV3Vegetation, skip_empty=True):
    ds = VegetationDataset(DATA_ROOT / split)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    out_dir = CACHE_ROOT / split
    out_dir.mkdir(parents=True, exist_ok=True)

   

    model.eval()
    idx = 0
    n = len(ds)
    cached_count = 0

    for images, masks in loader:
        B = images.shape[0]
        H, W = images.shape[2], images.shape[3]

        feats = model.encode(images)

        for b in range(B):
            mask_b = masks[b]

            # Skip empty masks if requested
            if skip_empty and mask_b.sum() == 0:
                continue

            sample = {
                "feat": feats[b].clone(),
                "mask": mask_b.clone(),
                "hw": torch.tensor([H, W], dtype=torch.int32)
            }

            torch.save(sample, out_dir / f"{idx:06d}.pt")
            idx += 1
            cached_count += 1

        print(f"[{split}] processed {idx}/{n}", end="\r")

    print(f"[{split}] done: {cached_count}/{n} tiles cached to {out_dir} (skipped {n - cached_count} empty)")


def main():
    model = DinoV3Vegetation(freeze_encoder=True)

    # Cache val with ALL masks (keep empty for realistic evaluation)
    #cache_split("val", model, skip_empty=False)

    # Cache train WITHOUT empty masks (only train on positive examples)
    cache_split("train", model, skip_empty=True)


if __name__ == "__main__":
    main()