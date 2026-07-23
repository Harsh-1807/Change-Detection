"""
Script 1: Run inference on the full validation set and save predictions.
Handles single-band rasters by replicating to 3 channels.
"""
import os
import sys
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm

# ── CONFIG ──────────────────────────────────────────────────────────
# Change these paths for your setup / future runs
DATA_ROOT   = Path(r"C:\Users\13519\data\train\water_bodies")
CHECKPOINT  = Path(r"C:\Users\13519\first_version_change_detection\water_bodies_decoder_best.pth")
OUT_DIR     = Path(r"C:\Users\13519\Water_Change_Dection\predicted")
SPLIT       = "val"
THRESHOLD   = 0.5
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

# Add project root to path (edit if your repo location changes)
PROJECT_ROOT = Path(r"C:\Users\13519\first_version_change_detection")
sys.path.append(str(PROJECT_ROOT))

from model_dinov3 import DinoV3Vegetation
from dataset_water import WaterBodiesDataset
# ────────────────────────────────────────────────────────────────────


def load_model():
    model = DinoV3Vegetation(freeze_encoder=True)
    state = torch.load(CHECKPOINT, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model


@torch.no_grad()
def predict(model, image):
    logits = model(image.unsqueeze(0).to(DEVICE))
    probs = torch.sigmoid(logits).squeeze().cpu().numpy()
    return probs


def fix_single_band(image):
    """
    If image is single-band (1, H, W), replicate to 3 channels (3, H, W).
    If already 3-band, return as-is.
    """
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=0)
    elif image.ndim == 3 and image.shape[0] == 1:
        image = np.repeat(image, 3, axis=0)
    return image


def main():
    model = load_model()
    dataset = WaterBodiesDataset(DATA_ROOT / SPLIT, normalize_mode="imagenet")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for idx in tqdm(range(len(dataset)), desc="Inferencing"):
        try:
            image, mask = dataset[idx]
        except IndexError as e:
            # Fallback: handle single-band rasters
            print(f"  [idx={idx}] Band error, trying single-band fix...")
            # Manually load and fix
            import rasterio
            img_dir = DATA_ROOT / SPLIT / "images"
            mask_dir = DATA_ROOT / SPLIT / "masks"
            img_files = sorted(img_dir.glob("*.tif"))
            mask_files = sorted(mask_dir.glob("*.tif"))

            with rasterio.open(img_files[idx]) as src:
                img = src.read(1).astype(np.float32)  # read band 1
                img = fix_single_band(img)
                # Normalize imagenet
                mean = np.array([0.485, 0.456, 0.406]).reshape(-1, 1, 1)
                std  = np.array([0.229, 0.224, 0.225]).reshape(-1, 1, 1)
                img = (img / 255.0 - mean) / std
                image = torch.from_numpy(img).float()

            with rasterio.open(mask_files[idx]) as src:
                mask = torch.from_numpy(src.read(1).astype(np.float32))

        probs = predict(model, image)
        preds = (probs > THRESHOLD).astype(np.float32)

        # Save as .npz: keys = probs, preds, mask
        np.savez(
            OUT_DIR / f"sample_{idx:05d}.npz",
            probs=probs,
            preds=preds,
            mask=mask.squeeze().cpu().numpy() if hasattr(mask, 'cpu') else np.array(mask).squeeze(),
        )

    print(f"Saved predictions to {OUT_DIR}")


if __name__ == "__main__":
    main()