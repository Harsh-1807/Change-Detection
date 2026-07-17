from pathlib import Path

import torch

from dataset import VegetationDataset
from model_dinov3 import DinoV3Vegetation

from visualize_utils import (
    compute_stats,
    create_visualization
)

# ==================================================
# PATHS
# ==================================================

CACHE_ROOT = Path(
    r"C:\Users\13519\data\train\vegetation\feature_cache\val"
)

RGB_ROOT = (
    r"C:\Users\13519\data\train\vegetation\vegetation_dino\val"
)

MODEL_PATH = (
    r"C:\Users\13519\first_version_change_detection"
    r"\vegetation_decoder_best.pth"
)

OUT_ROOT = Path(
    r"C:\Users\13519\first_version_change_detection\Images"
)

OUT_ROOT.mkdir(
    exist_ok=True
)

# ==================================================
# MODEL
# ==================================================

model = DinoV3Vegetation(
    freeze_encoder=True
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location="cpu"
    )
)

model.eval()

# ==================================================
# DATA
# ==================================================

rgb_ds = VegetationDataset(
    RGB_ROOT
)

sample_files = sorted(
    CACHE_ROOT.glob("*.pt")
)

# ==================================================
# GENERATE 10 VISUALIZATIONS
# ==================================================
import numpy as np

# ==================================================
# THRESHOLD SWEEP
# ==================================================

thresholds = [
    0.01,
    0.03,
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.40,
    0.50,
]

print("\n" + "=" * 60)
print("THRESHOLD SWEEP")
print("=" * 60)

best_threshold = None
best_dice = -1

for threshold in thresholds:

    dices = []

    for sample_file in sample_files:

        sample = torch.load(sample_file)

        feat = sample["feat"].unsqueeze(0)

        mask = (
            sample["mask"]
            .squeeze()
            .numpy()
        )

        with torch.no_grad():

            logits = model.forward_from_features(
                feat,
                out_size=(512, 512)
            )

        pred_np = (
            torch.sigmoid(logits)
            .squeeze()
            .cpu()
            .numpy()
        )

        pred_bin = (
            pred_np > threshold
        ).astype(np.uint8)

        intersection = (
            pred_bin * mask
        ).sum()

        union = (
            pred_bin.sum()
            +
            mask.sum()
        )

        dice = (
            2 * intersection + 1e-6
        ) / (
            union + 1e-6
        )

        dices.append(dice)

    mean_dice = np.mean(dices)

    print(
        f"Threshold {threshold:0.2f}  "
        f"Mean Dice = {mean_dice:.4f}"
    )

    if mean_dice > best_dice:

        best_dice = mean_dice
        best_threshold = threshold

print("\n" + "=" * 60)
print(
    f"BEST THRESHOLD = {best_threshold:.2f}"
)
print(
    f"BEST DICE      = {best_dice:.4f}"
)
print("=" * 60)

for idx in range(
    min(20, len(sample_files))
):

    print(
        f"Processing sample {idx}"
    )

    sample = torch.load(
        sample_files[idx]
    )

    feat = sample["feat"].unsqueeze(0)

    mask = (
        sample["mask"]
        .squeeze()
        .numpy()
    )

    rgb_image, _ = rgb_ds[idx]

    rgb = (
        rgb_image
        .permute(1, 2, 0)
        .numpy()
    )

    with torch.no_grad():

        logits = model.forward_from_features(
            feat,
            out_size=(512, 512)
        )

    pred_np = (
        torch.sigmoid(logits)
        .squeeze()
        .cpu()
        .numpy()
    )

    pred_binary, dice = compute_stats(
        pred_np,
        mask
    )

    sample_dir = (
        OUT_ROOT /
        f"sample_{idx:03d}"
    )

    sample_dir.mkdir(
        exist_ok=True
    )

    out_file = (
        sample_dir /
        "visualization.png"
    )

    create_visualization(
        rgb=rgb,
        mask_np=mask,
        pred_np=pred_np,
        pred_binary=pred_binary,
        dice=dice,
        out_file=out_file
    )

    with open(
        sample_dir / "stats.txt",
        "w"
    ) as f:

        f.write(
            f"Dice@0.1 = {dice:.6f}\n"
        )

        f.write(
            f"Pred Min = {pred_np.min()}\n"
        )

        f.write(
            f"Pred Max = {pred_np.max()}\n"
        )

        f.write(
            f"Pred Mean = {pred_np.mean()}\n"
        )

        f.write(
            f"Pred Std = {pred_np.std()}\n"
        )

print(
    "\nDone."
)