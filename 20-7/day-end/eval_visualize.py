"""Visualize individual samples."""

import numpy as np
import torch

from eval_config import IMAGENET_MEAN, IMAGENET_STD
from eval_utils import get_hw, compute_dice
from visualize_utils import create_visualization


def visualize_sample(idx, sample_files, rgb_ds, model, threshold, out_dir, subfolder):
    """Create detailed visualization for one sample."""
    sample_file = sample_files[idx]
    sample = torch.load(sample_file, weights_only=True)
    feat = sample["feat"].unsqueeze(0)
    mask = sample["mask"].squeeze().numpy()
    h, w = get_hw(sample["hw"])

    rgb_image, _ = rgb_ds[idx]
    rgb = rgb_image.numpy()
    rgb = (rgb * IMAGENET_STD[:, None, None]) + IMAGENET_MEAN[:, None, None]
    rgb = np.clip(rgb, 0.0, 1.0)
    rgb = np.transpose(rgb, (1, 2, 0))

    with torch.no_grad():
        logits = model.forward_from_features(feat, out_size=(h, w))

    pred_np = torch.sigmoid(logits).squeeze().cpu().numpy()
    pred_binary = (pred_np > threshold).astype(np.uint8)
    dice = compute_dice(pred_binary, mask)

    sample_dir = out_dir / subfolder / f"dice{dice:.3f}_gt{mask.sum():.0f}"
    sample_dir.mkdir(parents=True, exist_ok=True)

    create_visualization(
        rgb=rgb,
        mask_np=mask,
        pred_np=pred_np,
        pred_binary=pred_binary,
        dice=dice,
        out_file=sample_dir / "visualization.png"
    )