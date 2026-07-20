"""Mosaic building logic."""

import numpy as np
import torch

from eval_utils import get_hw


def build_mosaic(sample_files, model, threshold):
    """Build GT, prediction, and error mosaics from all tiles."""
    first_sample = torch.load(sample_files[0], weights_only=True)
    first_mask = first_sample["mask"].squeeze().numpy()
    tile_h, tile_w = first_mask.shape

    n_tiles = len(sample_files)
    grid_cols = int(np.ceil(np.sqrt(n_tiles)))
    grid_rows = int(np.ceil(n_tiles / grid_cols))

    print(f"Mosaic: {grid_rows} rows x {grid_cols} cols for {n_tiles} tiles")

    mosaic_gt = np.zeros((grid_rows * tile_h, grid_cols * tile_w), dtype=np.uint8)
    mosaic_pred = np.zeros((grid_rows * tile_h, grid_cols * tile_w), dtype=np.uint8)
    mosaic_diff = np.zeros((grid_rows * tile_h, grid_cols * tile_w, 3), dtype=np.uint8)

    for idx, sample_file in enumerate(sample_files):
        sample = torch.load(sample_file, weights_only=True)
        mask = sample["mask"].squeeze().numpy()
        h, w = get_hw(sample["hw"])
        feat = sample["feat"].unsqueeze(0)

        with torch.no_grad():
            logits = model.forward_from_features(feat, out_size=(h, w))

        pred_np = torch.sigmoid(logits).squeeze().cpu().numpy()
        pred_binary = (pred_np > threshold).astype(np.uint8)

        row = idx // grid_cols
        col = idx % grid_cols
        y1, y2 = row * tile_h, (row + 1) * tile_h
        x1, x2 = col * tile_w, (col + 1) * tile_w

        mosaic_gt[y1:y2, x1:x2] = (mask * 255).astype(np.uint8)
        mosaic_pred[y1:y2, x1:x2] = (pred_binary * 255).astype(np.uint8)

        tp = (pred_binary == 1) & (mask == 1)
        fp = (pred_binary == 1) & (mask == 0)
        fn = (pred_binary == 0) & (mask == 1)
        mosaic_diff[y1:y2, x1:x2, 0] = (fp * 255).astype(np.uint8)
        mosaic_diff[y1:y2, x1:x2, 1] = (tp * 255).astype(np.uint8)
        mosaic_diff[y1:y2, x1:x2, 2] = (fn * 255).astype(np.uint8)

    return mosaic_gt, mosaic_pred, mosaic_diff