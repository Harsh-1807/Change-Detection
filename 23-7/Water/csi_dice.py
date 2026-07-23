"""
Script 5: Compute Critical Success Index (CSI) and Dice coefficient
per-sample and as global averages across the validation set.

Ground truth is loaded from the .npz files (saved by Script 1).
If masks are missing from .npz, set GT_MASK_DIR to load them separately.
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# ── CONFIG ──────────────────────────────────────────────────────────
PRED_DIR     = Path(r"C:\Users\13519\Water_Change_Dection\predicted")
OUT_DIR      = Path(r"C:\Users\13519\Water_Change_Dection\metric_plots")
# Fallback: if masks not in .npz, point to GT mask folder (set to None if not needed)
GT_MASK_DIR  = Path(r"C:\Users\13519\data\train\water_bodies\val\masks")
# ────────────────────────────────────────────────────────────────────

OUT_DIR.mkdir(parents=True, exist_ok=True)


def dice(pred, target):
    inter = (pred * target).sum()
    return (2 * inter + 1e-6) / (pred.sum() + target.sum() + 1e-6)


def csi(pred, target):
    """Critical Success Index = TP / (TP + FN + FP)."""
    tp = (pred * target).sum()
    fn = ((1 - pred) * target).sum()
    fp = (pred * (1 - target)).sum()
    return tp / max(tp + fn + fp, 1e-8)


def load_mask(fpath, idx):
    """Load mask from .npz or fallback to GT_MASK_DIR."""
    data = np.load(fpath)
    if "mask" in data:
        return data["mask"]
    if GT_MASK_DIR is None:
        raise ValueError(f"No 'mask' in {fpath} and GT_MASK_DIR is not set!")
    mask_files = sorted(GT_MASK_DIR.glob("*.tif"))
    import rasterio
    with rasterio.open(mask_files[idx]) as src:
        return src.read(1).astype(np.float32)


def compute_metrics(pred_dir):
    files = sorted(pred_dir.glob("*.npz"))
    dice_vals = []
    csi_vals  = []
    for idx, fpath in enumerate(files):
        data = np.load(fpath)
        preds = data["preds"]
        mask  = load_mask(fpath, idx)
        dice_vals.append(dice(preds, mask))
        csi_vals.append(csi(preds, mask))
    return np.array(dice_vals), np.array(csi_vals)


def plot_metric_hist(values, metric_name, out_path, color="darkorange"):
    fig, ax = plt.subplots(figsize=(8, 5))

    counts, bins, _ = ax.hist(
        values, bins=30, range=(0, 1), color=color, edgecolor="black",
        alpha=0.7, label="Histogram"
    )

    mean_val = np.mean(values)
    ax.axvline(mean_val, color="green", linestyle="--", lw=2, label=f"Mean={mean_val:.4f}")

    ax.set_xlim(0, 1)
    ax.set_xlabel(metric_name.upper(), fontsize=12)
    ax.set_ylabel("Number of Samples", fontsize=12)
    ax.set_title(f"Distribution of {metric_name.upper()} across Validation Set", fontsize=13)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {metric_name.upper()} plot to {out_path}")


def main():
    print("Computing CSI and Dice across validation set...")
    dice_vals, csi_vals = compute_metrics(PRED_DIR)

    print("\n" + "=" * 50)
    print("CSI & DICE SUMMARY")
    print("=" * 50)
    print(f"  DICE  mean={np.mean(dice_vals):.4f}  std={np.std(dice_vals):.4f}")
    print(f"  CSI   mean={np.mean(csi_vals):.4f}  std={np.std(csi_vals):.4f}")
    print("=" * 50)

    plot_metric_hist(dice_vals, "dice", OUT_DIR / "hist_dice.png", color="coral")
    plot_metric_hist(csi_vals,  "csi",  OUT_DIR / "hist_csi.png",  color="teal")

    print(f"\nPlots saved to {OUT_DIR}")


if __name__ == "__main__":
    main()