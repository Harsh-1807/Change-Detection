"""
Script 3: KDE / histogram plots for FSS, PCC, and IoU across the validation set.
X-axis: metric value (0 to 1)
Y-axis: number of samples (density / count)

Ground truth is loaded from the .npz files (saved by Script 1).
If masks are missing from .npz, set GT_MASK_DIR to load them separately.
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter

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


def iou(pred, target):
    inter = (pred * target).sum()
    return inter / (pred.sum() + target.sum() - inter + 1e-6)


def pcc(probs, target):
    p, t = probs.flatten(), target.flatten()
    if p.std() == 0 or t.std() == 0:
        return 0.0
    return np.corrcoef(p, t)[0, 1]


def fss(probs, target, window=16):
    p = gaussian_filter(probs.astype(float), sigma=window / 4)
    t = gaussian_filter(target.astype(float), sigma=window / 4)
    p_bins = (p > p.mean()).astype(float)
    t_bins = (t > t.mean()).astype(float)
    mse = np.mean((p_bins - t_bins) ** 2)
    mse_ref = np.mean(p_bins ** 2) + np.mean(t_bins ** 2)
    return 1 - mse / max(mse_ref, 1e-8)


def load_mask(fpath, idx):
    """Load mask from .npz or fallback to GT_MASK_DIR."""
    data = np.load(fpath)
    if "mask" in data:
        return data["mask"]
    if GT_MASK_DIR is None:
        raise ValueError(f"No 'mask' in {fpath} and GT_MASK_DIR is not set!")
    # Load from external GT folder
    mask_files = sorted(GT_MASK_DIR.glob("*.tif"))
    import rasterio
    with rasterio.open(mask_files[idx]) as src:
        return src.read(1).astype(np.float32)


def compute_all_metrics(pred_dir):
    files = sorted(pred_dir.glob("*.npz"))
    metrics = {"fss": [], "pcc": [], "iou": []}
    for idx, fpath in enumerate(files):
        data = np.load(fpath)
        probs = data["probs"]
        preds = data["preds"]
        mask  = load_mask(fpath, idx)
        metrics["fss"].append(fss(probs, mask))
        metrics["pcc"].append(pcc(probs, mask))
        metrics["iou"].append(iou(preds, mask))
    return metrics


def plot_kde_hist(values, metric_name, out_path, color="steelblue"):
    fig, ax = plt.subplots(figsize=(8, 5))

    counts, bins, _ = ax.hist(
        values, bins=30, range=(0, 1), color=color, edgecolor="black",
        alpha=0.6, label="Histogram"
    )

    x = np.linspace(0, 1, 500)
    if len(np.unique(values)) > 1:
        kde = gaussian_kde(values)
        kde_vals = kde(x)
        scale = counts.sum() * (bins[1] - bins[0])
        ax.plot(x, kde_vals * scale, color="darkred", lw=2, label="KDE")

    ax.set_xlim(0, 1)
    ax.set_xlabel(metric_name.upper(), fontsize=12)
    ax.set_ylabel("Number of Samples", fontsize=12)
    ax.set_title(f"Distribution of {metric_name.upper()} across Validation Set", fontsize=13)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    mean_val = np.mean(values)
    ax.axvline(mean_val, color="green", linestyle="--", lw=2, label=f"Mean={mean_val:.3f}")
    ax.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {metric_name.upper()} plot to {out_path}")


def main():
    print("Computing metrics across validation set...")
    metrics = compute_all_metrics(PRED_DIR)

    for metric_name, values in metrics.items():
        out_path = OUT_DIR / f"kde_{metric_name}.png"
        plot_kde_hist(values, metric_name, out_path)

    print(f"\nAll plots saved to {OUT_DIR}")


if __name__ == "__main__":
    main()