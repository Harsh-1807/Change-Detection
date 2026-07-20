"""Plotting functions — side effects (file I/O)."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


def plot_kde_simple(values, metric_name, out_path, color="#3498db"):
    """Plot a single clean KDE with mean line."""
    fig, ax = plt.subplots(figsize=(8, 5))

    if len(values) < 2 or np.std(values) < 1e-10:
        ax.text(0.5, 0.5, f"{metric_name}\nconstant = {np.mean(values):.3f}",
                ha="center", va="center", transform=ax.transAxes, fontsize=14)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    else:
        kde = gaussian_kde(values)
        x_range = np.linspace(max(0, min(values) - 0.05), min(1, max(values) + 0.05), 500)
        kde_vals = kde(x_range)
        ax.fill_between(x_range, kde_vals, alpha=0.3, color=color)
        ax.plot(x_range, kde_vals, color=color, linewidth=2)
        ax.axvline(np.mean(values), color="red", linestyle="--", linewidth=2,
                   label=f"mean = {np.mean(values):.3f}")
        ax.set_xlabel(metric_name, fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.legend(fontsize=11)

    ax.set_title(f"{metric_name} (Non-Empty, n={len(values)})", fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path.name}")


def create_dashboard(nonempty_metrics, empty_metrics, all_metrics, out_path, threshold):
    """Create 6-panel overview dashboard."""
    fig = plt.figure(figsize=(16, 12))

    dice_n = [m["dice"] for m in nonempty_metrics]
    iou_n = [m["iou"] for m in nonempty_metrics]
    rec_n = [m["recall"] for m in nonempty_metrics]
    prec_n = [m["precision"] for m in nonempty_metrics]

    ax1 = fig.add_subplot(2, 3, 1)
    ax1.hist(dice_n, bins=30, color="#2ecc71", alpha=0.7, edgecolor="black")
    ax1.axvline(np.mean(dice_n), color="red", linestyle="--", linewidth=2)
    ax1.set_title("Dice (Non-Empty)", fontsize=12)
    ax1.set_xlabel("Dice Score")
    ax1.set_ylabel("Count")
    ax1.set_xlim(0, 1)

    ax2 = fig.add_subplot(2, 3, 2)
    ax2.hist(iou_n, bins=30, color="#3498db", alpha=0.7, edgecolor="black")
    ax2.axvline(np.mean(iou_n), color="red", linestyle="--", linewidth=2)
    ax2.set_title("IoU (Non-Empty)", fontsize=12)
    ax2.set_xlabel("IoU Score")
    ax2.set_xlim(0, 1)

    ax3 = fig.add_subplot(2, 3, 3)
    ax3.hist(rec_n, bins=30, color="#e74c3c", alpha=0.7, edgecolor="black")
    ax3.axvline(np.mean(rec_n), color="red", linestyle="--", linewidth=2)
    ax3.set_title("Recall (Non-Empty)", fontsize=12)
    ax3.set_xlabel("Recall")
    ax3.set_xlim(0, 1)

    ax4 = fig.add_subplot(2, 3, 4)
    gt_sums = [m["gt_sum"] for m in nonempty_metrics]
    ax4.scatter(gt_sums, dice_n, alpha=0.4, s=15, color="#2ecc71")
    ax4.set_xlabel("GT Vegetation Pixels")
    ax4.set_ylabel("Dice Score")
    ax4.set_title("Dice vs Vegetation Amount")
    ax4.set_xscale("log")
    ax4.grid(True, alpha=0.3)

    ax5 = fig.add_subplot(2, 3, 5)
    ax5.scatter(rec_n, prec_n, alpha=0.4, s=15, color="#9b59b6")
    ax5.set_xlabel("Recall")
    ax5.set_ylabel("Precision")
    ax5.set_title("Precision vs Recall")
    ax5.set_xlim(0, 1)
    ax5.set_ylim(0, 1)
    ax5.grid(True, alpha=0.3)

    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis("off")

    summary_text = f"""MODEL PERFORMANCE SUMMARY

Total Tiles: {len(all_metrics)}
  Non-Empty: {len(nonempty_metrics)}
  Empty:     {len(empty_metrics)}

NON-EMPTY TILES:
  Dice:        {np.mean(dice_n):.4f} ± {np.std(dice_n):.4f}
  IoU:         {np.mean(iou_n):.4f} ± {np.std(iou_n):.4f}
  Recall:      {np.mean(rec_n):.4f} ± {np.std(rec_n):.4f}
  Precision:   {np.mean(prec_n):.4f} ± {np.std(prec_n):.4f}

EMPTY TILES:
  Dice:        {np.mean([m['dice'] for m in empty_metrics]):.4f}

Threshold: {threshold}
"""

    ax6.text(0.1, 0.5, summary_text, transform=ax6.transAxes, fontsize=11,
             verticalalignment="center", fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.suptitle("Vegetation Segmentation — Validation Overview", fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path.name}")


def save_mosaic_plot(mosaic_gt, mosaic_pred, mosaic_diff, n_tiles, out_path):
    """Save mosaic overview as image."""
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    axes[0].imshow(mosaic_gt, cmap='gray', vmin=0, vmax=255)
    axes[0].set_title("Ground Truth Mosaic", fontsize=14)
    axes[0].axis('off')

    axes[1].imshow(mosaic_pred, cmap='gray', vmin=0, vmax=255)
    axes[1].set_title("Prediction Mosaic", fontsize=14)
    axes[1].axis('off')

    axes[2].imshow(mosaic_diff)
    axes[2].set_title("Error Map (Green=TP, Red=FP, Blue=FN)", fontsize=14)
    axes[2].axis('off')

    plt.suptitle(f"Full Validation Overview — {n_tiles} tiles", fontsize=16)
    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path.name}")