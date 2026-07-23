"""
Script 2: Visualize a random sample from the predicted folder.
Shows original image, ground truth, probability map, prediction, error map, and metrics.
"""
import random
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import rasterio

# ── CONFIG ──────────────────────────────────────────────────────────
PRED_DIR     = Path(r"C:\Users\13519\Water_Change_Dection\predicted")
OUT_PATH     = Path(r"C:\Users\13519\Water_Change_Dection\validation\random_sample_2.png")
THRESHOLD    = 0.5
# Point to your original validation images folder
IMG_DIR      = Path(r"C:\Users\13519\data\train\water_bodies\val\images")
# Fallback: if masks not in .npz, point to GT mask folder
GT_MASK_DIR  = Path(r"C:\Users\13519\data\train\water_bodies\val\masks")
# ────────────────────────────────────────────────────────────────────


def dice(pred, target):
    inter = (pred * target).sum()
    return (2 * inter + 1e-6) / (pred.sum() + target.sum() + 1e-6)


def iou(pred, target):
    inter = (pred * target).sum()
    return inter / (pred.sum() + target.sum() - inter + 1e-6)


def precision_recall(pred, target):
    tp = (pred * target).sum()
    fp = (pred * (1 - target)).sum()
    fn = ((1 - pred) * target).sum()
    prec = tp / max(tp + fp, 1e-8)
    rec  = tp / max(tp + fn, 1e-8)
    return prec, rec


def pcc(probs, target):
    p, t = probs.flatten(), target.flatten()
    if p.std() == 0 or t.std() == 0:
        return 0.0
    return np.corrcoef(p, t)[0, 1]


def fss(probs, target, window=16):
    from scipy.ndimage import gaussian_filter
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
    mask_files = sorted(GT_MASK_DIR.glob("*.tif"))
    with rasterio.open(mask_files[idx]) as src:
        return src.read(1).astype(np.float32)


def load_original_image(fpath, img_dir):
    """
    Load the original image that corresponds to the .npz prediction file.
    Matches by filename: sample_00005.npz -> looks for image file with '00005' in name.
    """
    if img_dir is None or not img_dir.exists():
        return None

    # Extract index from sample_00005.npz -> "00005"
    idx_str = fpath.stem.split("_")[1]  # "00005"

    # Try to find matching image file
    img_files = sorted(img_dir.glob("*.tif")) + sorted(img_dir.glob("*.png")) + sorted(img_dir.glob("*.jpg"))

    # Strategy 1: filename contains the index number
    for img_file in img_files:
        if idx_str in img_file.stem:
            return read_image(img_file)

    # Strategy 2: match by position (same index in sorted list)
    try:
        idx = int(idx_str)
        if idx < len(img_files):
            return read_image(img_files[idx])
    except (ValueError, IndexError):
        pass

    return None


def read_image(img_path):
    """Read image and normalize to [0,1] for display."""
    ext = img_path.suffix.lower()
    if ext in [".tif", ".tiff"]:
        with rasterio.open(img_path) as src:
            if src.count == 1:
                # Single band - grayscale
                img = src.read(1).astype(np.float32)
                img = (img - img.min()) / (img.max() - img.min() + 1e-8)
                img = np.stack([img, img, img], axis=2)
            else:
                # Multi-band
                img = src.read([1, 2, 3]).astype(np.float32)
                img = np.transpose(img, (1, 2, 0))
                img = (img - img.min()) / (img.max() - img.min() + 1e-8)
                img = np.clip(img, 0, 1)
        return img
    else:
        from PIL import Image
        img = np.array(Image.open(img_path).convert("RGB")).astype(np.float32) / 255.0
        return img


def visualize(probs, preds, mask, out_path, img=None):
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))

    # Row 0: Original Image, Ground Truth, Probability
    if img is not None:
        axes[0, 0].imshow(img)
    else:
        axes[0, 0].text(0.5, 0.5, "No original image found\nCheck IMG_DIR path", 
                        ha="center", va="center", fontsize=12, color="red")
    axes[0, 0].set_title("Original Image", fontsize=12, fontweight="bold")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(mask, cmap="Blues", vmin=0, vmax=1)
    axes[0, 1].set_title("Ground Truth Mask", fontsize=12, fontweight="bold")
    axes[0, 1].axis("off")

    im = axes[0, 2].imshow(probs, cmap="turbo", vmin=0, vmax=1)
    axes[0, 2].set_title("Probability Map", fontsize=12, fontweight="bold")
    axes[0, 2].axis("off")
    plt.colorbar(im, ax=axes[0, 2], fraction=0.046, pad=0.04)

    # Row 1: Prediction overlay, Error map, Metrics
    if img is not None:
        axes[1, 0].imshow(img)
    axes[1, 0].imshow(preds, cmap="Reds", alpha=0.5, vmin=0, vmax=1)
    axes[1, 0].set_title(f"Prediction Overlay (thr={THRESHOLD})", fontsize=12, fontweight="bold")
    axes[1, 0].axis("off")

    # Error map: FN=yellow, FP=green
    fn = (mask == 1) & (preds == 0)
    fp = (mask == 0) & (preds == 1)
    if img is not None:
        axes[1, 1].imshow(img, alpha=0.4)
    axes[1, 1].imshow(fn, cmap="Wistia", vmin=0, vmax=1, alpha=0.8)
    axes[1, 1].imshow(fp, cmap="Greens", vmin=0, vmax=1, alpha=0.6)
    axes[1, 1].set_title("Errors: Yellow=FN, Green=FP", fontsize=12, fontweight="bold")
    axes[1, 1].axis("off")

    # Metrics text panel
    d = dice(preds, mask)
    i = iou(preds, mask)
    p, r = precision_recall(preds, mask)
    c = pcc(probs, mask)
    f = fss(probs, mask)

    axes[1, 2].axis("off")
    axes[1, 2].text(
        0.1, 0.55,
        f"METRICS\n"
        f"{'='*22}\n"
        f"Dice      = {d:.4f}\n"
        f"IoU       = {i:.4f}\n"
        f"Precision = {p:.4f}\n"
        f"Recall    = {r:.4f}\n"
        f"PCC       = {c:.4f}\n"
        f"FSS       = {f:.4f}",
        fontsize=13, family="monospace", transform=axes[1, 2].transAxes,
        verticalalignment="center",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.3)
    )

    fig.suptitle(f"Random Sample Visualization", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved visualization to {out_path}")


def main():
    files = sorted(PRED_DIR.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files found in {PRED_DIR}")

    fpath = random.choice(files)
    idx = int(fpath.stem.split("_")[1])  # extract index from sample_00001.npz
    data = np.load(fpath)
    probs = data["probs"]
    preds = data["preds"]
    mask  = load_mask(fpath, idx)

    # Load original image
    img = load_original_image(fpath, IMG_DIR)
    if img is None:
        print(f"WARNING: Could not find original image for {fpath.name}")
        print(f"  Tried looking in: {IMG_DIR}")
        print(f"  Looking for files containing: '{fpath.stem.split('_')[1]}'")

    visualize(probs, preds, mask, OUT_PATH, img=img)


if __name__ == "__main__":
    main()