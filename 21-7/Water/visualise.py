"""
Visualize water bodies segmentation: individual plots + metrics.
"""

import random
import shutil
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter

import sys
sys.path.append(r"C:\Users\13519\first_version_change_detection")

from model_dinov3 import DinoV3Vegetation
from dataset_water import WaterBodiesDataset

# ── CONFIG ──────────────────────────────────────────────────────────
DATA_ROOT = Path(r"C:\Users\13519\data\train\water_bodies")
CHECKPOINT = Path(r"C:\Users\13519\first_version_change_detection\water_bodies_decoder_best.pth")
OUT_DIR = Path(r"C:\Users\13519\Water_Change_Dection\visualizations")
SPLIT = "val"
NUM_SAMPLES = 10
THRESHOLD = 0.5
DEVICE = "cpu"
# ────────────────────────────────────────────────────────────────────


def load_model():
    model = DinoV3Vegetation(freeze_encoder=True)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


@torch.no_grad()
def predict(model, image):
    logits = model(image.unsqueeze(0).to(DEVICE))
    return torch.sigmoid(logits).squeeze().cpu().numpy()


def pcc(pred, target):
    """Pearson Correlation Coefficient."""
    p, t = pred.flatten(), target.flatten()
    if p.std() == 0 or t.std() == 0:
        return 0.0
    return np.corrcoef(p, t)[0, 1]


def fss(pred, target, window=16):
    """Fractions Skill Score (windowed spatial correlation)."""
    p = gaussian_filter(pred.astype(float), sigma=window/4)
    t = gaussian_filter(target.astype(float), sigma=window/4)
    p_bins = (p > p.mean()).astype(float)
    t_bins = (t > t.mean()).astype(float)
    mse = np.mean((p_bins - t_bins) ** 2)
    mse_ref = np.mean(p_bins**2) + np.mean(t_bins**2)
    return 1 - mse / max(mse_ref, 1e-8)


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
    rec = tp / max(tp + fn, 1e-8)
    return prec, rec


def kde_gaussian(pred, target, ax):
    """KDE overlay of prediction vs target pixel distributions."""
    p, t = pred.flatten(), target.flatten()
    # Sample for speed
    idx = np.random.choice(len(p), min(5000, len(p)), replace=False)
    p_s, t_s = p[idx], t[idx]
    
    x = np.linspace(0, 1, 200)
    if len(np.unique(p_s)) > 1:
        kde_p = gaussian_kde(p_s)
        ax.fill_between(x, kde_p(x), alpha=0.4, color='blue', label='Pred')
    if len(np.unique(t_s)) > 1:
        kde_t = gaussian_kde(t_s)
        ax.fill_between(x, kde_t(x), alpha=0.4, color='red', label='Target')
    ax.set_xlim(0, 1)
    ax.set_title("KDE Gaussian")
    ax.legend()


def visualize_sample(model, dataset, idx, out_dir):
    image, mask = dataset[idx]
    
    # Denormalize
    img_np = image.cpu().numpy()
    mean = np.array([0.485, 0.456, 0.406]).reshape(-1, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(-1, 1, 1)
    img_display = np.clip((img_np * std + mean).transpose(1, 2, 0), 0, 1)
    
    mask_np = mask.squeeze().cpu().numpy()
    probs = predict(model, image)
    preds = (probs > THRESHOLD).astype(np.float32)
    
    # Metrics
    d = dice(preds, mask_np)
    i = iou(preds, mask_np)
    p, r = precision_recall(preds, mask_np)
    c = pcc(probs, mask_np)
    f = fss(probs, mask_np)
    
    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    
    axes[0, 0].imshow(img_display)
    axes[0, 0].set_title(f"Image [{idx}]")
    axes[0, 0].axis("off")
    
    axes[0, 1].imshow(mask_np, cmap="Blues", vmin=0, vmax=1)
    axes[0, 1].set_title("Ground Truth")
    axes[0, 1].axis("off")
    
    axes[0, 2].imshow(probs, cmap="turbo", vmin=0, vmax=1)
    axes[0, 2].set_title(f"Probability")
    axes[0, 2].axis("off")
    
    axes[1, 0].imshow(img_display)
    axes[1, 0].imshow(preds, cmap="Reds", alpha=0.5)
    axes[1, 0].set_title(f"Prediction (thr={THRESHOLD})")
    axes[1, 0].axis("off")
    
    # Error map: FN=yellow, FP=green
    fn = (mask_np == 1) & (preds == 0)
    fp = (mask_np == 0) & (preds == 1)
    axes[1, 1].imshow(img_display)
    axes[1, 1].imshow(fn, cmap="Wistia", vmin=0, vmax=1)
    axes[1, 1].imshow(fp, cmap="Greens", vmin=0, vmax=1)
    axes[1, 1].set_title("Errors: yellow=FN, green=FP")
    axes[1, 1].axis("off")
    
    # KDE
    kde_gaussian(probs, mask_np, axes[1, 2])
    
    # Metrics text
    fig.suptitle(
        f"Dice={d:.3f} | IoU={i:.3f} | Prec={p:.3f} | Rec={r:.3f} | PCC={c:.3f} | FSS={f:.3f}",
        fontsize=11, y=0.02
    )
    
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    out_path = out_dir / f"sample_{idx:04d}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    return {
        "idx": idx, "dice": d, "iou": i, "precision": p, "recall": r,
        "pcc": c, "fss": f
    }


def main():
    model = load_model()
    dataset = WaterBodiesDataset(DATA_ROOT / SPLIT, normalize_mode="imagenet")
    
    # Clean/create output dir
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    indices = random.sample(range(len(dataset)), min(NUM_SAMPLES, len(dataset)))
    
    print(f"Visualizing {len(indices)} samples to {OUT_DIR}")
    results = []
    for idx in indices:
        print(f"  Sample {idx}...", end=" ")
        metrics = visualize_sample(model, dataset, idx, OUT_DIR)
        results.append(metrics)
        print(f"Dice={metrics['dice']:.3f} PCC={metrics['pcc']:.3f} FSS={metrics['fss']:.3f}")
    
    # Summary
    print(f"\n{'='*50}")
    print("SUMMARY")
    for k in ["dice", "iou", "precision", "recall", "pcc", "fss"]:
        vals = [r[k] for r in results]
        print(f"  {k.upper():12s}: mean={np.mean(vals):.4f}  std={np.std(vals):.4f}")
    print(f"{'='*50}")
    print(f"Saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()