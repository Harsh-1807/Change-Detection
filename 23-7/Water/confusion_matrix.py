"""
Script 4: Confusion matrix and classification metrics (Acc, Precision, Recall, F1).
Aggregates pixel-level counts across the full validation set.

Ground truth is loaded from the .npz files (saved by Script 1).
If masks are missing from .npz, set GT_MASK_DIR to load them separately.
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# ── CONFIG ──────────────────────────────────────────────────────────
PRED_DIR     = Path(r"C:\Users\13519\Water_Change_Dection\predicted")
OUT_PATH     = Path(r"C:\Users\13519\Water_Change_Dection\validation\confusion_matrix.png")
# Fallback: if masks not in .npz, point to GT mask folder (set to None if not needed)
GT_MASK_DIR  = Path(r"C:\Users\13519\data\train\water_bodies\val\masks")
# ────────────────────────────────────────────────────────────────────


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


def compute_global_cm(pred_dir):
    files = sorted(pred_dir.glob("*.npz"))
    all_preds = []
    all_masks = []
    for idx, fpath in enumerate(files):
        data = np.load(fpath)
        all_preds.append(data["preds"].flatten())
        all_masks.append(load_mask(fpath, idx).flatten())
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_masks)
    return y_true, y_pred


def plot_confusion_matrix(cm, out_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Pastel2")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Non-Water", "Water"])
    ax.set_yticklabels(["Non-Water", "Water"])
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title("Confusion Matrix (Pixel-level)", fontsize=13)

    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}",
                    ha="center", va="center", color="black", fontsize=14)

    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved confusion matrix to {out_path}")


def main():
    print("Loading predictions...")
    y_true, y_pred = compute_global_cm(PRED_DIR)

    cm = confusion_matrix(y_true, y_pred)
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)

    print("\n" + "=" * 50)
    print("CLASSIFICATION REPORT (Pixel-level)")
    print("=" * 50)
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print("=" * 50)
    print(f"\nConfusion Matrix:\n{cm}")

    plot_confusion_matrix(cm, OUT_PATH)


if __name__ == "__main__":
    main()