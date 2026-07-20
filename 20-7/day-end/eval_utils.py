"""Evaluation utilities — pure functions, no side effects."""

import numpy as np
import torch


def get_hw(hw):
    """Extract H, W from cached hw (handles tensor, list, tuple)."""
    if isinstance(hw, (list, tuple)):
        return int(hw[0][0]), int(hw[0][1])
    elif isinstance(hw, torch.Tensor):
        if hw.dim() == 2:
            return int(hw[0][0]), int(hw[0][1])
        else:
            return int(hw[0]), int(hw[1])
    else:
        return int(hw[0]), int(hw[1])


def compute_dice(pred_bin, mask):
    """Dice score. Both empty = 1.0, one empty = 0.0."""
    pred_sum = pred_bin.sum()
    mask_sum = mask.sum()
    if mask_sum == 0 and pred_sum == 0:
        return 1.0
    if mask_sum == 0 or pred_sum == 0:
        return 0.0
    intersection = (pred_bin * mask).sum()
    union = pred_sum + mask_sum
    return (2.0 * intersection + 1e-6) / (union + 1e-6)


def compute_iou(pred_bin, mask):
    """IoU score. Both empty = 1.0."""
    pred_sum = pred_bin.sum()
    mask_sum = mask.sum()
    if mask_sum == 0 and pred_sum == 0:
        return 1.0
    intersection = (pred_bin * mask).sum()
    union = pred_sum + mask_sum - intersection
    if union == 0:
        return 0.0
    return (intersection + 1e-6) / (union + 1e-6)


def compute_precision_recall_f1(pred_bin, mask):
    """Precision, Recall, F1."""
    tp = ((pred_bin == 1) & (mask == 1)).sum()
    fp = ((pred_bin == 1) & (mask == 0)).sum()
    fn = ((pred_bin == 0) & (mask == 1)).sum()
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    return float(precision), float(recall), float(f1)


def compute_pcc(pred_np, mask):
    """Pearson correlation. Returns None if either array is constant."""
    pred_flat = pred_np.flatten()
    mask_flat = mask.flatten()
    if pred_flat.std() > 1e-10 and mask_flat.std() > 1e-10:
        return float(np.corrcoef(pred_flat, mask_flat)[0, 1])
    return None


def evaluate_tile(sample, model, threshold):
    """Run model on one cached tile and return all metrics."""
    feat = sample["feat"].unsqueeze(0)
    mask = sample["mask"].squeeze().numpy()
    h, w = get_hw(sample["hw"])

    with torch.no_grad():
        logits = model.forward_from_features(feat, out_size=(h, w))

    pred_np = torch.sigmoid(logits).squeeze().cpu().numpy()
    pred_binary = (pred_np > threshold).astype(np.uint8)

    dice = compute_dice(pred_binary, mask)
    iou = compute_iou(pred_binary, mask)
    precision, recall, f1 = compute_precision_recall_f1(pred_binary, mask)
    pcc = compute_pcc(pred_np, mask)

    return {
        "mask_sum": int(mask.sum()),
        "is_empty": bool(mask.sum() == 0),
        "pred_sum": int(pred_binary.sum()),
        "pred_empty": bool(pred_binary.sum() == 0),
        "dice": float(dice),
        "iou": float(iou),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pcc": pcc,
        "pred_min": float(pred_np.min()),
        "pred_max": float(pred_np.max()),
        "pred_mean": float(pred_np.mean()),
        "pred_std": float(pred_np.std()),
    }


def print_stats(metrics_list, label):
    """Print summary statistics for a list of metric dicts."""
    if not metrics_list:
        print(f"\n{label}: No samples")
        return

    dice_vals = [m["dice"] for m in metrics_list]
    iou_vals = [m["iou"] for m in metrics_list]
    pcc_vals = [m["pcc"] for m in metrics_list if m["pcc"] is not None]
    prec_vals = [m["precision"] for m in metrics_list]
    rec_vals = [m["recall"] for m in metrics_list]
    f1_vals = [m["f1"] for m in metrics_list]

    print(f"\n{label} (n={len(metrics_list)}):")
    print(f"  Dice:      mean={np.mean(dice_vals):.4f}  std={np.std(dice_vals):.4f}")
    print(f"  IoU:       mean={np.mean(iou_vals):.4f}  std={np.std(iou_vals):.4f}")
    if pcc_vals:
        print(f"  PCC:       mean={np.mean(pcc_vals):.4f}  std={np.std(pcc_vals):.4f}  (n={len(pcc_vals)})")
    else:
        print(f"  PCC:       N/A (all constant masks)")
    print(f"  Precision: mean={np.mean(prec_vals):.4f}")
    print(f"  Recall:    mean={np.mean(rec_vals):.4f}")
    print(f"  F1:        mean={np.mean(f1_vals):.4f}")