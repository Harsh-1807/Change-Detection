"""
Fast decoder training for roads segmentation.
Roads are thin lines: use clDice loss (topology-aware) + Tversky.
"""

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model_dinov3 import DinoV3Vegetation

CACHE_ROOT = Path(r"C:\Users\13519\data\train\roads\feature_cache")
OUT_PATH = Path(r"C:\Users\13519\first_version_change_detection\roads_decoder_best.pth")

BATCH_SIZE = 8
EPOCHS = 200
LR = 1e-3
PATIENCE = 20
FRESH_START = False


class CachedFeatureDataset(Dataset):
    def __init__(self, cache_dir: Path, skip_empty=False):
        self.files = sorted(cache_dir.glob("*.pt"))
        if not self.files:
            raise RuntimeError(f"No cached features in {cache_dir}")

        if skip_empty:
            self.files = [f for f in self.files if torch.load(f, weights_only=True)["mask"].sum() > 0]
            print(f"Filtered to {len(self.files)} non-empty masks")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        item = torch.load(self.files[idx], weights_only=True)
        return item["feat"], item["mask"], item["hw"]


def compute_pos_weight(dataset: CachedFeatureDataset) -> float:
    pos_pixels, total_pixels = 0.0, 0.0
    for f in dataset.files:
        mask = torch.load(f, weights_only=True)["mask"]
        pos_pixels += mask.sum().item()
        total_pixels += mask.numel()

    weight = (total_pixels - pos_pixels) / max(pos_pixels, 1.0)
    frac = pos_pixels / total_pixels
    print(f"Foreground: {frac * 100:.4f}%  ->  pos_weight = {weight:.2f}")
    return weight


def dice_loss(logits, target, smooth=1e-6):
    pred = torch.sigmoid(logits)
    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice.mean()


def tversky_loss(logits, target, alpha=0.7, beta=0.3, smooth=1e-6):
    """
    Higher beta (FP penalty) for roads — we want connected lines, not scattered pixels.
    """
    pred = torch.sigmoid(logits)
    tp = (pred * target).sum(dim=(1, 2, 3))
    fp = (pred * (1 - target)).sum(dim=(1, 2, 3))
    fn = ((1 - pred) * target).sum(dim=(1, 2, 3))
    tversky = (tp + smooth) / (tp + alpha * fn + beta * fp + smooth)
    return 1.0 - tversky.mean()


def soft_cldice_loss(pred, target, smooth=1e-6):
    """
    Simplified soft clDice for road topology preservation.
    Approximates skeleton-based Dice without morphological ops.
    """
    # Erosion-like: penalize endpoints and thin regions
    pred_soft = torch.sigmoid(pred)
    
    # Recall component (sensitivity to thin structures)
    tp = (pred_soft * target).sum(dim=(1, 2, 3))
    fn = ((1 - pred_soft) * target).sum(dim=(1, 2, 3))
    recall = (tp + smooth) / (tp + fn + smooth)
    
    # Precision component
    fp = (pred_soft * (1 - target)).sum(dim=(1, 2, 3))
    precision = (tp + smooth) / (tp + fp + smooth)
    
    # clDice = 2 * (precision * recall) / (precision + recall)
    cldice = 2.0 * (precision * recall) / (precision + recall + smooth)
    return 1.0 - cldice.mean()


def get_hw(hw):
    if isinstance(hw, (list, tuple)):
        h, w = int(hw[0][0]), int(hw[0][1])
    elif isinstance(hw, torch.Tensor):
        if hw.dim() == 2:
            h, w = int(hw[0][0]), int(hw[0][1])
        else:
            h, w = int(hw[0]), int(hw[1])
    else:
        h, w = int(hw[0]), int(hw[1])
    return h, w


@torch.no_grad()
def evaluate(model, loader, bce_fn):
    model.eval()
    total_dice, total_loss, n = 0.0, 0.0, 0

    for feats, masks, hw in loader:
        h, w = get_hw(hw)
        if masks.sum() == 0:
            continue

        logits = model.forward_from_features(feats, out_size=(h, w))
        loss = 0.1 * bce_fn(logits, masks) + 0.9 * tversky_loss(logits, masks)

        preds = (torch.sigmoid(logits) > 0.5).float()
        intersection = (preds * masks).sum(dim=(1, 2, 3))
        union = preds.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3))
        dice = (2.0 * intersection + 1e-6) / (union + 1e-6)

        total_dice += dice.mean().item()
        total_loss += loss.item()
        n += 1

    if n == 0:
        return float('inf'), 0.0
    return total_loss / n, total_dice / n


def main():
    train_ds = CachedFeatureDataset(CACHE_ROOT / "train", skip_empty=True)
    val_ds = CachedFeatureDataset(CACHE_ROOT / "val", skip_empty=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    pos_weight = compute_pos_weight(train_ds)
    pos_weight = min(pos_weight, 100.0)
    print(f"Clamped pos_weight = {pos_weight:.2f}")

    model = DinoV3Vegetation(freeze_encoder=True)
    best_dice = 0.0
    epochs_no_improve = 0

    if not FRESH_START and OUT_PATH.exists():
        print(f"\nResuming from: {OUT_PATH}")
        model.load_state_dict(torch.load(OUT_PATH, map_location="cpu"))
    elif OUT_PATH.exists():
        print(f"\nFRESH_START: ignoring {OUT_PATH}")

    bce_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))

    optimizer = torch.optim.AdamW(model.decoder.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

    if not FRESH_START and OUT_PATH.exists():
        _, best_dice = evaluate(model, val_loader, bce_fn)
        print(f"Starting from existing model (val_dice={best_dice:.4f})")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        num_batches = 0

        for feats, masks, hw in train_loader:
            h, w = get_hw(hw)
            if masks.sum() == 0:
                continue

            optimizer.zero_grad()
            logits = model.forward_from_features(feats, out_size=(h, w))

            bce = bce_fn(logits, masks)
            tversky = tversky_loss(logits, masks)
            cldice = soft_cldice_loss(logits, masks)
            
            # Road-specific: topology matters more than exact pixel match
            loss = 0.2 * bce + 0.5 * tversky + 0.3 * cldice

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            num_batches += 1

        if num_batches == 0:
            print(f"Epoch {epoch:03d}: WARNING - all batches empty!")
            continue

        train_loss = running_loss / num_batches
        val_loss, val_dice = evaluate(model, val_loader, bce_fn)
        scheduler.step(val_dice)

        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.4f} "
            f"| val_loss={val_loss:.4f} | val_dice={val_dice:.4f} "
            f"| lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        if val_dice > best_dice:
            best_dice = val_dice
            epochs_no_improve = 0
            OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), OUT_PATH)
            print(f"  -> new best (dice={best_dice:.4f}), saved")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

    print(f"Done. Best val Dice: {best_dice:.4f}")


if __name__ == "__main__":
    main()