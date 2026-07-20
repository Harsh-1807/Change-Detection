"""
Fast decoder training on CPU, using cached encoder features.

BEFORE RUNNING THIS:
  1. dataset.py must have the ImageNet normalization fix.
  2. Delete the old feature_cache/ folder and vegetation_decoder_best.pth —
     both were produced from un-normalized inputs and are stale.
  3. Rerun cache_features.py to regenerate features from the fixed dataset.

FRESH_START defaults to True on purpose: your previous checkpoint was
trained on bad features, so resuming from it would keep pulling you back
toward that bad optimum. Only set it False once you've done a real training
run on the corrected pipeline and want to continue improving it.
"""

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model_dinov3 import DinoV3Vegetation

CACHE_ROOT = Path(r"C:\Users\13519\data\train\vegetation\feature_cache")
OUT_PATH = Path(r"C:\Users\13519\first_version_change_detection\vegetation_decoder_best.pth")

BATCH_SIZE = 8
EPOCHS = 200
LR = 5e-4
PATIENCE = 15
FRESH_START = True  # set False only after a clean run on corrected features


class CachedFeatureDataset(Dataset):
    def __init__(self, cache_dir: Path):
        self.files = sorted(cache_dir.glob("*.pt"))
        if not self.files:
            raise RuntimeError(f"No cached features found in {cache_dir}. Run cache_features.py first.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        item = torch.load(self.files[idx])
        return item["feat"], item["mask"], item["hw"]


def compute_pos_weight(dataset: CachedFeatureDataset) -> float:
    """Derive BCE pos_weight from actual foreground/background pixel counts,
    instead of guessing. Scans cached masks only — no encoder cost."""
    pos_pixels, total_pixels = 0.0, 0.0
    for f in dataset.files:
        mask = torch.load(f)["mask"]
        pos_pixels += mask.sum().item()
        total_pixels += mask.numel()

    neg_pixels = total_pixels - pos_pixels
    frac = pos_pixels / total_pixels
    weight = neg_pixels / max(pos_pixels, 1.0)
    print(f"Foreground pixels: {frac * 100:.2f}%  ->  computed pos_weight = {weight:.2f}")
    return weight


def dice_loss(logits, target):
    pred = torch.sigmoid(logits)
    smooth = 1e-6
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    return 1 - (2 * intersection + smooth) / (union + smooth)


@torch.no_grad()
def evaluate(model, loader, bce_fn):
    model.eval()
    total_dice, total_loss, n = 0.0, 0.0, 0
    for feats, masks, hw in loader:
        h, w = int(hw[0][0]), int(hw[1][0])
        logits = model.forward_from_features(feats, out_size=(h, w))
        loss = 0.3 * bce_fn(logits, masks) + 0.7 * dice_loss(logits, masks)

        preds = (torch.sigmoid(logits) > 0.5).float()
        intersection = (preds * masks).sum(dim=(1, 2, 3))
        union = preds.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3))
        dice = (2 * intersection + 1e-6) / (union + 1e-6)

        total_dice += dice.mean().item()
        total_loss += loss.item()
        n += 1
    return total_loss / n, total_dice / n


def main():
    train_ds = CachedFeatureDataset(CACHE_ROOT / "train")
    val_ds = CachedFeatureDataset(CACHE_ROOT / "val")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    pos_weight = compute_pos_weight(train_ds)

    model = DinoV3Vegetation(freeze_encoder=True)

    best_dice = 0.0
    epochs_no_improve = 0

    if not FRESH_START and OUT_PATH.exists():
        print(f"\nResuming from checkpoint: {OUT_PATH}\n")
        model.load_state_dict(torch.load(OUT_PATH, map_location="cpu"))
    elif OUT_PATH.exists():
        print(f"\nFRESH_START=True: ignoring existing checkpoint at {OUT_PATH}\n")

    bce_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
    optimizer = torch.optim.AdamW(model.decoder.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    if not FRESH_START and OUT_PATH.exists():
        _, best_dice = evaluate(model, val_loader, bce_fn)
        print(f"Starting from existing model (val_dice={best_dice:.4f})")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0

        for feats, masks, hw in train_loader:
            h, w = int(hw[0][0]), int(hw[1][0])
            optimizer.zero_grad()
            logits = model.forward_from_features(feats, out_size=(h, w))
            loss = 0.3 * bce_fn(logits, masks) + 0.7 * dice_loss(logits, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
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
            print(f"  -> new best (dice={best_dice:.4f}), saved to {OUT_PATH}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

    print(f"Done. Best val Dice: {best_dice:.4f}")


if __name__ == "__main__":
    main()
