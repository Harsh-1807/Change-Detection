"""
Fast decoder training on CPU, using cached encoder features.

Run cache_features.py first. This script never touches the ViT encoder —
it only trains the small conv decoder on precomputed features, which is
cheap enough to iterate on quickly even on CPU.

When to stop using this script and do something else:
  Full encoder fine-tuning (unfreezing the ViT, like your old cell 45) is
  NOT practical on CPU. A ViT-S/16 forward+backward pass at 512x512 is
  already the slow part even frozen; doing it every step, for every tile,
  across 100 epochs, is realistically a multi-day-to-week job on CPU, not
  hours. If you want to fine-tune the encoder itself, that's the point to
  find GPU access (even a free Colab/Kaggle session) rather than push this
  further on CPU — decoder-only training is the CPU-appropriate stopping
  point.
"""

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model_dinov3 import DinoV3Vegetation

CACHE_ROOT = Path(r"C:\Users\13519\data\train\vegetation\feature_cache")
OUT_PATH = Path(r"C:\Users\13519\first_version_change_detection\vegetation_decoder_best.pth")

BATCH_SIZE = 8
EPOCHS = 500
LR = 5e-4
POS_WEIGHT = 25.0
PATIENCE = 25


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
        loss = (
            0.3 * bce_fn(logits, masks)
            +
            0.7 * dice_loss(logits, masks)
        )

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

    model = DinoV3Vegetation(
        freeze_encoder=True
    )

    # -----------------------------
    # Resume from existing checkpoint
    # -----------------------------

    best_dice = 0.0
    start_epoch = 1
    epochs_no_improve = 0

    if OUT_PATH.exists():

        print(
            f"\nLoading existing checkpoint:\n{OUT_PATH}\n"
        )

        state_dict = torch.load(
            OUT_PATH,
            map_location="cpu"
        )

        model.load_state_dict(
            state_dict
        )

        print(
            "Checkpoint loaded successfully."
        )

    # -----------------------------
    # Loss / Optimizer
    # -----------------------------

    bce_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([POS_WEIGHT])
    )

    optimizer = torch.optim.AdamW(
        model.decoder.parameters(),
        lr=LR
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2
    )

    # -----------------------------
    # Evaluate existing checkpoint
    # -----------------------------

    if OUT_PATH.exists():

        _, best_dice = evaluate(
            model,
            val_loader,
            bce_fn
        )

        print(
            f"Starting from existing model "
            f"(val_dice={best_dice:.4f})"
        )

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0

        for feats, masks, hw in train_loader:
            h, w = int(hw[0][0]), int(hw[1][0])
            optimizer.zero_grad()
            logits = model.forward_from_features(feats, out_size=(h, w))
            loss = (
                0.3 * bce_fn(logits, masks)
                +
                0.7 * dice_loss(logits, masks)
            )
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