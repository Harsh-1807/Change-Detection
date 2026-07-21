"""
Visualize road segmentation results.
"""

import random
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from skimage.morphology import skeletonize

from model_dinov3 import DinoV3Vegetation
from dataset_roads import RoadsDataset

DATA_ROOT = Path(r"C:\Users\13519\data\train\roads")
CHECKPOINT = Path(r"C:\Users\13519\first_version_change_detection\roads_decoder_best.pth")
SPLIT = "val"
NUM_SAMPLES = 8
THRESHOLD = 0.5
DEVICE = "cpu"


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


def visualize(model, dataset, num=8):
    indices = random.sample(range(len(dataset)), min(num, len(dataset)))
    n = len(indices)
    fig, axes = plt.subplots(n, 5, figsize=(20, 4 * n))
    if n == 1:
        axes = axes.reshape(1, -1)

    road_cmap = LinearSegmentedColormap.from_list("road", [(0, 0, 0, 0), (1, 0.3, 0, 0.6)])

    for row, idx in enumerate(indices):
        image, mask = dataset[idx]
        img_np = image.cpu().numpy()
        mean = np.array([0.485, 0.456, 0.406]).reshape(-1, 1, 1)
        std = np.array([0.229, 0.224, 0.225]).reshape(-1, 1, 1)
        img_display = np.clip((img_np * std + mean).transpose(1, 2, 0), 0, 1)

        mask_np = mask.squeeze().cpu().numpy()
        probs = predict(model, image)
        preds = (probs > THRESHOLD).astype(np.float32)

        # Skeletonize prediction for topology check
        skel = skeletonize(preds > 0.5).astype(np.float32)

        dice = (2 * (preds * mask_np).sum() + 1e-6) / (preds.sum() + mask_np.sum() + 1e-6)

        axes[row, 0].imshow(img_display)
        axes[row, 0].set_title(f"Image [{idx}]")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(mask_np, cmap="Reds", vmin=0, vmax=1)
        axes[row, 1].set_title("Ground Truth")
        axes[row, 1].axis("off")

        axes[row, 2].imshow(probs, cmap="turbo", vmin=0, vmax=1)
        axes[row, 2].set_title(f"Probability\nDice={dice:.3f}")
        axes[row, 2].axis("off")

        axes[row, 3].imshow(img_display)
        axes[row, 3].imshow(preds, cmap=road_cmap, vmin=0, vmax=1)
        axes[row, 3].set_title("Prediction Overlay")
        axes[row, 3].axis("off")

        axes[row, 4].imshow(img_display)
        axes[row, 4].imshow(skel, cmap="hot", vmin=0, vmax=1)
        axes[row, 4].set_title("Skeleton (topology)")
        axes[row, 4].axis("off")

    plt.tight_layout()
    plt.savefig("roads_predictions.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved to roads_predictions.png")


def main():
    model = load_model()
    dataset = RoadsDataset(DATA_ROOT / SPLIT, road_buffer_meters=5.0)
    visualize(model, dataset, NUM_SAMPLES)


if __name__ == "__main__":
    main()