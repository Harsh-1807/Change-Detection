from pathlib import Path
import rasterio
import torch
import numpy as np
from torch.utils.data import Dataset

# Required for DINOv3 LVD-1689M pretrained weights (per facebookresearch/dinov3):
# standard ImageNet eval transform, applied AFTER scaling pixels to [0, 1].
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


class VegetationDataset(Dataset):
    def __init__(self, root):
        self.img_dir = Path(root) / "images"
        self.mask_dir = Path(root) / "masks"
        self.images = sorted(self.img_dir.glob("*.tif"))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.mask_dir / img_path.name

        with rasterio.open(img_path) as src:
            image = src.read([1, 2, 3]).astype(np.float32)

        image /= 255.0
        image = (image - IMAGENET_MEAN) / IMAGENET_STD  

        with rasterio.open(mask_path) as src:
            mask = src.read(1).astype(np.float32)

        image = torch.tensor(image, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
        return image, mask