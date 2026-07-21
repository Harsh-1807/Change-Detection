"""
Vegetation dataset: reads raster tiles and rasterizes GeoJSON polygon vectors to masks.
"""

from pathlib import Path

import rasterio
import torch
import numpy as np
import geopandas as gpd
from rasterio import features
from shapely import make_valid
from shapely.geometry import box
from shapely.affinity import translate, scale
from torch.utils.data import Dataset

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


class VegetationDataset(Dataset):
    """
    Reads raster tiles from `images/` and rasterizes GeoJSON polygon
    vectors from `masks/` into binary masks.
    """

    def __init__(self, root: str | Path):
        self.img_dir = Path(root) / "images"
        self.mask_dir = Path(root) / "masks"
        self.images = sorted(self.img_dir.glob("*.tif"))

        if not self.images:
            raise RuntimeError(f"No .tif files found in {self.img_dir}")

        # Filter out images with missing or unreadable masks
        self.valid_images = []
        for img_path in self.images:
            mask_path = self.mask_dir / img_path.name
            if mask_path.exists():
                try:
                    with rasterio.open(mask_path) as src:
                        _ = src.read(1)
                    self.valid_images.append(img_path)
                except Exception as e:
                    print(f"  Skipping corrupt mask: {mask_path.name} ({e})")
            else:
                print(f"  Skipping missing mask: {mask_path.name}")

        self.images = self.valid_images
        print(f"Found {len(self.images)} valid raster tiles")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.mask_dir / img_path.name

        # Read raster image
        with rasterio.open(img_path) as src:
            image = src.read([1, 2, 3]).astype(np.float32)

        image /= 255.0
        image = (image - IMAGENET_MEAN) / IMAGENET_STD

        # Read pre-tiled mask
        mask = np.zeros((image.shape[1], image.shape[2]), dtype=np.float32)
        try:
            with rasterio.open(mask_path) as src:
                mask = src.read(1).astype(np.float32)
        except Exception as e:
            print(f"Warning: failed to read mask {mask_path.name}: {e}")

        image = torch.tensor(image, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
        return image, mask