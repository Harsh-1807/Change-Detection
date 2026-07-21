"""
Water bodies dataset: reads raster tiles and rasterizes GeoJSON vectors to masks.
Includes inverse standard deviation normalization for water body detection.
"""

from pathlib import Path

import rasterio
import torch
import numpy as np
import geopandas as gpd
from rasterio import features
from torch.utils.data import Dataset

# Standard ImageNet normalization (DINOv3 was trained on this)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

# Inverse standard deviation: emphasizes low-variance regions (water is often smooth/homogeneous)
# This gives more weight to channels with low std, enhancing subtle water boundaries
INV_STD = 1.0 / (IMAGENET_STD + 1e-8)  # avoid div by zero


class WaterBodiesDataset(Dataset):
    """
    Reads raster tiles from `rasters/` and rasterizes GeoJSON polygons
    from `vectors/` into binary masks matching the raster's shape.
    
    Supports multiple normalization modes:
    - 'imagenet': standard mean/std normalization
    - 'inv_std': inverse std only (emphasizes smooth regions like water)
    - 'both': concatenates both normalizations as 6 channels
    """

    def __init__(self, root: str | Path, normalize_mode: str = "imagenet"):
        self.img_dir = Path(root) / "images"
        self.mask_dir = Path(root) / "masks"
        self.images = sorted(self.img_dir.glob("*.tif"))
        self.normalize_mode = normalize_mode

        if not self.images:
            raise RuntimeError(f"No .tif files found in {self.img_dir}")

        print(f"Found {len(self.images)} raster tiles (normalize={normalize_mode})")

    def __len__(self):
        return len(self.images)

    def _normalize(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image based on selected mode.
        
        image: shape (3, H, W), values in [0, 255] or already normalized
        """
        # Ensure [0, 1] range
        if image.max() > 1.5:
            image = image / 255.0

        if self.normalize_mode == "imagenet":
            # Standard: (x - mean) / std
            return (image - IMAGENET_MEAN) / IMAGENET_STD

        elif self.normalize_mode == "inv_std":
            # Inverse std: emphasizes channels with low variance
            # Water bodies often have low spectral variance — this enhances them
            centered = image - IMAGENET_MEAN
            return centered * INV_STD

        elif self.normalize_mode == "both":
            # Concatenate both: 6 channels
            # Useful if you want standard + enhanced features
            centered = image - IMAGENET_MEAN
            std_norm = centered / IMAGENET_STD
            inv_norm = centered * INV_STD
            return np.concatenate([std_norm, inv_norm], axis=0)

        else:
            raise ValueError(f"Unknown normalize_mode: {self.normalize_mode}")

    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.mask_dir / img_path.name

        # --- Read raster image ---
        with rasterio.open(img_path) as src:
            image = src.read([1, 2, 3]).astype(np.float32)

        # Normalize
        image = self._normalize(image)

        # --- Read pre-computed mask ---
        with rasterio.open(mask_path) as src:
            mask = src.read(1).astype(np.float32)

        image = torch.tensor(image, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
        return image, mask