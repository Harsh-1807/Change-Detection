"""Configuration constants."""

from pathlib import Path
import numpy as np

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

CACHE_ROOT = Path(r"C:\Users\13519\data\train\vegetation\feature_cache\val")
RGB_ROOT = r"C:\Users\13519\data\train\vegetation\vegetation_dino\val"
MODEL_PATH = r"C:\Users\13519\first_version_change_detection\vegetation_decoder_best_v3.pth"
OUT_ROOT = Path(r"C:\Users\13519\first_version_change_detection\Images_v3_all")

THRESHOLD = 0.40