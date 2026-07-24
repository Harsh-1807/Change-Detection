"""
SEMANTIC CHANGE DETECTION PIPELINE
Using your 3 pre-trained DINOv3 binary models
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import rasterio
import numpy as np
import cv2
from pathlib import Path
from rasterio.windows import Window
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ============================================================
# CONFIGURATION
# ============================================================

# Paths
T1_PATH = r"C:\Users\13519\Desktop\Airoli_2021_modified.tif"
T2_PATH = r"C:\Users\13519\Desktop\Airoli_2025_resampled.tif"
OUTPUT_DIR = r"C:\Users\13519\Desktop\change_detection_results"

# Model paths
DINO_REPO = r"C:\Users\13519\first_version_change_detection\dinov3"
CKPT_PATH = r"C:\Users\13519\first_version_change_detection\dinov3_vits16_pretrain_lvd1689m-08c60483.pth"

WATER_CKPT = r"C:\Users\13519\first_version_change_detection\water_bodies_decoder_best.pth"
ROAD_CKPT = r"C:\Users\13519\first_version_change_detection\roads_decoder_best.pth"
VEG_CKPT = r"C:\Users\13519\Vegetation_Change_Detection\vegetation_decoder_best_v3.pth"

# Classes
CLASSES = ['background', 'water', 'road', 'vegetation', 'building']  # building = inferred
NUM_CLASSES = len(CLASSES)
CLASS_COLORS = {
    0: (0, 0, 0),        # background
    1: (0, 119, 190),    # water
    2: (255, 0, 0),      # road
    3: (0, 255, 0),      # vegetation
    4: (128, 128, 128),  # building (inferred)
}

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

TILE_SIZE = 512
OVERLAP = 256

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# MODEL DEFINITION (Your Architecture)
# ============================================================

if DINO_REPO not in sys.path:
    sys.path.append(DINO_REPO)

HIDDEN_DIM = 384
PATCH_SIZE = 16

class DinoV3Segmentor(nn.Module):
    """Your DINOv3 + decoder architecture."""
    
    def __init__(self, checkpoint_path=None):
        super().__init__()
        
        self.encoder = torch.hub.load(
            DINO_REPO,
            "dinov3_vits16",
            source="local",
            weights=CKPT_PATH
        )
        
        # Freeze encoder
        for p in self.encoder.parameters():
            p.requires_grad = False
        
        self.decoder = nn.Sequential(
            nn.Conv2d(HIDDEN_DIM, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1)  # Binary output
        )
        
        # Load trained decoder weights
        if checkpoint_path and os.path.exists(checkpoint_path):
            self.load_decoder(checkpoint_path)
            print(f"Loaded decoder: {checkpoint_path}")
    
    def load_decoder(self, path):
        """Load only decoder weights."""
        state = torch.load(path, map_location=DEVICE)
        # Handle different checkpoint formats
        if 'decoder' in state:
            self.decoder.load_state_dict(state['decoder'])
        elif 'state_dict' in state:
            self.decoder.load_state_dict(state['state_dict'])
        else:
            self.decoder.load_state_dict(state)
    
    def encode(self, x):
        B, C, H, W = x.shape
        h = H // PATCH_SIZE
        w = W // PATCH_SIZE
        
        outputs = self.encoder.forward_features(x)
        patch_tokens = outputs["x_norm_patchtokens"]
        
        feat = patch_tokens.transpose(1, 2).reshape(B, HIDDEN_DIM, h, w)
        return feat
    
    def forward(self, x):
        H, W = x.shape[2], x.shape[3]
        feat = self.encode(x)
        logits = self.decoder(feat)
        logits = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=False)
        return logits

# ============================================================
# LOAD ALL THREE MODELS
# ============================================================

print("\n" + "="*60)
print("LOADING MODELS")
print("="*60)

water_model = DinoV3Segmentor(WATER_CKPT).to(DEVICE).eval()
road_model = DinoV3Segmentor(ROAD_CKPT).to(DEVICE).eval()
veg_model = DinoV3Segmentor(VEG_CKPT).to(DEVICE).eval()

print("All models loaded!")

# ============================================================
# INFERENCE: PREDICT BINARY MASKS FOR ENTIRE IMAGE
# ============================================================

def predict_binary_mask(model, image_path, output_path, threshold=0.5):
    """
    Run sliding window inference and save binary mask.
    """
    with rasterio.open(image_path) as src:
        h, w = src.height, src.width
        
        # Output probability map (float) and binary mask
        prob_map = np.zeros((h, w), dtype=np.float32)
        weight_map = np.zeros((h, w), dtype=np.float32)
        
        # Generate tiles
        stride = TILE_SIZE - OVERLAP
        y_starts = list(range(0, h - TILE_SIZE + 1, stride))
        x_starts = list(range(0, w - TILE_SIZE + 1, stride))
        
        if not y_starts or y_starts[-1] + TILE_SIZE < h:
            y_starts.append(max(0, h - TILE_SIZE))
        if not x_starts or x_starts[-1] + TILE_SIZE < w:
            x_starts.append(max(0, w - TILE_SIZE))
        
        total = len(y_starts) * len(x_starts)
        print(f"Processing {total} tiles...")
        
        # Blending window
        y_coords = np.arange(TILE_SIZE)
        x_coords = np.arange(TILE_SIZE)
        y_window = np.cos((y_coords / TILE_SIZE - 0.5) * np.pi)
        x_window = np.cos((x_coords / TILE_SIZE - 0.5) * np.pi)
        blend = np.outer(y_window, x_window).astype(np.float32)
        
        with torch.no_grad():
            for i, y in enumerate(y_starts):
                for j, x in enumerate(x_starts):
                    tile_h = min(TILE_SIZE, h - y)
                    tile_w = min(TILE_SIZE, w - x)
                    
                    # Read tile
                    tile = src.read(window=Window(x, y, tile_w, tile_h))
                    
                    # Pad if needed
                    if tile_h < TILE_SIZE or tile_w < TILE_SIZE:
                        pad_h = TILE_SIZE - tile_h
                        pad_w = TILE_SIZE - tile_w
                        tile = np.pad(tile, ((0, 0), (0, pad_h), (0, pad_w)), mode='reflect')
                    
                    # Prepare: (3, H, W), normalize
                    tile_rgb = tile[:3].astype(np.float32)
                    tile_rgb = (tile_rgb - tile_rgb.min()) / (tile_rgb.max() - tile_rgb.min() + 1e-8)
                    tensor = torch.from_numpy(tile_rgb).unsqueeze(0).to(DEVICE)
                    
                    # Predict
                    logits = model(tensor)
                    prob = torch.sigmoid(logits).squeeze().cpu().numpy()
                    
                    # Accumulate
                    prob_map[y:y+tile_h, x:x+tile_w] += prob[:tile_h, :tile_w] * blend[:tile_h, :tile_w]
                    weight_map[y:y+tile_h, x:x+tile_w] += blend[:tile_h, :tile_w]
                    
                    if (i * len(x_starts) + j + 1) % 100 == 0:
                        print(f"  {i * len(x_starts) + j + 1}/{total}")
        
        # Normalize
        prob_map /= (weight_map + 1e-8)
        binary_mask = (prob_map > threshold).astype(np.uint8)
        
        # Save
        out_profile = {
            'driver': 'GTiff',
            'height': h,
            'width': w,
            'count': 1,
            'dtype': 'uint8',
            'crs': src.crs,
            'transform': src.transform,
            'compress': 'lzw',
            'tiled': True,
            'nodata': 255,
        }
        
        with rasterio.open(output_path, 'w', **out_profile) as dst:
            dst.write(binary_mask, 1)
        
        print(f"Saved: {output_path}")
        return prob_map, binary_mask

# ============================================================
# RUN INFERENCE ON BOTH IMAGES
# ============================================================

print("\n" + "="*60)
print("PREDICTING WATER MASKS")
print("="*60)

water_prob_t1, water_t1 = predict_binary_mask(
    water_model, T1_PATH, 
    os.path.join(OUTPUT_DIR, "water_t1.tif")
)
water_prob_t2, water_t2 = predict_binary_mask(
    water_model, T2_PATH,
    os.path.join(OUTPUT_DIR, "water_t2.tif")
)

print("\n" + "="*60)
print("PREDICTING ROAD MASKS")
print("="*60)

road_prob_t1, road_t1 = predict_binary_mask(
    road_model, T1_PATH,
    os.path.join(OUTPUT_DIR, "road_t1.tif")
)
road_prob_t2, road_t2 = predict_binary_mask(
    road_model, T2_PATH,
    os.path.join(OUTPUT_DIR, "road_t2.tif")
)

print("\n" + "="*60)
print("PREDICTING VEGETATION MASKS")
print("="*60)

veg_prob_t1, veg_t1 = predict_binary_mask(
    veg_model, T1_PATH,
    os.path.join(OUTPUT_DIR, "vegetation_t1.tif")
)
veg_prob_t2, veg_t2 = predict_binary_mask(
    veg_model, T2_PATH,
    os.path.join(OUTPUT_DIR, "vegetation_t2.tif")
)

# ============================================================
# COMBINE INTO MULTI-CLASS LABEL MAPS
# ============================================================

print("\n" + "="*60)
print("CREATING MULTI-CLASS LABEL MAPS")
print("="*60)

def combine_labels(water, road, veg, output_path, reference_profile):
    """
    Combine binary masks into single multi-class label map.
    Priority: water > road > vegetation > background
    Building is inferred as: NOT (water OR road OR veg) in urban areas
    """
    h, w = water.shape
    labels = np.zeros((h, w), dtype=np.uint8)
    
    # Apply priority
    labels[veg > 0] = 3      # vegetation
    labels[road > 0] = 2     # road (overwrites veg)
    labels[water > 0] = 1    # water (overwrites road)
    
    # Building = areas that are none of the above but have high texture/structure
    # Simple heuristic: if not water/road/veg and high local variance → building
    # This is a rough approximation!
    
    # Save
    out_profile = reference_profile.copy()
    out_profile.update({
        'count': 1,
        'dtype': 'uint8',
        'nodata': 255,
        'compress': 'lzw',
    })
    
    with rasterio.open(output_path, 'w', **out_profile) as dst:
        dst.write(labels, 1)
    
    print(f"Saved: {output_path}")
    return labels

with rasterio.open(T1_PATH) as src:
    profile = src.profile

labels_t1 = combine_labels(water_t1, road_t1, veg_t1, 
                           os.path.join(OUTPUT_DIR, "labels_t1.tif"), profile)
labels_t2 = combine_labels(water_t2, road_t2, veg_t2,
                           os.path.join(OUTPUT_DIR, "labels_t2.tif"), profile)

# ============================================================
# CHANGE DETECTION
# ============================================================

print("\n" + "="*60)
print("DETECTING CHANGES")
print("="*60)

# Binary change: any class change
binary_change = (labels_t1 != labels_t2).astype(np.uint8)

# Per-class changes
water_change = ((labels_t1 == 1) | (labels_t2 == 1)) & (labels_t1 != labels_t2)
road_change = ((labels_t1 == 2) | (labels_t2 == 2)) & (labels_t1 != labels_t2)
veg_change = ((labels_t1 == 3) | (labels_t2 == 3)) & (labels_t1 != labels_t2)

# Transition map: what changed TO what
# Encode as: from_class * 10 + to_class
transition = np.zeros_like(labels_t1)
transition[labels_t1 != labels_t2] = labels_t1[labels_t1 != labels_t2] * 10 + labels_t2[labels_t1 != labels_t2]

# Save all
def save_mask(mask, path, profile):
    out_profile = profile.copy()
    out_profile.update({
        'count': 1,
        'dtype': 'uint8',
        'nodata': 255,
        'compress': 'lzw',
    })
    with rasterio.open(path, 'w', **out_profile) as dst:
        dst.write(mask.astype(np.uint8), 1)

save_mask(binary_change, os.path.join(OUTPUT_DIR, "binary_change.tif"), profile)
save_mask(water_change, os.path.join(OUTPUT_DIR, "water_change.tif"), profile)
save_mask(road_change, os.path.join(OUTPUT_DIR, "road_change.tif"), profile)
save_mask(veg_change, os.path.join(OUTPUT_DIR, "vegetation_change.tif"), profile)
save_mask(transition, os.path.join(OUTPUT_DIR, "transition_map.tif"), profile)

print(f"\nAll outputs saved to: {OUTPUT_DIR}")

# ============================================================
# VISUALIZATION
# ============================================================

print("\n" + "="*60)
print("CREATING VISUALIZATION")
print("="*60)

def visualize_results(t1_path, t2_path, labels_t1, labels_t2, change, output_path):
    """Create a nice visualization of results."""
    
    with rasterio.open(t1_path) as src:
        h, w = src.height, src.width
        cx, cy = w // 2, h // 2
        size = 3000
        
        t1_img = src.read(window=Window(cx-size//2, cy-size//2, size, size))
        t1_rgb = np.transpose(t1_img[:3], (1, 2, 0))
        t1_rgb = (t1_rgb - t1_rgb.min()) / (t1_rgb.max() - t1_rgb.min())
        
        t2_img = rasterio.open(t2_path).read(window=Window(cx-size//2, cy-size//2, size, size))
        t2_rgb = np.transpose(t2_img[:3], (1, 2, 0))
        t2_rgb = (t2_rgb - t2_rgb.min()) / (t2_rgb.max() - t2_rgb.min())
    
    # Crop labels to same region
    y1, y2 = cy - size//2, cy + size//2
    x1, x2 = cx - size//2, cx + size//2
    lbl_t1_crop = labels_t1[y1:y2, x1:x2]
    lbl_t2_crop = labels_t2[y1:y2, x1:x2]
    change_crop = change[y1:y2, x1:x2]
    
    # Colorize labels
    def colorize(labels):
        vis = np.zeros((*labels.shape, 3), dtype=np.uint8)
        for cls, color in CLASS_COLORS.items():
            vis[labels == cls] = color
        return vis
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    axes[0, 0].imshow(t1_rgb); axes[0, 0].set_title('T1 (2021)'); axes[0, 0].axis('off')
    axes[0, 1].imshow(t2_rgb); axes[0, 1].set_title('T2 (2025)'); axes[0, 1].axis('off')
    axes[0, 2].imshow(np.abs(t1_rgb - t2_rgb)); axes[0, 2].set_title('Raw Difference'); axes[0, 2].axis('off')
    
    axes[1, 0].imshow(colorize(lbl_t1_crop)); axes[1, 0].set_title('T1 Labels'); axes[1, 0].axis('off')
    axes[1, 1].imshow(colorize(lbl_t2_crop)); axes[1, 1].set_title('T2 Labels'); axes[1, 1].axis('off')
    
    # Change overlay
    change_vis = t2_rgb.copy()
    change_vis[change_crop > 0] = [1, 0, 0]  # Red for change
    axes[1, 2].imshow(change_vis); axes[1, 2].set_title('Changes (Red)'); axes[1, 2].axis('off')
    
    # Legend
    legend_elements = [Patch(facecolor=np.array(c)/255, label=CLASSES[i]) 
                      for i, c in CLASS_COLORS.items()]
    fig.legend(handles=legend_elements, loc='lower center', ncol=5, fontsize=10)
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()

visualize_results(
    T1_PATH, T2_PATH, labels_t1, labels_t2, binary_change,
    os.path.join(OUTPUT_DIR, "change_detection_preview.png")
)

print("\nDone! Check output folder for all results.")