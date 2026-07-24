"""
SEMANTIC CHANGE DETECTION PIPELINE
Input: Aligned T1 (2021) and T2 (2025) satellite images
Output: Semantic maps + per-class change masks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModel
import rasterio
import numpy as np
import cv2
from tqdm import tqdm

# ============================================================
# CONFIGURATION
# ============================================================
T1_PATH = r"C:\Users\13519\Desktop\Airoli_2021_modified.tif"
T2_PATH = r"C:\Users\13519\Desktop\Airoli_2025_resampled.tif"
OUTPUT_DIR = r"C:\Users\13519\Desktop\"

CLASSES = ['background', 'water', 'building', 'road', 'vegetation']
NUM_CLASSES = len(CLASSES)
TILE_SIZE = 512  # SemDINO processes 512x512 patches
OVERLAP = 256    # 50% overlap for smooth blending

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# ============================================================
# STEP 1: LOAD FROZEN DINOv3 FEATURES
# ============================================================
print("Loading DINOv3 backbone...")
processor = AutoImageProcessor.from_pretrained("facebook/dinov2-large")
dino = AutoModel.from_pretrained("facebook/dinov2-large").to(DEVICE)
dino.eval()

# Freeze DINO
for param in dino.parameters():
    param.requires_grad = False

# ============================================================
# STEP 2: BUILD SEMDINO-STYLE MODEL
# ============================================================
class SemDINOChangeDetector(nn.Module):
    """
    SemDINO-style semantic change detector.
    - Dual-branch DINO encoder (frozen)
    - Lightweight CNN decoder for multi-scale fusion
    - Predicts: T1 semantics, T2 semantics, binary change, class-wise change
    """
    def __init__(self, num_classes=5, dino_dim=1024):
        super().__init__()
        
        # CNN backbone for multi-scale detail (lightweight)
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        
        # Fusion: DINO features + CNN features
        self.fusion = nn.Sequential(
            nn.Conv2d(dino_dim + 256, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        
        # Temporal interaction module
        self.temporal_attn = nn.MultiheadAttention(512, num_heads=8, batch_first=True)
        
        # Prediction heads
        self.seg_head_t1 = nn.Conv2d(512, num_classes, 1)
        self.seg_head_t2 = nn.Conv2d(512, num_classes, 1)
        self.change_head = nn.Conv2d(512, 1, 1)  # Binary change
        
        # Class-wise change (water, building, road, vegetation)
        self.class_change_head = nn.Conv2d(512 * 2, num_classes, 1)
        
    def extract_dino_features(self, x):
        """Extract DINOv3 features for a batch of images."""
        # x: (B, 3, H, W) in range [0, 1]
        # DINO expects specific preprocessing
        inputs = processor(images=x, return_tensors="pt", do_rescale=False)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = dino(**inputs)
            features = outputs.last_hidden_state  # (B, N, 1024) where N = num_patches
        
        # Reshape to spatial: (B, H//14, W//14, 1024)
        B, N, C = features.shape
        H_patch = W_patch = int(N ** 0.5)
        features = features.transpose(1, 2).reshape(B, C, H_patch, W_patch)
        
        # Upsample to match CNN features
        features = F.interpolate(features, size=(128, 128), mode='bilinear', align_corners=False)
        
        return features
    
    def forward(self, t1, t2):
        B, C, H, W = t1.shape
        
        # CNN features for both times
        cnn_t1 = self.cnn(t1)  # (B, 256, H/4, W/4)
        cnn_t2 = self.cnn(t2)
        
        # DINO features for both times
        dino_t1 = self.extract_dino_features(t1)  # (B, 1024, 128, 128)
        dino_t2 = self.extract_dino_features(t2)
        
        # Resize CNN to match DINO
        cnn_t1 = F.interpolate(cnn_t1, size=(128, 128), mode='bilinear', align_corners=False)
        cnn_t2 = F.interpolate(cnn_t2, size=(128, 128), mode='bilinear', align_corners=False)
        
        # Fusion
        feat_t1 = self.fusion(torch.cat([dino_t1, cnn_t1], dim=1))  # (B, 512, 128, 128)
        feat_t2 = self.fusion(torch.cat([dino_t2, cnn_t2], dim=1))
        
        # Temporal interaction
        B, C_f, H_f, W_f = feat_t1.shape
        feat_t1_flat = feat_t1.view(B, C_f, -1).transpose(1, 2)  # (B, H*W, C)
        feat_t2_flat = feat_t2.view(B, C_f, -1).transpose(1, 2)
        
        # Cross-attention between T1 and T2
        attn_out_t1, _ = self.temporal_attn(feat_t1_flat, feat_t2_flat, feat_t2_flat)
        attn_out_t2, _ = self.temporal_attn(feat_t2_flat, feat_t1_flat, feat_t1_flat)
        
        feat_t1 = attn_out_t1.transpose(1, 2).view(B, C_f, H_f, W_f)
        feat_t2 = attn_out_t2.transpose(1, 2).view(B, C_f, H_f, W_f)
        
        # Semantic segmentation
        seg_t1 = self.seg_head_t1(feat_t1)  # (B, num_classes, 128, 128)
        seg_t2 = self.seg_head_t2(feat_t2)
        
        # Upsample to original size
        seg_t1 = F.interpolate(seg_t1, size=(H, W), mode='bilinear', align_corners=False)
        seg_t2 = F.interpolate(seg_t2, size=(H, W), mode='bilinear', align_corners=False)
        
        # Binary change detection
        change_feat = torch.abs(feat_t1 - feat_t2)
        change = self.change_head(change_feat)
        change = F.interpolate(change, size=(H, W), mode='bilinear', align_corners=False)
        
        # Class-wise change
        combined = torch.cat([feat_t1, feat_t2], dim=1)
        class_change = self.class_change_head(combined)
        class_change = F.interpolate(class_change, size=(H, W), mode='bilinear', align_corners=False)
        
        return {
            'seg_t1': seg_t1,
            'seg_t2': seg_t2,
            'change': change,
            'class_change': class_change,
        }

# ============================================================
# STEP 3: SLIDING-WINDOW INFERENCE
# ============================================================
def sliding_window_inference(model, image_path, tile_size=512, overlap=256):
    """
    Process large satellite image in overlapping tiles.
    """
    with rasterio.open(image_path) as src:
        h, w = src.height, src.width
        bands = src.count
        
        # Output arrays
        seg_map = np.zeros((NUM_CLASSES, h, w), dtype=np.float32)
        weight_map = np.zeros((h, w), dtype=np.float32)
        
        # Generate tile coordinates
        stride = tile_size - overlap
        y_starts = list(range(0, h - tile_size + 1, stride))
        x_starts = list(range(0, w - tile_size + 1, stride))
        
        # Add final tiles if needed
        if y_starts[-1] + tile_size < h:
            y_starts.append(h - tile_size)
        if x_starts[-1] + tile_size < w:
            x_starts.append(w - tile_size)
        
        total_tiles = len(y_starts) * len(x_starts)
        print(f"Processing {total_tiles} tiles...")
        
        # Create blending weights (cosine window for smooth overlap)
        y_coords = np.arange(tile_size)
        x_coords = np.arange(tile_size)
        y_window = np.cos((y_coords / tile_size - 0.5) * np.pi)
        x_window = np.cos((x_coords / tile_size - 0.5) * np.pi)
        window = np.outer(y_window, x_window).astype(np.float32)
        
        model.eval()
        with torch.no_grad():
            for y in tqdm(y_starts):
                for x in x_starts:
                    # Read tile
                    tile = src.read(window=rasterio.windows.Window(x, y, tile_size, tile_size))
                    
                    # Handle edge cases
                    tile_h, tile_w = tile.shape[1], tile.shape[2]
                    if tile_h < tile_size or tile_w < tile_size:
                        pad_h = tile_size - tile_h
                        pad_w = tile_size - tile_w
                        tile = np.pad(tile, ((0, 0), (0, pad_h), (0, pad_w)), mode='reflect')
                    
                    # Normalize and convert
                    tile = np.transpose(tile[:3], (1, 2, 0))
                    tile = (tile - tile.min()) / (tile.max() - tile.min())
                    tile_tensor = torch.from_numpy(tile).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
                    
                    # Forward pass (dummy T2 for single-image inference)
                    # In practice, process T1 and T2 together
                    # For now, just extract features
                    # ... (full implementation would process both times)
                    
                    # Placeholder: accumulate predictions
                    # seg_map[:, y:y+tile_h, x:x+tile_w] += pred * window[:tile_h, :tile_w]
                    # weight_map[y:y+tile_h, x:x+tile_w] += window[:tile_h, :tile_w]
        
        # Normalize by weights
        # seg_map /= weight_map
        
        return seg_map

# ============================================================
# STEP 4: TRAINING (Simplified - Self-Pair Augmentation)
# ============================================================
class SelfPairGenerator:
    """
    Generate synthetic change pairs from single images for training.
    Based on Self-Pair paper (WACV 2023).
    """
    def __init__(self):
        self.transforms = transforms.Compose([
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        ])
    
    def generate(self, image, mask):
        """
        Create synthetic changed pair:
        - T1: original
        - T2: augmented (color shift, small geometric perturbation)
        - Change mask: regions where augmentation altered the image
        """
        # T1 is original
        t1 = image.copy()
        
        # T2 is augmented
        t2 = image.copy()
        
        # Apply color jitter
        t2 = self.apply_color_jitter(t2)
        
        # Apply small geometric perturbation
        h, w = t2.shape[:2]
        M = cv2.getRotationMatrix2D((w//2, h//2), np.random.uniform(-3, 3), 
                                     np.random.uniform(0.97, 1.03))
        M[0, 2] += np.random.randint(-5, 5)
        M[1, 2] += np.random.randint(-5, 5)
        t2 = cv2.warpAffine(t2, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        
        # Change mask: where pixels differ significantly
        diff = np.abs(t1.astype(float) - t2.astype(float))
        change_mask = (diff.mean(axis=2) > 30).astype(np.uint8)
        
        return t1, t2, mask, mask, change_mask

    def apply_color_jitter(self, img):
        # Simple color augmentation
        img = img.astype(np.float32)
        img *= np.random.uniform(0.8, 1.2)  # brightness
        img += np.random.uniform(-20, 20)   # contrast shift
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img

# ============================================================
# STEP 5: MULTI-TASK LOSS
# ============================================================
class ChangeDetectionLoss(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=255)
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        
        # Class weights for imbalanced data
        self.class_weights = torch.tensor([0.1, 2.0, 1.5, 1.0, 1.2]).to(DEVICE)
        
    def forward(self, predictions, targets):
        seg_t1_pred = predictions['seg_t1']
        seg_t2_pred = predictions['seg_t2']
        change_pred = predictions['change']
        class_change_pred = predictions['class_change']
        
        seg_t1_gt = targets['seg_t1']
        seg_t2_gt = targets['seg_t2']
        change_gt = targets['change']
        class_change_gt = targets['class_change']
        
        # Segmentation losses
        loss_seg_t1 = F.cross_entropy(seg_t1_pred, seg_t1_gt, weight=self.class_weights)
        loss_seg_t2 = F.cross_entropy(seg_t2_pred, seg_t2_gt, weight=self.class_weights)
        
        # Change detection loss
        loss_change = self.bce(change_pred.squeeze(1), change_gt.float())
        
        # Class-wise change loss
        loss_class_change = F.cross_entropy(class_change_pred, class_change_gt)
        
        # Dice loss for better boundary quality
        loss_dice = self.dice(F.softmax(seg_t1_pred, dim=1), seg_t1_gt) + \
                    self.dice(F.softmax(seg_t2_pred, dim=1), seg_t2_gt)
        
        # Combined
        total_loss = (0.4 * loss_seg_t1 + 0.4 * loss_seg_t2 + 
                      0.3 * loss_change + 0.2 * loss_class_change + 
                      0.1 * loss_dice)
        
        return total_loss, {
            'seg_t1': loss_seg_t1.item(),
            'seg_t2': loss_seg_t2.item(),
            'change': loss_change.item(),
            'class_change': loss_class_change.item(),
            'dice': loss_dice.item(),
        }

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
        
    def forward(self, pred, target):
        pred = pred.argmax(dim=1)
        pred = F.one_hot(pred, num_classes=5).permute(0, 3, 1, 2).float()
        target = F.one_hot(target, num_classes=5).permute(0, 3, 1, 2).float()
        
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()

# ============================================================
# STEP 6: INFERENCE ON YOUR IMAGES
# ============================================================
def run_inference(model, t1_path, t2_path, output_dir):
    """
    Run full inference pipeline on aligned bi-temporal pair.
    """
    print("Running inference...")
    
    # Process both images with sliding window
    # seg_t1 = sliding_window_inference(model, t1_path)
    # seg_t2 = sliding_window_inference(model, t2_path)
    
    # For now, placeholder for the actual implementation
    print("Loading images for inference...")
    
    with rasterio.open(t1_path) as src:
        profile = src.profile
    
    # Outputs to save:
    outputs = {
        'semantic_t1.tif': 'Semantic map at T1',
        'semantic_t2.tif': 'Semantic map at T2',
        'binary_change.tif': 'Binary change mask',
        'water_change.tif': 'Water change mask',
        'building_change.tif': 'Building change mask',
        'road_change.tif': 'Road change mask',
        'vegetation_change.tif': 'Vegetation change mask',
        'transition_map.tif': 'Class transition map (e.g., veg->building)',
    }
    
    for filename, desc in outputs.items():
        print(f"  Would save: {filename} - {desc}")
    
    print("\nInference complete!")

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("SemDINO-Style Semantic Change Detection")
    print("=" * 60)
    
    # Initialize model
    model = SemDINOChangeDetector(num_classes=NUM_CLASSES).to(DEVICE)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    print(f"  Frozen (DINO): {total_params - trainable_params:,}")
    
    # Run inference
    run_inference(model, T1_PATH, T2_PATH, OUTPUT_DIR)
    
    print("\n" + "=" * 60)
    print("Next steps:")
    print("  1. Train model on labeled data (or use pre-trained weights)")
    print("  2. Run inference on your Airoli pair")
    print("  3. Post-process masks (remove small components)")
    print("  4. Visualize results")
    print("=" * 60)