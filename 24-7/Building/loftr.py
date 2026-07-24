"""
LoFTR Alignment Verification
Double-check your georeferenced pair before SemDINO
"""

import torch
import numpy as np
import cv2
import rasterio
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURATION
# ============================================================
T1 = r"C:\Users\13519\Desktop\Airoli_2021_modified.tif"
T2 = r"C:\Users\13519\Desktop\Airoli_2025_resampled.tif"

# LoFTR needs GPU — CPU will be very slow
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

if device.type == 'cpu':
    print("WARNING: LoFTR on CPU will take 10-30 minutes!")
    print("Consider using a smaller max_dim or skipping LoFTR")

# ============================================================
# INSTALL KORNIA IF NEEDED
# ============================================================
try:
    import kornia.feature as KF
except ImportError:
    print("Installing kornia...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'kornia'])
    import kornia.feature as KF

# ============================================================
# LOAD IMAGES (resize for LoFTR memory)
# ============================================================
def load_for_loftr(path, max_dim=2048):
    """Load and resize image for LoFTR."""
    with rasterio.open(path) as src:
        img = src.read()
        img = np.transpose(img[:3], (1, 2, 0))
        img = (img - img.min()) / (img.max() - img.min())
        img = (img * 255).astype(np.uint8)
        
        # Resize if too large
        h, w = img.shape[:2]
        scale = max_dim / max(h, w)
        if scale < 1:
            new_w, new_h = int(w * scale), int(h * scale)
            img = cv2.resize(img, (new_w, new_h))
            print(f"  Resized to {new_h}x{new_w}")
        else:
            scale = 1.0
        
        return img, scale

print("\nLoading T1 (2021)...")
t1_img, s1 = load_for_loftr(T1, max_dim=256)
print(f"  Shape: {t1_img.shape}")

print("\nLoading T2 (2025)...")
t2_img, s2 = load_for_loftr(T2, max_dim=256)
print(f"  Shape: {t2_img.shape}")

# ============================================================
# PREPARE TENSORS
# ============================================================
def prepare(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    tensor = torch.from_numpy(gray).float()[None, None] / 255.0
    return tensor.to(device)

t1_tensor = prepare(t1_img)
t2_tensor = prepare(t2_img)

print(f"\nTensor shapes: {t1_tensor.shape}, {t2_tensor.shape}")

# ============================================================
# LOAD LoFTR
# ============================================================
print("\nLoading LoFTR (downloads ~100MB weights on first run)...")
loftr = KF.LoFTR(pretrained='outdoor').eval().to(device)

# ============================================================
# RUN MATCHING
# ============================================================
print("\nRunning LoFTR matching...")
print("  This takes 1-5 minutes on GPU, 10-30 minutes on CPU...")

with torch.no_grad():
    input_dict = {"image0": t1_tensor, "image1": t2_tensor}
    correspondences = loftr(input_dict)

mkpts0 = correspondences['keypoints0'].cpu().numpy()
mkpts1 = correspondences['keypoints1'].cpu().numpy()
conf = correspondences['confidence'].cpu().numpy()

print(f"\n{'='*60}")
print("RESULTS")
print(f"{'='*60}")
print(f"Total matches found: {len(mkpts0)}")
print(f"Confidence range: {conf.min():.3f} - {conf.max():.3f}")

# ============================================================
# FILTER BY CONFIDENCE
# ============================================================
threshold = 0.3
mask = conf > threshold
mkpts0_f = mkpts0[mask]
mkpts1_f = mkpts1[mask]
conf_f = conf[mask]

print(f"\nAfter filtering (conf > {threshold}): {len(mkpts0_f)} matches")

# ============================================================
# ESTIMATE HOMOGRAPHY
# ============================================================
if len(mkpts0_f) >= 10:
    print("\nEstimating homography with RANSAC...")
    H, mask_h = cv2.findHomography(
        mkpts1_f, mkpts0_f,
        cv2.RANSAC, 5.0
    )
    
    if H is not None:
        inliers = np.sum(mask_h)
        print(f"  RANSAC inliers: {inliers} / {len(mkpts0_f)}")
        print(f"  Inlier ratio: {inliers/len(mkpts0_f)*100:.1f}%")
        
        print(f"\nHomography matrix H:")
        print(f"  [{H[0,0]:10.4f}  {H[0,1]:10.4f}  {H[0,2]:12.4f}]")
        print(f"  [{H[1,0]:10.4f}  {H[1,1]:10.4f}  {H[1,2]:12.4f}]")
        print(f"  [{H[2,0]:10.4f}  {H[2,1]:10.4f}  {H[2,2]:12.4f}]")
        
        # Check if H is close to identity
        I = np.eye(3)
        deviation = np.mean(np.abs(H - I))
        print(f"\nDeviation from identity: {deviation:.6f}")
        
        if deviation < 0.01:
            print(f"\n{'='*60}")
            print("CONCLUSION")
            print(f"{'='*60}")
            print("LoFTR confirms: Images are ALREADY ALIGNED!")
            print("H is essentially identity.")
            print("Your georeferencing worked perfectly.")
            print("\n→ Skip LoFTR, proceed to SemDINO!")
        else:
            print(f"\n{'='*60}")
            print("CONCLUSION")
            print(f"{'='*60}")
            print("LoFTR found significant transform.")
            print(f"Consider applying this H to T2.")
    else:
        print("  ERROR: Could not compute homography")
else:
    print(f"\nNot enough matches ({len(mkpts0_f)}) for homography")

# ============================================================
# VISUALIZE MATCHES
# ============================================================
print("\nCreating visualization...")

fig, axes = plt.subplots(1, 2, figsize=(14, 7))

axes[0].imshow(t1_img)
axes[0].set_title(f'T1 (2021) - {t1_img.shape[1]}x{t1_img.shape[0]}')
axes[0].axis('off')

axes[1].imshow(t2_img)
axes[1].set_title(f'T2 (2025) - {t2_img.shape[1]}x{t2_img.shape[0]}')
axes[1].axis('off')

# Draw top 200 matches
n_show = min(200, len(mkpts0_f))
colors = plt.cm.viridis(conf_f[:n_show] / conf_f[:n_show].max())

for i in range(n_show):
    pt1 = mkpts0_f[i]
    pt2 = mkpts1_f[i]
    axes[0].plot(pt1[0], pt1[1], 'o', color=colors[i], markersize=3, alpha=0.6)
    axes[1].plot(pt2[0], pt2[1], 'o', color=colors[i], markersize=3, alpha=0.6)

plt.suptitle(f'LoFTR Matches (Top {n_show} of {len(mkpts0_f)})', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('loftr_matches.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nVisualization saved: loftr_matches.png")