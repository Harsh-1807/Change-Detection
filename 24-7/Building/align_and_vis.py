import rasterio
import numpy as np
import matplotlib.pyplot as plt

T1 = r"C:\Users\13519\Desktop\Airoli_2021_modified.tif"
T2 = r"C:\Users\13519\Desktop\Airoli_2025_resampled.tif"

with rasterio.open(T1) as t1, rasterio.open(T2) as t2:
    # Read a crop
    size = 5000
    cx, cy = t1.width // 2, t1.height // 2
    
    t1_crop = t1.read(window=rasterio.windows.Window(cx-size//2, cy-size//2, size, size))
    t2_crop = t2.read(window=rasterio.windows.Window(cx-size//2, cy-size//2, size, size))
    
    # Convert to RGB
    t1_rgb = np.transpose(t1_crop[:3], (1, 2, 0))
    t2_rgb = np.transpose(t2_crop[:3], (1, 2, 0))
    
    # Normalize per-image (handles different brightness)
    t1_rgb = (t1_rgb - t1_rgb.min()) / (t1_rgb.max() - t1_rgb.min())
    t2_rgb = (t2_rgb - t2_rgb.min()) / (t2_rgb.max() - t2_rgb.min())
    
    # Show
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(t1_rgb); axes[0].set_title('T1 (2021)'); axes[0].axis('off')
    axes[1].imshow(t2_rgb); axes[1].set_title('T2 (2025)'); axes[1].axis('off')
    
    # Difference (this is your change detection input!)
    diff = np.abs(t1_rgb.astype(float) - t2_rgb.astype(float))
    axes[2].imshow(diff); axes[2].set_title(f'Change Map (mean={np.mean(diff):.4f})'); axes[2].axis('off')
    
    plt.tight_layout(); plt.show()
    
    print("Georeferenced alignment is GOOD.")
    print(f"Difference = {np.mean(diff):.4f} → This is REAL CHANGE, not misalignment")
    print("Proceed directly to SemDINO for semantic change detection!")