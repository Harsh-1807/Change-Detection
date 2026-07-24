import rasterio
import numpy as np
import matplotlib.pyplot as plt

T1 = r"C:\Users\13519\Desktop\Airoli_2021_modified.tif"
T2 = r"C:\Users\13519\Desktop\Airoli_2025_resampled.tif"

with rasterio.open(T1) as t1, rasterio.open(T2) as t2:
    # Read a crop with clear linear features (roads)
    size = 3000
    cx, cy = t1.width // 2, t1.height // 2
    
    t1_crop = t1.read(window=rasterio.windows.Window(cx-size//2, cy-size//2, size, size))
    t2_crop = t2.read(window=rasterio.windows.Window(cx-size//2, cy-size//2, size, size))
    
    # Convert to RGB
    def to_rgb(arr):
        rgb = np.transpose(arr[:3], (1, 2, 0))
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())
        return rgb
    
    t1_rgb = to_rgb(t1_crop)
    t2_rgb = to_rgb(t2_crop)
    
    # BLEND: 50% T1 + 50% T2
    blended = 0.5 * t1_rgb + 0.5 * t2_rgb
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(t1_rgb); axes[0].set_title('T1 (2021)'); axes[0].axis('off')
    axes[1].imshow(t2_rgb); axes[1].set_title('T2 (2025)'); axes[1].axis('off')
    axes[2].imshow(blended); axes[2].set_title('BLENDED: Check for double edges!'); axes[2].axis('off')
    
    plt.tight_layout(); plt.show()
    
    print("If roads/buildings look SINGLE in blended image → alignment is PERFECT")
    print("If you see GHOSTING/DOUBLE edges → need SIFT/LoFTR")