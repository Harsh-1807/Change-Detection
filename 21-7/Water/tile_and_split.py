"""
Tile large raster/vector pairs into smaller chips for training.
Creates train/val split from a single data folder.
"""

import os
import shutil
import random
from pathlib import Path

import rasterio
import numpy as np
import geopandas as gpd
from rasterio import features
from shapely.geometry import box, Polygon, MultiPolygon
from shapely.affinity import translate, scale
from shapely import make_valid


# ── CONFIG ──────────────────────────────────────────────────────────
RASTER_DIR = Path(r"C:\Users\13519\data\train\water_bodies\rasters")
VECTOR_DIR = Path(r"C:\Users\13519\data\train\water_bodies\vectors")

OUT_ROOT = Path(r"C:\Users\13519\data\train\water_bodies")

TILE_SIZE = 512
STRIDE = 512
VAL_RATIO = 0.15
RANDOM_SEED = 42
# ────────────────────────────────────────────────────────────────────


def world_to_pixel(geom, left, top, pixel_size_x, pixel_size_y):
    """
    Translate a geometry from world coords to pixel coords within a tile.
    """
    geom_translated = translate(geom, xoff=-left, yoff=-top)
    geom_scaled = scale(
        geom_translated,
        xfact=1.0 / pixel_size_x,
        yfact=1.0 / pixel_size_y,
        origin=(0, 0)
    )
    return geom_scaled


def get_valid_geometries(gdf):
    """
    Extract valid Polygon/MultiPolygon geometries from a GeoDataFrame.
    Fixes invalid geometries, drops non-polygon types and empties.
    """
    valid_geoms = []
    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        
        # Fix invalid geometries
        if not geom.is_valid:
            geom = make_valid(geom)
            if geom is None or geom.is_empty:
                continue
        
        # Handle GeometryCollection — extract polygons
        if geom.geom_type == 'GeometryCollection':
            for part in geom.geoms:
                if part.geom_type in ('Polygon', 'MultiPolygon') and not part.is_empty:
                    valid_geoms.append(part)
        elif geom.geom_type in ('Polygon', 'MultiPolygon'):
            valid_geoms.append(geom)
        # Skip Point, LineString, etc.
    
    return valid_geoms


def tile_raster_vector_pair(raster_path: Path, vector_path: Path, out_root: Path, tile_size: int, stride: int):
    with rasterio.open(raster_path) as src:
        img = src.read([1, 2, 3]).astype(np.uint8)
        transform = src.transform
        crs = src.crs
        height, width = src.height, src.width

    # Read & reproject vector if needed
    all_geoms = []
    if vector_path.exists():
        try:
            gdf = gpd.read_file(vector_path)
            if gdf.crs is not None and gdf.crs != crs:
                gdf = gdf.to_crs(crs)
            
            all_geoms = get_valid_geometries(gdf)
            print(f"  Vector has {len(gdf)} raw geometries -> {len(all_geoms)} valid polygons")
        except Exception as e:
            print(f"  ERROR reading vector {vector_path.name}: {e}")
    else:
        print(f"  WARNING: No vector file found at {vector_path}")

    tile_count = 0

    for y in range(0, height - tile_size + 1, stride):
        for x in range(0, width - tile_size + 1, stride):
            chip = img[:, y:y+tile_size, x:x+tile_size]

            window_transform = rasterio.windows.transform(
                rasterio.windows.Window(x, y, tile_size, tile_size),
                transform
            )

            left, top = window_transform * (0, 0)
            right, bottom = window_transform * (tile_size, tile_size)
            
            pixel_size_x = (right - left) / tile_size
            pixel_size_y = (bottom - top) / tile_size

            tile_box = box(left, bottom, right, top)

            # Clip geometries to this tile
            clipped_geoms = []
            for geom in all_geoms:
                try:
                    inter = geom.intersection(tile_box)
                    if inter.is_empty:
                        continue
                    
                    local_geom = world_to_pixel(inter, left, top, pixel_size_x, pixel_size_y)
                    
                    minx, miny, maxx, maxy = local_geom.bounds
                    if maxx < 0 or maxy < 0 or minx > tile_size or miny > tile_size:
                        continue
                    
                    # After scaling, geometry might become invalid — fix it
                    if not local_geom.is_valid:
                        local_geom = make_valid(local_geom)
                        if local_geom is None or local_geom.is_empty:
                            continue
                    
                    clipped_geoms.append(local_geom)
                except Exception as e:
                    # Skip problematic geometries silently
                    continue

            # Rasterize
            mask = np.zeros((tile_size, tile_size), dtype=np.uint8)
            if clipped_geoms:
                shapes = ((g, 1) for g in clipped_geoms)
                try:
                    burned = features.rasterize(
                        shapes=shapes,
                        out_shape=(tile_size, tile_size),
                        transform=rasterio.Affine.identity(),
                        fill=0,
                        default_value=1,
                        all_touched=True,
                    )
                    mask = burned
                except Exception as e:
                    print(f"    Rasterize error at tile ({x},{y}): {e}")
                    continue

            if mask.sum() == 0:
                continue

            # Save chip
            chip_name = f"{raster_path.stem}_{x}_{y}.tif"
            chip_raster_path = out_root / "images" / chip_name
            chip_mask_path = out_root / "masks" / chip_name

            chip_raster_path.parent.mkdir(parents=True, exist_ok=True)
            chip_mask_path.parent.mkdir(parents=True, exist_ok=True)

            with rasterio.open(
                chip_raster_path,
                'w',
                driver='GTiff',
                height=tile_size,
                width=tile_size,
                count=3,
                dtype=img.dtype,
                crs=crs,
                transform=window_transform,
            ) as dst:
                dst.write(chip)

            with rasterio.open(
                chip_mask_path,
                'w',
                driver='GTiff',
                height=tile_size,
                width=tile_size,
                count=1,
                dtype=mask.dtype,
                crs=crs,
                transform=window_transform,
            ) as dst:
                dst.write(mask, 1)

            tile_count += 1

    return tile_count


def split_train_val(all_tiles, val_ratio, seed):
    random.seed(seed)
    random.shuffle(all_tiles)
    n_val = max(1, int(len(all_tiles) * val_ratio))
    return all_tiles[n_val:], all_tiles[:n_val]


def main():
    for split in ["train", "val"]:
        for sub in ["images", "masks"]:
            p = OUT_ROOT / split / sub
            if p.exists():
                shutil.rmtree(p)
            p.mkdir(parents=True, exist_ok=True)

    raster_files = sorted(RASTER_DIR.glob("*.tif"))
    print(f"Found {len(raster_files)} raster files to tile")

    all_train_tiles = []
    all_val_tiles = []

    for raster_path in raster_files:
        vector_path = VECTOR_DIR / raster_path.with_suffix(".geojson").name
        print(f"\nTiling: {raster_path.name}")

        scratch = OUT_ROOT / "_scratch"
        scratch.mkdir(parents=True, exist_ok=True)

        n_tiles = tile_raster_vector_pair(raster_path, vector_path, scratch, TILE_SIZE, STRIDE)
        print(f"  -> {n_tiles} non-empty tiles")

        if n_tiles == 0:
            shutil.rmtree(scratch)
            continue

        scratch_images = sorted((scratch / "images").glob("*.tif"))
        train_tiles, val_tiles = split_train_val(scratch_images, VAL_RATIO, RANDOM_SEED)

        for img_path in train_tiles:
            mask_path = scratch / "masks" / img_path.name
            shutil.move(str(img_path), str(OUT_ROOT / "train" / "images" / img_path.name))
            shutil.move(str(mask_path), str(OUT_ROOT / "train" / "masks" / img_path.name))
            all_train_tiles.append(img_path.name)

        for img_path in val_tiles:
            mask_path = scratch / "masks" / img_path.name
            shutil.move(str(img_path), str(OUT_ROOT / "val" / "images" / img_path.name))
            shutil.move(str(mask_path), str(OUT_ROOT / "val" / "masks" / img_path.name))
            all_val_tiles.append(img_path.name)

        shutil.rmtree(scratch)

    print(f"\n{'='*50}")
    print(f"Train tiles: {len(all_train_tiles)}")
    print(f"Val tiles:   {len(all_val_tiles)}")
    print(f"Output:      {OUT_ROOT}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()