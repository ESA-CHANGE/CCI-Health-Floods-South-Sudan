# -*- coding: utf-8 -*-

__author__ = "Macarena Merida Floriano, mmef"
__maintainer__ = "Macarena Merida Floriano, mmef"
__email__ = "mmef@gmv.com"


"""
CHANGE: data preprocessing script
========================================

Script to binarize flood maps for South Sudan.

"""

import numpy as np
import rasterio as rs
from tqdm import tqdm
import os
from pathlib import Path



maps_path = "/mnt/staas/CLICHE/00_DATA/daily_new/"
save_path = "/mnt/staas/CLICHE/00_DATA/bin_maps/daily_new/"
os.makedirs(save_path, exist_ok=True)
maps_files = sorted([m for m in os.listdir(maps_path) if m.endswith('.tif')])

# Code values
no_data_out_aoi = 1
no_data_real1 = 30
no_data_real2 = 50
flood = 200
no_data_out = -9999

# =========================
# AOI FROM FIRST MAP
# =========================
first_map = maps_files[0]

with rs.open(os.path.join(maps_path, first_map)) as src:
    img0 = src.read(1)
    profile0 = src.profile.copy()

# --- compute AOI  ---
mask_valid = img0 != no_data_out_aoi
if not np.any(mask_valid):
    raise ValueError("First map has no valid AOI")

rows, cols = np.where(mask_valid)
row_min, row_max = rows.min(), rows.max()
col_min, col_max = cols.min(), cols.max()

# --- crop first image ---
img0_crop = img0[row_min:row_max + 1, col_min:col_max + 1]

# --- update transform ---
transform = profile0["transform"]
new_transform = rs.Affine(
    transform.a,
    transform.b,
    transform.c + col_min * transform.a,
    transform.d,
    transform.e,
    transform.f + row_min * transform.e
)

# --- fixed output profile ---
out_profile = profile0.copy()
out_profile.update(
    height=img0_crop.shape[0],
    width=img0_crop.shape[1],
    transform=new_transform,
    dtype=rs.int16,
    nodata=no_data_out,
    compress="deflate",
    count=1
)

def crop_aoi(img, profile):
    mask_valida = (img != no_data_out_aoi)
    if not np.any(mask_valida):
        raise ValueError("Image %s have does not have valid data within the AOI" % map)

    # Compute bounding-box
    rows, cols = np.where(mask_valida)

    min_row, max_row = rows.min(), rows.max()
    min_col, max_col = cols.min(), cols.max()

    # Crop raster
    img_crop = img[min_row:max_row+1, min_col:max_col+1]
    
    # Update profile
    transform = profile["transform"]
    new_transform = rs.Affine(
        transform.a,
        transform.b,
        transform.c + min_col * transform.a,
        transform.d,
        transform.e,
        transform.f + min_row * transform.e
    )
    
    profile_crop = profile.copy()
    profile_crop.update({
        "height": img_crop.shape[0],
        "width": img_crop.shape[1],
        "transform": new_transform
    })

    ref_shape = (img_crop.shape[0], img_crop.shape[1])
    return img_crop, profile_crop, ref_shape


def binarize(img):
    binary = np.zeros_like(img, dtype=np.int16)
    binary[img == flood] = 1
    binary[img == no_data_real1] = no_data_out
    binary[img == no_data_real2] = no_data_out
    return binary

bad_maps = []
for map_name in tqdm(maps_files, desc="Processing maps"):
    
    if Path(os.path.join(save_path, map_name)).is_file():
        print("File %s already exists" % map_name)
        continue 
    
    # -------- Load map --------
    with rs.open(os.path.join(maps_path, map_name)) as src:
        img = src.read(1)              # leer banda 1
        # --- apply FIXED AOI ---
        img_crop = img[row_min:row_max + 1, col_min:col_max + 1]
        #profile = src.profile 
        # --- sanity check ---
        if img_crop.shape != img0_crop.shape:
            raise ValueError(f"Shape mismatch in {map_name}")
    
    # -------- Check valid code values --------
    valid_values = {1, 16, 17, 20, 30, 50, 99, 200}
    #unique_values = np.unique(img)
    #if not set(list(unique_values)).issubset(valid_values):
    #    print(f"Unexpected values in {map}: {unique_values}")
    #    bad_maps.append(map)
    unique_values = set(np.unique(img_crop))
    if not unique_values.issubset(valid_values):
        bad_maps.append(map_name)
    
    # -------- Auto-crop --------
    # This created a dynamic AOI depending on valid data within the master AOI
    #img_crop, profile_crop = crop_aoi(img, profile)
    
    # -------- Binarize map --------
    print("Staring binarization...")
    binary = binarize(img_crop)
    
    # -------- Save binary maps --------
    #profile_crop.update(dtype=rs.int16, nodata=-9999)

    with rs.open(os.path.join(save_path, map_name), 'w', **out_profile) as dst_r:
        dst_r.write(binary, 1)
    
    print(f"✔ Binarized map saved: {os.path.join(save_path, map_name)}")

# Save to-review maps list
if  bad_maps:
    with open(save_path + "maps2review.txt", 'w') as text_file:
        text_file.write("\n".join(bad_maps))

print("✔ Binarization finished")
print(f"✔ Output directory: {save_path}")
print(f"⚠ Maps to review: {len(bad_maps)}")