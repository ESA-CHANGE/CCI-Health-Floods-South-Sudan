# -*- coding: utf-8 -*-

__author__ = "Macarena Merida Floriano, mmef"
__maintainer__ = "Macarena Merida Floriano, mmef"
__email__ = "mmef@gmv.com"


"""
CHANGE: data preprocessing script (v2.1 - Monte Carlo ready)
===================================================================================

Script to extract flood hazard metrics around functional health facilities (HF)
for Monte Carlo simulation pool generation.

CHANGES from v2.0 (eo_pool_v2.py):
- Removed aditional metrics, only keeping f, d, r, c:
    - f: mean frequency of HF patch. Mean annual proportoion of flooded days in the HF patch.
    - d: mean duration of HF patch. Mean duration of flooded days in the HF.
    - r: maximum consecutive flooded days in the HF patch. We take percentile 75 of the patch.
    - c: temporal coverage: mean valid_days/theoretical_days.

The output dataset respond to a simplified methodology to get preliminary results, based on 
copules, with a descriptive Monte Carlo rather than a process Monte Carlo. 
"""

# -------- IMPORTS --------
import numpy as np
import pandas as pd
import rasterio
import argparse
import logging
from rasterio.windows import Window
from pathlib import Path
from tqdm import tqdm
from scipy import ndimage

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Instantiate argument parser
parser = argparse.ArgumentParser(description='CHANGE - Flood EO timeseries (Monte Carlo Pool)')
parser.add_argument('--map_type', type=str, choices=['annual', 'daily'], default='annual',
                    help='Type of hazard map: annual or daily (default: annual)')
parser.add_argument('--maps_path', type=str, default= "/mnt/staas/CLICHE/00_DATA/hazard_maps/annual_daily_new",
                    help='Path to the directory containing the hazard maps')
parser.add_argument('--output_path', type=str, default="/mnt/staas/CLICHE/00_DATA/facility_eo_pool_v3.csv",
                    help='Path to save the output CSV file with the EO timeseries')

args = parser.parse_args()

# ---------- CONFIGURATION ----------
# Band indices in the multi-band GeoTIFF (adjust if your structure differs)
BAND_CONFIG = {
    'frequency': 1,           # Band 1: flood frequency (probability 0-1 or 0-100)
    'duration': 2,            # Band 2: flood duration (days)
    'valid_observations': 3,  # Band 3: number of valid observation days
    'max_consecutive': 4      # Band 4: maximum consecutive flooded days
}

# Window sizes by facility type (pixels at 90m resolution)
WINDOW_SIZES = {
    'phcc': 40,      # 3600m x 3600m
    'hospital': 40,  # 3600m x 3600m  
    'phcu': 60       # 5400m x 5400m
}

# ---------- Auxiliar functions ----------------------------------------
def get_window_size(facility_name):
    """Determine window size based on facility type.
    
    Parameters
    ----------
    facility_name : str
        Name of the facility (used to determine window size).
    
    Returns
    -------
    size : int
        Window size in pixels (e.g., 40 for PHCC/Hospital, 60 for PHCU).
    
    """
    fac_lower = facility_name.lower()
    for fac_type, size in WINDOW_SIZES.items():
        if fac_type in fac_lower:
            return size
    logger.warning(f"Unknown facility type for '{facility_name}', using default PHCU size")
    return WINDOW_SIZES['phcu']

def extract_patch(lon, lat, facility_name, array, transform, nodata, band_type='generic'):
    """
    Extract patch around facility coordinates.
    
    Parameters
    ----------
    lon, lat: float
        Facility coordinates in the same CRS as the raster.
    facility_name: str
        Name of the facility (used to determine window size).
    array: numpy array
        2D array of the raster band from which to extract the patch.
    transform: affine.Affine
        Affine transform of the raster for coordinate conversion.
    nodata: value
        NoData value to identify invalid pixels.
    band_type: str
        Type of band being extracted (used for specific metrics, e.g., valid_observations).

    Returns
    -------
    patch : numpy array
        numpy array with extracted data
    coverage : float
        ratio of valid pixels to ideal window size
    min_valid : int or float
        minimum valid observations in any pixel of the patch
    """
    
    col, row = ~transform * (lon, lat)
    col, row = int(round(col)), int(round(row))
    
    size = get_window_size(facility_name)
    half = size // 2
    ideal_pixels = size * size
    
    # Desired window bounds
    row_min = row - half
    row_max = row + half
    col_min = col - half
    col_max = col + half
    
    # Clip to array bounds
    row_min_clip = max(row_min, 0)
    col_min_clip = max(col_min, 0)
    row_max_clip = min(row_max, array.shape[0])
    col_max_clip = min(col_max, array.shape[1])
    
    # If no overlap at all
    if row_min_clip >= row_max_clip or col_min_clip >= col_max_clip:
        return None, 0.0, 0
    
    # Extract valid sub-patch
    patch = array[row_min_clip:row_max_clip,
                  col_min_clip:col_max_clip].astype("float32")
    
    # Replace nodata by NaN
    if nodata is not None:
        patch[patch == nodata] = np.nan
    
    # Calculate coverage metrics
    valid_mask = ~np.isnan(patch)
    valid_pixels = np.sum(valid_mask)
    real_pixels = patch.size
    coverage = valid_pixels / real_pixels
    
    # For valid_observations band, track minimum (worst case)
    if band_type == 'valid_observations':
        min_valid = np.nanmin(patch) if valid_pixels > 0 else 0
    else:
        min_valid = -9999  # Placeholder for other bands
    
    if valid_pixels == 0:
        return None, 0.0, 0
    
    return patch, coverage, min_valid

def summarize_frequency_patch(freq_patch, valid_patch, min_valid_threshold=300, min_pct_good=0.5):
    r"""Summarize flood frequency patch (already a probability).
    
    Parameters
    ----------
    freq_patch : numpy array
        2D array of flood frequency probabilities in the patch (NaN for invalid pixels).
    valid_patch : numpy array
        2D array of valid observation counts in the patch (NaN for invalid pixels).
       
    Returns
    -------
    stats : dict
        Dictionary with frequency statistics and quality descriptors.
    """
    
    if freq_patch is None or np.all(np.isnan(freq_patch)):
        return {
            "mean_freq": np.nan}
    
    # Paso 1: Identificar píxeles de calidad (≥ 300 días válidos)
    good_pixel_mask = (valid_patch >= min_valid_threshold) & (~np.isnan(freq_patch))
    n_good_pixels = np.sum(good_pixel_mask)
    total_valid_pixels = np.sum(~np.isnan(valid_patch))
    if total_valid_pixels == 0:
        return {'mean_freq': np.nan}
    
    pct_good_pixels = n_good_pixels / total_valid_pixels
    
    # Paso 2: ¿Tenemos suficientes píxeles buenos?
    if pct_good_pixels < min_pct_good:
        return {'mean_freq': np.nan}
    
    # Paso 3: Filtrar datos
    freq_good = freq_patch[good_pixel_mask]
    
    # Core statistics
    mean_freq = np.nanmean(freq_good)

    return {
        "mean_freq": mean_freq}

def summarize_duration_patch(dur_patch, valid_patch, min_valid_threshold=300, min_pct_good=0.5):
    r"""Summarize flood duration patch with full descriptors.
    
    Parameters
    ----------
    dur_patch : numpy array
        2D array of flood duration values in the patch (NaN for invalid pixels).
    valid_patch : numpy array
        2D array of valid observation counts in the patch (NaN for invalid pixels).
    
    Returns
    -------
    stats : dict
        Dictionary with duration statistics and quality descriptors.
    """
    
    if dur_patch is None or np.all(np.isnan(dur_patch)):
        return {'mean_dur': np.nan}
    
    # Paso 1: Identificar píxeles de calidad (≥ 300 días válidos)
    good_pixel_mask = (valid_patch >= min_valid_threshold) & (~np.isnan(dur_patch))
    n_good_pixels = np.sum(good_pixel_mask)
    total_valid_pixels = np.sum(~np.isnan(valid_patch))
    if total_valid_pixels == 0:
        return {'mean_dur': np.nan}
    
    pct_good_pixels = n_good_pixels / total_valid_pixels
    
    # Paso 2: ¿Tenemos suficientes píxeles buenos?
    if pct_good_pixels < min_pct_good:
        return {'mean_dur': np.nan}
    
    # Paso 3: Filtrar datos
    dur_good = dur_patch[good_pixel_mask]
    
    # Core statistics
    mean_val = np.nanmean(dur_good)

    return {
        "mean_dur": mean_val
    }
    
def summarize_max_consecutive_patch(cons_patch, valid_patch, min_valid_threshold=300, min_pct_good=0.5):
    r"""Summarize maximum consecutive flooded days patch.
    
    Parameters
    ----------
    cons_patch : numpy array
        2D array of maximum consecutive flooded days in the patch (NaN for invalid pixels).
    
    Returns
    -------
    stats : dict
        Dictionary with max consecutive statistics and quality descriptors.
    """
    
    if cons_patch is None or np.all(np.isnan(cons_patch)):
        return {
            "pct75_consecutive": np.nan}
    
    # Paso 1: Identificar píxeles de calidad (≥ 300 días válidos)
    good_pixel_mask = (valid_patch >= min_valid_threshold) & (~np.isnan(cons_patch))
    n_good_pixels = np.sum(good_pixel_mask)
    total_valid_pixels = np.sum(~np.isnan(valid_patch))
    if total_valid_pixels == 0:
        return {'pct75_consecutive': np.nan}
    
    pct_good_pixels = n_good_pixels / total_valid_pixels
    
    # Paso 2: ¿Tenemos suficientes píxeles buenos?
    if pct_good_pixels < min_pct_good:
        return {'pct75_consecutive': np.nan}
    
    # Paso 3: Filtrar datos
    cons_good = cons_patch[good_pixel_mask]
    
    # Core statistics
    pct75_consecutive = np.nanpercentile(cons_good, 75)

    
    return {
        "pct75_consecutive": pct75_consecutive
    }
    
def summarize_temporal_coverage_patch(obs_patch, valid_patch, observed_days=365, min_valid_threshold=300, min_pct_good=0.5):
    r"""Summarize temporal coverage patch.
    
    Parameters
    ----------
    obs_patch : numpy array
        2D array of maximum consecutive flooded days in the patch (NaN for invalid pixels).
    observed_days : int, optional
        Number of observed days in the period (default is 365).
    
    Returns
    -------
    stats : dict
        Dictionary with max consecutive statistics and quality descriptors.
    """
    
    if obs_patch is None or np.all(np.isnan(obs_patch)):
        return {
            "temporal_coverage": np.nan}
    
    # Paso 1: Identificar píxeles de calidad (≥ 300 días válidos)
    good_pixel_mask = (valid_patch >= min_valid_threshold) & (~np.isnan(obs_patch))
    n_good_pixels = np.sum(good_pixel_mask)
    total_valid_pixels = np.sum(~np.isnan(valid_patch))
    if total_valid_pixels == 0:
        return {'temporal_coverage': np.nan}
    
    pct_good_pixels = n_good_pixels / total_valid_pixels
    
    # Paso 2: ¿Tenemos suficientes píxeles buenos?
    if pct_good_pixels < min_pct_good:
        return {'temporal_coverage': np.nan}
    
    # Paso 3: Filtrar datos
    obs_good = obs_patch[good_pixel_mask]
    
    # Core statistics
    mean_valid_days = np.nanmean(obs_good)
    temporal_coverage = mean_valid_days / observed_days
    
    return {
        "temporal_coverage": temporal_coverage
    }

# ---------- Main process ------------------------------------------
def process_annual_maps(hazard_dir, output_path):
    r"""Process annual hazard maps."""
    
    logger.info("Processing ANNUAL flood hazard maps...")
    
    facilities_csv = '/mnt/staas/CLICHE/00_DATA/valid_facilities_coordinates.csv'
    
    fac_df = pd.read_csv(facilities_csv, header=None, 
                        names=['lon', 'lat', 'Health Facility Name', 'Functional Status'])
    

    hazard_files = sorted(hazard_dir.glob("floods_annual_*.tif"))
    print(f"Number of maps found: {len(hazard_files)}")
    print(f"Total HF in CSV: {len(fac_df)}")
    print(f"Expected iterations if we had the whole country: {len(hazard_files)} × {len(fac_df)} = {len(hazard_files) * len(fac_df)}")

    results = []
        
    for hazard_file in tqdm(hazard_files, desc="Processing years"):
        year = int(hazard_file.stem.split("_")[-1])
        logger.info(f"Processing year: {year}")
        
        try:
            with rasterio.open(hazard_file) as src:
                # Read all bands
                freq = src.read(BAND_CONFIG['frequency'])
                dur = src.read(BAND_CONFIG['duration'])
                valid_obs = src.read(BAND_CONFIG['valid_observations'])
                max_cons = src.read(BAND_CONFIG['max_consecutive'])
                transform = src.transform
                nodata = src.nodata
                observed_days = int(src.tags().get('OBSERVED_DAYS', 365))
                
                for _, fac in fac_df.iterrows():
                    # Extract patches for all bands
                    dur_patch, _, _ = extract_patch(
                        fac.lon, fac.lat, fac["Health Facility Name"],
                        dur, transform, nodata, 'duration'
                    )
                    freq_patch, _, _ = extract_patch(
                        fac.lon, fac.lat, fac["Health Facility Name"],
                        freq, transform, nodata, 'frequency'
                    )
                    valid_patch, valid_cov, min_valid = extract_patch(
                        fac.lon, fac.lat, fac["Health Facility Name"],
                        valid_obs, transform, nodata, 'valid_observations'
                    )
                    cons_patch, _, _ = extract_patch(
                        fac.lon, fac.lat, fac["Health Facility Name"],
                        max_cons, transform, nodata, 'max_consecutive'
                    )
                    
                    # Summarize
                    dur_stats = summarize_duration_patch(dur_patch, valid_patch, min_pct_good=0.3)
                    freq_stats = summarize_frequency_patch(freq_patch, valid_patch, min_pct_good=0.3)
                    cons_stats = summarize_max_consecutive_patch(cons_patch, valid_patch, min_pct_good=0.3)
                    temp_stats = summarize_temporal_coverage_patch(valid_patch, valid_patch, observed_days, min_pct_good=0.3)
                    
                    # Build result row
                    result = {
                        # Identifiers
                        "facility_id": f"{fac['Health Facility Name']}_{fac.lon}_{fac.lat}",
                        "lon": fac.lon,
                        "lat": fac.lat,
                        "facility_name": fac["Health Facility Name"],
                        "facility_status": fac["Functional Status"],
                        "facility_type": "PHCC" if any(x in fac["Health Facility Name"].lower() 
                                                       for x in ['phcc', 'hospital']) else "PHCU",
                        "year": year,
                        
                        "dur_mean": dur_stats["mean_dur"],
                        "pct75_consecutive": cons_stats["pct75_consecutive"],
                        "freq_mean": freq_stats["mean_freq"],
                        "temporal_coverage": temp_stats["temporal_coverage"],
                    }
                    results.append(result)
                    
        except Exception as e:
            logger.error(f"Error processing {hazard_file}: {e}")
            continue
    
    logger.info(f"Total HF-years processed: {len(results)}")
    
    # Create DataFrame
    columns = [
        "facility_id", "lon", "lat", "facility_name", "facility_status", "facility_type", "year",
        "freq_mean",
        "dur_mean", 
        "pct75_consecutive",
        "temporal_coverage"
    ]
    
    out_df = pd.DataFrame(results, columns=columns)
    out_df.to_csv(output_path, index=False)
    print("Mean number of nans: ", out_df.isna().mean())
    logger.info(f"✔ Timeseries saved to {output_path}")
    
    return out_df


# ---------- Execution ----------
if __name__ == "__main__":
    if args.map_type == 'annual':
        process_annual_maps(Path(args.maps_path), Path(args.output_path))
    else: logger.error("Daily maps processing not implemented yet.")