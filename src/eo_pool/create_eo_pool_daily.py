# -*- coding: utf-8 -*-

__author__ = "Macarena Merida Floriano, mmef"
__maintainer__ = "Macarena Merida Floriano, mmef"
__email__ = "mmef@gmv.com"


"""
CHANGE: daily series 
===========================================================================

Script to extract daily flood series for health facilities (HF) using daily 
binary flood maps. This script assumes that the maps have already been binarized 
using the binarize_maps.py script.

Key features:
- AOI determined by raster extent (no shapefile needed)
- HF inside map verification via coordinate transformation
- Date extraction from filename pattern _eYYYYMMDD
- Spatial metrics: binary, % flooded pixels, distance to water

Data is stored as Parquet with metadata, and also as compressed NPZ files 
per HF-year for efficient access. Each facility has a time series of daily 
flood metrics for each year it is covered by the maps.
"""

# --------------- IMPORTS ---------------
from datetime import datetime
import geopandas as gpd
import json, rasterio, re, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from rasterio.windows import Window
from tqdm import tqdm


class FloodSeriesExtractor:
    r"""This class extracts daily flood series for health facilities (HF) 
    using daily binary flood maps.
    
    Methods
    -------
    - `__init__(hf_list_path, daily_maps_dir, output_dir)`: initializes the
    extractor with the HF list, maps directory, and output directory.
    - `_load_hf_list(path)`: loads the HF list from a CSV and converts it to 
    a GeoDataFrame.
    - `_classify_hf_type(nombre)`: classifies the HF type (PHCC/Hospital or PHCU) 
    based on its name.
    - `_get_buffer_size_pixels(hf_type)`: returns the buffer size in pixels based 
    on the HF type.
    - `_pixels_to_meters(n_pixels)`: converts a number of pixels to meters using 
    the raster resolution.
    - `_coord_to_pixel(lat, lon, transform)`: converts geographic coordinates to 
    raster row/column indices.
    - `_is_hf_inside_raster(latitude, longitude, src)`: verifies if HF coordinates
    lie within raster extent and calculates buffer limits.
    - `_extract_date_from_filename(filename)`: extracts the date from the filename 
    using a specific pattern.
    - `_calculate_distance_in_patch(flooded_mask, center_row, center_col, pixel_size_m)`:
    calculates the distance from the HF center to the nearest flooded pixel within the patch.
    - `_extract_metrics_from_patch(patch, row, col, row_min_clip, col_min_clip, row_max_clip, col_max_clip, half, pixel_size_m)`: 
    extracts metrics from a loaded raster patch.
    - `_extract_metrics_from_raster(raster_path, latitude, longitude, buffer_pixels)`: 
    extracts metrics using pixel-based approximation directly from the raster.
    - `_discover_all_maps()`: discovers all daily maps in a directory and organizes 
    the information.
    - `extract_all_series()`: main pipeline that extracts time series for all HFs, 
    optimizing AOI verification.
    - `_save_parquet()`: saves the results as a DataFrame in Parquet format.
    """
    
    def __init__(self, hf_list_path, daily_maps_dir, output_dir):
        """
        Parameters:
        -----------
        hf_list_path : str
            Path to the CSV file containing the list of HFs (columns: name, lat, lon)
            Assumed CRS: EPSG:4326 (WGS84)
        daily_maps_dir : str
            Directory containing daily maps organized by year/subfolders (e.g., daily_maps/2012/*.tif)
        output_dir : str
            Directory to save results (Parquet and NPZ files)
        """
        
        # Working CRS: WGS84 for HF coordinates
        self.hf_crs = "EPSG:4326"
        self.hf_list = self._load_hf_list(hf_list_path)
        self.daily_maps_dir = Path(daily_maps_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.pixel_size_m = 90.0
        self.no_data_value = -9999
        self.flood_value = 1
        self.test_single_hf = None # for all, for specific facility provide ID (ie, "dingding phcu")        
        print(f"Total HFs in list: {len(self.hf_list)}")
        
    def _load_hf_list(self, path):
        r"""Load HF list from CSV.
        
        Parameters
        ----------
        path : str
            Path to the CSV file containing the list of HFs. Expected columns:
            'County_code', 'Payam_code', 'Health Facility Name', 'longitude', 
            'latitude', 'Functional Status'
        
        Returns
        -------
        gdf : gpd.GeoDataFrame
            GeoDataFrame with HF information and geometry in WGS84.
        """
        
        path = Path(path)
        df = pd.read_csv(path)
        
        # Verify required columns
        required_cols = ["County_code", "Payam_code", "Health Facility Name", "longitude", "latitude", "Functional Status"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in CSV: {missing_cols}")
        
        # Create GeoDataFrame in WGS84
        gdf = gpd.GeoDataFrame(
            df, 
            geometry=gpd.points_from_xy(df.longitude, df.latitude),
            crs=self.hf_crs
        )
        return gdf
    
    def _classify_hf_type(self, name):
        r"""Classify HF type based on name. Returns 'PHCC/Hospital' or 'PHCU'.
        
        Parameters
        ----------
        name : str
            Name of the health facility.
        
        Returns
        -------
        str
            'PHCC' if the facility is a PHCC or Hospital, 'PHCU' otherwise.
        """
        
        name_upper = str(name).upper()
        if any(keyword in name_upper for keyword in ['PHCC', 'HOSPITAL']):
            return 'PHCC'
        else:
            return 'PHCU'
    
    
    def _get_buffer_size_pixels(self, hf_type):
        r"""Get buffer size in pixels based on HF type.
        
        Parameters
        ----------
        hf_type : str
            Type of health facility ('PHCC' or 'PHCU').
            
        Returns
        -------
        int
            Buffer size in pixels (side of the square).
        """
        
        if hf_type == 'PHCC':
            return 40  # 40x40 pixels
        else:
            return 60  # 60x60 pixels
    
    def _pixels_to_meters(self, n_pixels):
        r"""Convert pixels to meters.
        
        Parameters
        ----------
        n_pixels : int
            Number of pixels (side of the square).
        
        Returns
        -------
        float
            Size in meters corresponding to the given number of pixels.
        """
        
        return n_pixels * self.pixel_size_m
    
    def _coord_to_pixel(self, lat, lon, transform):
        r"""Convert geographic coordinates to raster row/column indices.
        
        Parameters
        ----------
        lat : float
            Latitude of the point.
        lon : float
            Longitude of the point.
        transform : affine.Affine
            Affine transformation of the raster.
        
        Returns
        -------
        tuple
            (row, col) indices corresponding to the geographic coordinates.
        """
        
        col, row = ~transform * (lon, lat)
        col, row = int(round(col)), int(round(row))
        return row, col
    
    def _is_hf_inside_raster(self, latitude, longitude, src):
        r"""Check if the HF coordinates fall within the raster extent.
        Also calculates buffer limits for later use.
        
        Parameters
        ----------
        latitude : float
            Latitude of the HF.
        longitude : float
            Longitude of the HF.
        src : rasterio.io.DatasetReader
            Opened raster dataset to check against.
        
        Returns
        -------
        tuple
            (bool, row, col, row_buffer_min, row_buffer_max, col_buffer_min, col_buffer_max)        
        """
        
        try:
            # Transform WGS84 coordinates to raster row/column
            row, col = self._coord_to_pixel(latitude, longitude, src.transform)
            # Verify basic limits
            if not (0 <= row < src.height and 0 <= col < src.width):
                return (False, None, None, None, None, None, None)
            
            # Calculate buffer limits in pixels (for complete verification)
            # Assume maximum buffer size (60) for conservative verification
            max_buffer_pixels = 60
            row_min = max(0, row - max_buffer_pixels // 2)
            row_max = min(src.height, row + max_buffer_pixels // 2 + 1)
            col_min = max(0, col - max_buffer_pixels // 2)
            col_max = min(src.width, col + max_buffer_pixels // 2 + 1)
            return (True, row, col, row_min, row_max, col_min, col_max)
                
        except Exception as e:
            return (False, None, None, None, None, None, None)
        
    def _extract_date_from_filename(self, filename):
        r"""Extract date from the pattern _eYYYYMMDD in the filename.
        Example: VIIRS-Flood-5day-..._e202307202359590_...tif -> 2023-07-20
        
        Parameters
        ----------
        filename : str
            Name of the file from which to extract the date.
        Returns
        -------
        str or None
            Extracted date in 'YYYY-MM-DD' format, or None if not found.
        """
        
        # Search for pattern _e followed by 8 digits (YYYYMMDD)
        match = re.search(r'_e(\d{4})(\d{2})(\d{2})\d+_', filename)
        
        if match:
            year = match.group(1)
            month = match.group(2)
            day = match.group(3)
            return f"{year}-{month}-{day}"
        # Fallback
        match = re.search(r'(\d{4})(\d{2})(\d{2})', filename)
        if match:
            return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
        
        return None
    
    def _calculate_distance_in_patch(self, flooded_mask, center_row, center_col, pixel_size_m):
        r"""Calculate distance from center to nearest flooded pixel.
        
        Parameters
        ----------
        flooded_mask : 2D np.array of bool
            Mask indicating which pixels in the patch are flooded.
        center_row : int
            Row index of the center pixel.
        center_col : int
            Column index of the center pixel.
        pixel_size_m : float
            Size of a pixel in meters.
        
        Returns
        -------
        float or None
            Distance from the center to the nearest flooded pixel in meters, 
            or None if no flooded pixels are present.
        """
        
        if not np.any(flooded_mask):
            return None  # No flooded pixels present
        
        if flooded_mask[center_row, center_col]:
            return 0.0  # Center is flooded
        
        flooded_indices = np.argwhere(flooded_mask)
        distances_pixels = np.sqrt(
            (flooded_indices[:, 0] - center_row)**2 +
            (flooded_indices[:, 1] - center_col)**2
        )
        min_dist_pixels = np.min(distances_pixels)
        return min_dist_pixels * pixel_size_m
    
    def _extract_metrics_from_patch(self, patch, row, col, row_min_clip, col_min_clip,
                                    row_max_clip, col_max_clip, half, pixel_size_m):
        r"""Extract metrics from a loaded raster patch.
        
        Parameters
        ----------
        patch : 2D np.array
            Raster values in the patch.
        row : int 
            Row index of the HF center in the original raster.
        col : int
            Column index of the HF center in the original raster.
        row_min_clip, col_min_clip, row_max_clip, col_max_clip : int
            Clipped limits of the patch in the original raster.
        half : int
            Half of the buffer size in pixels (e.g., 20 for PHCC, 30 for PHCU).
        pixel_size_m : float
            Size of a pixel in meters.
        
        Returns
        -------
        dict
            Dictionary containing the extracted metrics:
            - 'binary': 1 if any pixel is flooded, 0 otherwise
            - 'pct_pixels': % of valid pixels that are flooded
            - 'distance_m': distance from center to nearest flooded pixel in meters
            - 'n_no_data': number of no-data pixels in the patch
            - 'n_valid_pixels': number of valid pixels in the patch
            - 'total_pixels_buffer': total number of pixels expected in the buffer
            - 'pct_valid_data': % of the buffer that has valid data
            - 'inside_aoi': True if HF is inside AOI, False otherwise
            - 'buffer_partial': True if buffer was partially outside raster, False if complete
        """
        
        # HF center in patch coordinates
        center_row_patch = row - row_min_clip
        center_col_patch = col - col_min_clip
        
        # Check if center is in the patch
        if not (0 <= center_row_patch < patch.shape[0] and 
                0 <= center_col_patch < patch.shape[1]):
            return {
                'binary': np.nan,
                'pct_pixels': np.nan,
                'distance_m': np.nan,
                'n_no_data': patch.size,
                'n_valid_pixels': 0,
                'total_pixels_buffer': (2*half)**2,
                'pct_valid_data': 0.0,
                'inside_aoi': True,
                'buffer_partial': True
            }
        
        # Masks
        no_data_mask = (patch == self.no_data_value)
        valid_mask = ~no_data_mask
        flooded_mask = (patch == self.flood_value) & valid_mask
        
        n_total_expected = (2 * half) * (2 * half)
        n_valid = np.sum(valid_mask)
        n_flooded = np.sum(flooded_mask)
        n_no_data = n_total_expected - n_valid
        
        # % of valid pixels flooded
        pct_pixels = (n_flooded / n_valid) * 100 if n_valid > 0 else 0.0
        
        # % of the total buffer that is valid data
        pct_valid_data = (n_valid / n_total_expected) * 100
        
        # Binary
        binary = 1 if n_flooded > 0 else 0
        
        # Distance to water
        distance_m = self._calculate_distance_in_patch(
            flooded_mask, center_row_patch, center_col_patch, pixel_size_m
        )
        
        # Detect if buffer was clipped
        buffer_partial = (row_min_clip != row - half or 
                         col_min_clip != col - half or
                         row_max_clip != row + half or
                         col_max_clip != col + half)
        
        return {
            'binary': int(binary),
            'pct_pixels': float(pct_pixels),
            'distance_m': float(distance_m) if distance_m is not None else np.nan,
            'n_no_data': int(n_no_data),
            'n_valid_pixels': int(n_valid),
            'total_pixels_buffer': int(n_total_expected),
            'pct_valid_data': float(pct_valid_data),
            'inside_aoi': True,
            'buffer_partial': buffer_partial
        }
    
    def _extract_metrics_from_raster(self, raster_path, latitude, longitude, buffer_pixels):
        r"""Extract metrics using pixel-based approximation directly from the raster.
        This method performs a quick check using pixel-based approximation without
        loading the entire patch into memory. It checks the center pixel and its immediate
        neighborhood to determine if the HF is flooded, and estimates the percentage of flooded pixels
        and distance to water based on a small window around the center pixel.
        
        Parameters
        ----------
        raster_path : str or Path
            Path to the raster file to extract metrics from.
        latitude : float
            Latitude of the HF.
        longitude : float
            Longitude of the HF.
        buffer_pixels : int
            Size of the buffer in pixels (side of the square).
        
        Returns
        -------
        dict
            Dictionary containing the extracted metrics (same format 
            as _extract_metrics_from_patch).
        """
        
        try:
            with rasterio.open(raster_path) as src:
                # Transform point to pixels
                col, row = ~src.transform * (longitude, latitude)
                col, row = int(round(col)), int(round(row))
                
                # Check if center is within the raster
                if not (0 <= row < src.height and 0 <= col < src.width):
                    return {'inside_aoi': False}
                
                half = buffer_pixels // 2
                
                # Desired bounds
                row_min, row_max = row - half, row + half
                col_min, col_max = col - half, col + half
                
                # Clip to raster bounds
                row_min_clip = max(row_min, 0)
                col_min_clip = max(col_min, 0)
                row_max_clip = min(row_max, src.height)
                col_max_clip = min(col_max, src.width)
                
                # Check for overlap
                if row_min_clip >= row_max_clip or col_min_clip >= col_max_clip:
                    total_pixels = buffer_pixels * buffer_pixels
                    return {
                        'binary': 0,
                        'pct_pixels': 0.0,
                        'distance_m': np.nan,
                        'n_no_data': total_pixels,
                        'n_valid_pixels': 0,
                        'total_pixels_buffer': total_pixels,
                        'pct_valid_data': 0.0,
                        'inside_aoi': True,
                        'buffer_partial': True
                    }
                
                # Read patch
                window = Window(col_min_clip, row_min_clip,
                               col_max_clip - col_min_clip,
                               row_max_clip - row_min_clip)
                patch = src.read(1, window=window)
                
                # Calculate resolution in meters
                pixel_size_m = 90
                
                # Extract metrics from the patch
                return self._extract_metrics_from_patch(
                    patch, row, col, row_min_clip, col_min_clip,
                    row_max_clip, col_max_clip, half, pixel_size_m
                )
                
        except Exception as e:
            warnings.warn(f"Error processing {raster_path}: {e}")
            return {'inside_aoi': False, 'error': str(e)}
            
    def _discover_all_maps(self):
        r"""Discover all daily maps in a single directory.
        No subfolders by year.
        
        Returns
        -------
        files_by_year : dict
            Dictionary organized by year with lists of maps and their metadata.
        valid_files : list
            List of all valid maps with their metadata.
        """
        
        print(f"\nSearching for maps in: {self.daily_maps_dir}")
        
        # Search for all .tif files in the directory
        all_files = sorted(self.daily_maps_dir.glob('*.tif*'))
        
        if not all_files:
            raise ValueError(f"No .tif files found in {self.daily_maps_dir}")
        
        print(f"Total files found: {len(all_files)}")
        
        # Extract dates and organize
        valid_files = []
        years_found = set()
        
        for f in all_files:
            date_str = self._extract_date_from_filename(f.name)
            if date_str:
                year = int(date_str.split('-')[0])
                years_found.add(year)
                valid_files.append({
                    'path': f,
                    'date': date_str,
                    'year': year,
                    'filename': f.name
                })
            else:
                warnings.warn(f"Could not extract date from: {f.name}")
        
        if not valid_files:
            raise ValueError("Could not extract date from any file")
        
        # Organize by year for ordered processing
        files_by_year = {}
        for item in valid_files:
            year = item['year']
            if year not in files_by_year:
                files_by_year[year] = []
            files_by_year[year].append(item)
        
        # Sort each year by date
        for year in files_by_year:
            files_by_year[year].sort(key=lambda x: x['date'])
        
        print(f"Valid maps: {len(valid_files)}")
        print(f"Years found: {sorted(years_found)}")
        for year in sorted(files_by_year.keys()):
            print(f"  {year}: {len(files_by_year[year])} maps")
        
        return files_by_year, valid_files
    
    def extract_all_series(self):
        r"""Main pipeline: extracts time series for all HFs.
        OPTIMIZED: Precalculated AOI verification once per year.
        
        Returns
        -------
        results : list
            List of dictionaries with the extracted time series and metadata for each HF-year.
        """
        
        files_by_year, all_files = self._discover_all_maps()
        all_years = sorted(files_by_year.keys())
        
        print(f"\nProcessing {len(self.hf_list)} HFs...")
        print(f"Total maps to process: {len(all_files)}")
        
        # OPTIMIZATION: Precalculate which HFs are inside each year (ONLY ONCE PER YEAR)
        print("Precalculating spatial coverage by year...")
        hf_inside_by_year = {}
        for year in tqdm(all_years, desc="Verifying AOI by year"):
            year_maps = files_by_year[year]
            if len(year_maps) == 0:
                hf_inside_by_year[year] = np.zeros(len(self.hf_list), dtype=bool)
                continue
            
            first_map = year_maps[0]
            inside_flags = []
            
            try:
                with rasterio.open(first_map['path']) as src:
                    for idx, hf_row in self.hf_list.iterrows():
                        is_inside, _, _, _, _, _, _ = self._is_hf_inside_raster(
                            hf_row['latitude'], hf_row['longitude'], src
                        )
                        inside_flags.append(is_inside)
            except Exception as e:
                warnings.warn(f"Error verifying year {year}: {e}")
                inside_flags = [False] * len(self.hf_list)
            
            hf_inside_by_year[year] = np.array(inside_flags)
            n_inside = np.sum(hf_inside_by_year[year])
            print(f"  Year {year}: {n_inside}/{len(self.hf_list)} HFs inside the AOI")
        
        results = []
        stats = {
            'hf_processed': 0,
            'hf_outside_all_maps': set(),
            'hf_inside_some_maps': set(),
            'total_hf_maps_checks': 0,
            'hf_map_inside': 0,
            'hf_map_outside': 0,
            'maps_with_no_data_issues': 0
        }
        
        print(f"DEBUG: len(hf_list) = {len(self.hf_list)}")
        print(f"DEBUG: hf_list index = {self.hf_list.index.tolist()[:10]}...")
        print(f"DEBUG: hf_inside_by_year[2012] shape = {hf_inside_by_year[2012].shape}")
        
        hf_subset = self.hf_list[self.hf_list['Health Facility Name'] == self.test_single_hf].reset_index(drop=True) if self.test_single_hf else self.hf_list.reset_index(drop=True)
        # For each HF
        for pos, (idx, hf_row) in enumerate(tqdm(hf_subset.iterrows(), 
                                total=len(hf_subset),
                                desc="Processing HF")):
            
            hf_county = hf_row['County_code']
            hf_payam = hf_row['Payam_code']
            hf_id = hf_row['Health Facility Name']
            hf_lat = hf_row['latitude']
            hf_lon = hf_row['longitude']
            hf_point = (hf_lat, hf_lon)
            
            # Classify type and determine buffer size in pixels
            hf_type = self._classify_hf_type(hf_id)
            buffer_pixels = self._get_buffer_size_pixels(hf_type)
            buffer_meters = self._pixels_to_meters(buffer_pixels)
            
            was_inside_any_map = False
            
            # For each year, process only if inside (using precalculation)
            for year in all_years:
                year_maps = files_by_year[year]
                n_days = len(year_maps)
                
                if n_days == 0:
                    continue
                
                # USE PRECALCULATION instead of opening the raster
                if not hf_inside_by_year[year][idx]:
                    print("HF OUTSIDE AOI")
                    stats['hf_map_outside'] += n_days
                    stats['total_hf_maps_checks'] += n_days
                    continue
                
                was_inside_any_map = True
                
                # Initialize arrays for this HF-year
                series_binary = np.zeros(n_days, dtype=np.int8)
                series_pct_flood = np.zeros(n_days, dtype=np.float32)
                series_dist = np.full(n_days, np.nan, dtype=np.float32)
                series_no_data = np.zeros(n_days, dtype=np.int32)
                dates = []
                
                # Process each day of the year
                for day_idx, map_info in enumerate(tqdm(year_maps, desc=f"{hf_id[:20]} {year}", 
                                                         leave=False)):
                    stats['total_hf_maps_checks'] += 1
                    stats['hf_map_inside'] += 1
                    
                    dates.append(map_info['date'])
                    
                    metrics = self._extract_metrics_from_raster(
                        map_info['path'], hf_lat, hf_lon, buffer_pixels)
                    
                    # Handle NaN values (no-data or error)
                    if not metrics['inside_aoi']:
                        series_binary[day_idx] = 0
                        series_pct_flood[day_idx] = 0.0
                        series_dist[day_idx] = np.nan
                        series_no_data[day_idx] = -1
                    elif np.isnan(metrics['binary']):
                        series_binary[day_idx] = 0
                        series_pct_flood[day_idx] = 0.0
                        series_dist[day_idx] = np.nan
                        series_no_data[day_idx] = metrics['n_no_data']
                        stats['maps_with_no_data_issues'] += 1
                    else:
                        series_binary[day_idx] = metrics['binary']
                        series_pct_flood[day_idx] = metrics['pct_pixels']
                        series_dist[day_idx] = metrics['distance_m']
                        series_no_data[day_idx] = metrics['n_no_data']
                
                # Save result for this year
                result = {
                    'hf_county': hf_county,
                    'hf_payam': hf_payam,
                    'hf_id': hf_id,
                    'hf_type': hf_type,
                    'latitude': hf_lat,
                    'longitude': hf_lon,
                    'year': year,
                    'n_days': n_days,
                    'buffer_pixels': buffer_pixels,
                    'buffer_meters': buffer_meters,
                    'total_pixels_buffer': buffer_pixels * buffer_pixels,
                    'dates': dates,
                    'series_binary': series_binary,
                    'series_pct_flood_pixels': series_pct_flood,
                    'series_distance_m': series_dist,
                    'series_no_data_count': series_no_data,
                    'total_flooded_days': int(np.nansum(series_binary)),
                }
                results.append(result)
            
            # Update statistics
            stats['hf_processed'] += 1
            if was_inside_any_map:
                stats['hf_inside_some_maps'].add(hf_id)
            else:
                stats['hf_outside_all_maps'].add(hf_id)
        
        self.results = results
        self.stats = stats
        
        # Final report
        print(f"\n{'='*70}")
        print("PROCESSING SUMMARY")
        print(f"{'='*70}")
        print(f"HFs processed: {stats['hf_processed']}")
        print(f"HFs inside at least one map: {len(stats['hf_inside_some_maps'])}")
        print(f"HFs outside all maps: {len(stats['hf_outside_all_maps'])}")
        if stats['hf_outside_all_maps']:
            hf_out_list = list(stats['hf_outside_all_maps'])[:5]
            print(f"  Examples: {', '.join(hf_out_list)}...")
        print(f"\nTotal HF-map checks: {stats['total_hf_maps_checks']:,}")
        print(f"  Inside AOI: {stats['hf_map_inside']:,}")
        print(f"  Outside AOI: {stats['hf_map_outside']:,}")
        print(f"\nMaps with no-data issues: {stats['maps_with_no_data_issues']}")
        print(f"HF-year records generated: {len(results)}")
        
        return results
    
    def _save_parquet(self):
        r"""Save as DataFrame in Parquet format.
        
        Returns
        -------
        df : pd.DataFrame
            DataFrame containing the results, or empty DataFrame if no results to save.
        """
        
        if not self.results:
            print("WARNING: No results to save")
            return pd.DataFrame()
        
        records = []
        
        for res in self.results:
            # Convert arrays to lists for serialization
            binary_list = [int(x) for x in res['series_binary']]
            pct_list = [round(float(x), 2) for x in res['series_pct_flood_pixels']]
            dist_list = [
                round(float(d), 2) if not np.isnan(d) else None 
                for d in res['series_distance_m']
            ]
            no_data_list = [int(x) for x in res['series_no_data_count']]
            
            records.append({
                'hf_county': res['hf_county'],
                'hf_payam': res['hf_payam'],
                'hf_id': res['hf_id'],
                'hf_type': res['hf_type'],
                'latitude': res['latitude'],
                'longitude': res['longitude'],
                'year': res['year'],
                'n_days': res['n_days'],
                'buffer_pixels': res['buffer_pixels'],
                'buffer_meters': res['buffer_meters'],
                'total_pixels_buffer': res['total_pixels_buffer'],
                'dates': json.dumps(res['dates']),
                'series_binary': json.dumps(binary_list),
                'series_pct_flood_pixels': json.dumps(pct_list),
                'series_distance_m': json.dumps(dist_list),
                'series_no_data_count': json.dumps(no_data_list),
                'total_flooded_days': res['total_flooded_days'],
            })
        
        df = pd.DataFrame(records)
        output_path = self.output_dir / 'eo_pool.parquet'
        df.to_parquet(output_path, compression='zstd')
        
        size_mb = output_path.stat().st_size / 1024 / 1024
        print(f"\nSAVED TO {output_path}")
        print(f"SIZE: {size_mb:.2f} MB")
        print(f"RECORDS: {len(df)}")
        
        # Metadata
        metadata = {
            'n_hf_total': len(self.hf_list),
            'n_hf_inside_maps': len(self.stats['hf_inside_some_maps']),
            'n_hf_outside_maps': len(self.stats['hf_outside_all_maps']),
            'n_hf_year_records': len(df),
            'years': sorted(df['year'].unique().tolist()),
            'hf_types': df['hf_type'].value_counts().to_dict(),
            'buffer_config': {
                'PHCC/Hospital': '40x40 pixels (3600m)',
                'PHCU': '60x60 pixels (5400m)',
                'pixel_size_m': self.pixel_size_m
            },
            'raster_values': {
                '0': 'no flood',
                '1': 'flood',
                '-9999': 'no-data'
            },
            'date_generated': datetime.now().isoformat()
        }
        
        with open(self.output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Resumen CSV
        summary_cols = ['hf_id', 'hf_type', 'latitude', 'longitude', 'year', 
                       'n_days', 'total_flooded_days', 'buffer_pixels', 
                       'buffer_meters']
        summary = df[summary_cols].copy()
        summary.to_csv(self.output_dir / 'summary_hf_years.csv', index=False)
        return df
    
    def _save_npz(self):
        r"""Save compressed numpy arrays per HF-year.
        """
        
        for res in self.results:
            filename = f"{res['hf_id'].replace(' ', '_')}_{res['year']}.npz"
            np.savez_compressed(
                self.output_dir / filename,
                binary=res['series_binary'],
                pct_flood_pixels=res['series_pct_flood_pixels'],
                distance_m=res['series_distance_m'],
                no_data_count=res['series_no_data_count'],
                dates=np.array(res['dates']),
                metadata=np.array([
                    res['hf_id'], res['hf_type'], res['latitude'], 
                    res['longitude'], res['year'], res['n_days'],
                    res['buffer_pixels'], res['buffer_meters']
                ])
            )
            
    def save_results(self, format='parquet'):
        r"""Save results in an efficient format.
        
        Parameters
        ----------
        format : str
            Format to save results. Options: 'parquet' (default), 'npz'.
        """
        
        if format == 'parquet':
            return self._save_parquet()
        elif format == 'npz':
            return self._save_npz()
        else:
            raise ValueError(f"Format {format} not supported")
        

def main():
    r"""Example usage of the extractor.
    To use with all HFs, set `test_single_hf` to None in the configuration.
    """
    
    # CONFIGURATION
    CONFIG = {
        'hf_list_path': '../data/valid_facilities.csv',
        'daily_maps_dir': '../data/flood_maps/binary_maps/',  # Single folder, not by years
        'output_dir': '../data/eo_pool/'
    }
    
    print("="*70)
    print("FLOOD SERIES EXTRACTOR")
    print("="*70)
    print(f"Configuration:")
    print(f"  HF list: {CONFIG['hf_list_path']}")
    print(f"  Maps: {CONFIG['daily_maps_dir']}")
    print(f"  Output: {CONFIG['output_dir']}")
    print()
    
    # Initialize
    extractor = FloodSeriesExtractor(
        hf_list_path=CONFIG['hf_list_path'],
        daily_maps_dir=CONFIG['daily_maps_dir'],
        output_dir=CONFIG['output_dir']
    )
    
    print(f"    Facility: {extractor.test_single_hf}")
    # Extract series
    results = extractor.extract_all_series()
    # Save
    df = extractor.save_results(format='parquet')
    
    # Sample of results
    print(f"\n{'='*70}")
    print("SAMPLE OF RESULTS")
    print(f"{'='*70}")
    
    if results:
        sample = results[0]
        print(f"\nHF: {sample['hf_id']}")
        print(f"Type: {sample['hf_type']}")
        print(f"Buffer: {sample['buffer_pixels']}x{sample['buffer_pixels']} pixels = {sample['buffer_meters']}m")
        print(f"Year: {sample['year']}, DDays: {sample['n_days']}")
        print(f"Flooded days: {sample['total_flooded_days']}")
        
        print(f"\nFirst 10 days:")
        print(f"  Dates: {sample['dates'][:10]}")
        print(f"  Binary: {sample['series_binary'][:10].tolist()}")
        print(f"  % PPixels: {[round(x,1) for x in sample['series_pct_flood_pixels'][:10]]}")
        print(f"  Distance (m): {[round(x,1) if not np.isnan(x) else None for x in sample['series_distance_m'][:10]]}")
        
        sample = results[1]
        print(f"\nHF: {sample['hf_id']}")
        print(f"Type: {sample['hf_type']}")
        print(f"Buffer: {sample['buffer_pixels']}x{sample['buffer_pixels']} pixels = {sample['buffer_meters']}m")
        print(f"Year: {sample['year']}, DDays: {sample['n_days']}")
        print(f"Flooded days: {sample['total_flooded_days']}")
        
        print(f"\nFirst 10 days:")
        print(f"  Dates: {sample['dates'][:10]}")
        print(f"  Binary: {sample['series_binary'][:10].tolist()}")
        print(f"  % PPixels: {[round(x,1) for x in sample['series_pct_flood_pixels'][:10]]}")
        print(f"  Distance (m): {[round(x,1) if not np.isnan(x) else None for x in sample['series_distance_m'][:10]]}")
    
    return extractor

if __name__ == "__main__":
    extractor = main()