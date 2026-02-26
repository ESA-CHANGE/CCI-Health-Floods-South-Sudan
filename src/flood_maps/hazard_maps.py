# -*- coding: utf-8 -*-

__author__ = "Macarena Merida Floriano, mmef"
__maintainer__ = "Macarena Merida Floriano, mmef"
__email__ = "mmef@gmv.com"


"""
CHANGE: data processing script
========================================

Script to create hazard flood maps for South Sudan.
This script assumes binary maps are available. The output is a 
raster per year with two bands or two maps per year (seasonal):

- Band 1: Frequency of flooding (number of times a pixel is flooded / total number of maps)
- Band 2: Flooding duraction (number of times a pixel is flooded).

"""

import argparse
import re, os
from datetime import datetime
import rasterio
import numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm


def extract_date_from_filename(fname):
    match = re.search(r"_e(\d{8})", fname)
    if not match:
        raise ValueError(f"No date found in {fname}")
    return datetime.strptime(match.group(1), "%Y%m%d")

def get_season(year, month):
    """
    Returns season label and season-year.
    Dry season: Nov (11) to Apr (4)
    Wet season: May (5) to Oct (10)
    """
    if month >= 11:
        return "dry", year + 1   # Nov–Dec belong to next hydrological year
    elif month <= 4:
        return "dry", year
    else:
        return "wet", year


def max_consecutive_flood_with_nodata_tolerance(
    ts,
    nodata=-9999,
    max_nodata_gap=3
):
    """
    ts: 1D array (time) for one pixel
    Returns maximum consecutive flooded days (1s),
    allowing up to max_nodata_gap nodata values to be interpolated as 1.
    """

    max_run = 0
    current_run = 0
    nodata_count = 0

    for v in ts:
        if v == 1:
            current_run += 1
            nodata_count = 0

        elif v == nodata:
            nodata_count += 1
            if nodata_count <= max_nodata_gap:
                current_run += 1
            else:
                max_run = max(max_run, current_run)
                current_run = 0
                nodata_count = 0

        else:  # v == 0
            max_run = max(max_run, current_run)
            current_run = 0
            nodata_count = 0

    max_run = max(max_run, current_run)
    return max_run


# Instantiate argument parser
parser = argparse.ArgumentParser(description='CHANGE - Flood hazard maps generation')
parser.add_argument('--map_type', type=str, choices=['annual', 'annual_2', 'annual_3', 'seasonal', 'seasonal_2'], default='annual',
                    help='Type of hazard map to generate: annual or seasonal (default: annual)')
args = parser.parse_args()


# Get binary maps
tif_dir = Path("/mnt/staas/CLICHE/00_DATA/bin_maps/")
files = sorted(tif_dir.glob("*.tif"))

if args.map_type == 'annual':
    print("Generating ANNUAL flood hazard maps...")

    files_by_year = defaultdict(list)
    print("There are {} files".format(len(files)))
    print("Grouping files by year...")
    for f in files:
        date = extract_date_from_filename(f.name)
        files_by_year[date.year].append((date, f))
        
    # Annual processing
    output_dir = Path("/mnt/staas/CLICHE/00_DATA/hazard_maps/annual/")
    output_dir.mkdir(exist_ok=True)
    for year, flist in sorted(files_by_year.items()):
        print("Processing year:", year)
        arrays = []

        for _, f in sorted(flist):
            with rasterio.open(f) as src:
                arr = src.read(1).astype("float32")

                # Save base profile just once
                if not arrays:
                    profile = src.profile.copy()
                    nodata_in = src.nodata

                # Hadle NoData
                if nodata_in is not None:
                    arr = np.where(arr == nodata_in, np.nan, arr)

                arrays.append(arr)

        # Stack temporal: (time, rows, cols)
        stack = np.stack(arrays, axis=0)

        # =========================
        # Maps: frecuency and duration
        # =========================
        flood_frequency = np.nanmean(stack, axis=0)                 # [0–1]
        flood_duration = np.nansum(stack, axis=0)  # días/año

        # Replace NaNs with nodata
        flood_frequency = np.where(np.isnan(flood_frequency), nodata_in, flood_frequency)
        flood_duration = np.where(np.isnan(flood_duration), nodata_in, flood_duration)

        # =========================
        # Output profile
        # =========================
        profile.update(
            dtype="float32",
            count=2,
            nodata=nodata_in,
            compress="deflate"
        )

        out_file = output_dir / f"flood_hazard_metrics_{year}.tif"

        # =========================
        # Multiband output
        # =========================
        with rasterio.open(out_file, "w", **profile) as dst:
            dst.write(flood_frequency.astype("float32"), 1)
            dst.write(flood_duration.astype("float32"), 2)

            dst.set_band_description(1, "Flood frequency (probability)")
            dst.set_band_description(2, "Flood duration (days per year)")

            dst.update_tags(
                PRODUCT="Flood hazard metrics",
                DESCRIPTION="Flood frequency and duration derived from VIIRS binary flood maps",
                FLOOD_FREQUENCY="Probability of pixel being inundated on a given day",
                FLOOD_DURATION="Total number of inundated days per year",
                TEMPORAL_RESOLUTION="Daily, from 5 days accumulated VIIRS flood maps",
                PERIOD=str(year),
                REGION="South Sudan",
                SOURCE="VIIRS flood extent product",
                PROCESSING="Annual aggregation of binary flood maps (0/1)"
            )

        print(f"✔ Generado: {out_file}")

elif args.map_type == 'annual_2':
    print("Generating ANNUAL flood hazard maps from seasonal_2 hazard maps...")
    
    tif_dir = Path("/mnt/staas/CLICHE/00_DATA/hazard_maps/seasonal_check/")
    files = sorted(tif_dir.glob("*.tif"))
    
    output_dir = Path("/mnt/staas/CLICHE/00_DATA/hazard_maps/annual_check/")
    output_dir.mkdir(exist_ok=True)
    
    groups = defaultdict(list)
    for f in files:
        match = re.search(r"_(dry|wet)_(\d{4})", f.name)
        if not match:
            raise ValueError(f"No season-year found in {f.name}")
        season, year = match.groups()
        groups[year].append((season, f))
    
    for year, flist in tqdm(groups.items(), desc="Processing years"):
        print(f"\n→ Year {year} | {len(flist)} seasonal maps")

        # Initialize accumulators
        count_flooded = None
        count_valid = None
        count_max = None

        for season, f in flist:
            with rasterio.open(f) as src:
                data = src.read()  # (3, rows, cols)
                nodata = src.nodata

                if count_flooded is None:
                    count_flooded = np.zeros(data.shape[1:], dtype=np.int32)
                    count_valid = np.zeros(data.shape[1:], dtype=np.int32)
                    count_max = np.zeros(data.shape[1:], dtype=np.int32)

                # Accumulate flooded days and valid observations
                count_flooded += data[0]  # flooded days
                count_valid += data[1]     # valid obs
                count_max = np.maximum(count_max, data[2])  # max consecutive flooded days

        # Calculate annual metrics
        flood_frequency = np.where(count_valid > 0, count_flooded / count_valid, nodata)
        flood_duration = np.where(count_valid > 0, count_flooded, nodata)
        flood_valid_observations = np.where(count_valid > 0, count_valid, nodata)
        flood_max_duration = np.where(count_valid > 0, count_max, nodata)

        # Save output raster (4 bands)
        with rasterio.open(flist[0][1]) as src_ref:
            profile = src_ref.profile

        profile.update(
            dtype="float32",
            count=4,
            nodata=nodata,
            compress="deflate"
        )

        out_file = output_dir / f"flood_hazard_metrics_annual_{year}.tif"

        with rasterio.open(out_file, "w", **profile) as dst:
            dst.write(flood_frequency.astype("float32"), 1)
            dst.write(flood_duration.astype("float32"), 2)
            dst.write(flood_valid_observations.astype("float32"), 3)
            dst.write(flood_max_duration.astype("float32"), 4)
            dst.set_band_description(1, "Flood frequency (probability)")
            dst.set_band_description(2, "Flood duration (days per year)")
            dst.set_band_description(3, "Valid flood observations")
            dst.set_band_description(4, "Maximum consecutive flooded days")

            dst.update_tags(
                PRODUCT="Annual flood hazard metrics (from seasonal_2)",
                DESCRIPTION="Annual flood frequency and duration derived from manual seasonal flood diagnostics",
                FLOOD_FREQUENCY="Probability of pixel being inundated on a given day",
                FLOOD_DURATION="Total number of inundated days per year",
                FLOOD_VALID_OBSERVATIONS="Number of valid flood observations in the year",
                FLOOD_MAX_DURATION="Maximum consecutive flooded days",
                TEMPORAL_RESOLUTION="Daily, from manual seasonal flood diagnostics",
                PERIOD=str(year),
                REGION="South Sudan",
                SOURCE="Manual aggregation of seasonal flood diagnostics",
                PROCESSING="Annual aggregation of seasonal_2 hazard maps"
            )

        print(f"✔ Generado: {out_file}")

elif args.map_type == 'seasonal':
    
    output_dir = Path("/mnt/staas/CLICHE/00_DATA/hazard_maps/seasonal/")
    output_dir.mkdir(exist_ok=True)
    # Group files by (season, year)
    print("\nGenerating SEASONAL flood hazard maps...")
    groups = defaultdict(list)

    for f in files:
        date = extract_date_from_filename(f.name)
        season, season_year = get_season(date.year, date.month)
        groups[(season_year, season)].append(f)
    
    for (year, season), flist in tqdm(groups.items(), desc="Processing seasons"):
        arrays = []
        for fname in flist:
            with rasterio.open(os.path.join(tif_dir, fname)) as src:
                arr = src.read(1).astype("float32")
                nodata = src.nodata
                if nodata is not None:
                    arr[arr == nodata] = np.nan
                arrays.append(arr)

        # Stack temporal: (time, rows, cols)
        stack = np.stack(arrays, axis=0)

        # =========================
        # Hazard metrics
        # =========================
        flood_duration = np.nansum(stack, axis=0)       # days flooded
        flood_frequency = np.nanmean(stack, axis=0)     # fraction of days

        # =========================
        # Save multiband raster
        # =========================
        with rasterio.open(os.path.join(tif_dir, flist[0])) as src_ref:
            profile = src_ref.profile

        profile.update(
            count=2,
            dtype="float32",
            nodata=np.nan,
            compress="deflate"
        )

        out_name = f"flood_hazard_{season}_{year}.tif"
        out_file = os.path.join(output_dir, out_name)

        with rasterio.open(out_file, "w", **profile) as dst:
            dst.write(flood_frequency.astype("float32"), 1)
            dst.write(flood_duration.astype("float32"), 2)

            dst.set_band_description(1, "flood_frequency")
            dst.set_band_description(2, "flood_duration")

            dst.update_tags(
                season=season,
                season_year=year,
                description="Seasonal flood hazard metrics",
                band_1="Flood frequency (fraction of flooded days)",
                band_2="Flood duration (number of flooded days)"
            )

        print(f"Saved: {out_name}")
        
elif args.map_type == "annual_3":
    print("\nGenerating ANNUAL flood hazard maps (OPTIMIZED) from daily maps...")
    
    tif_dir = Path("/mnt/staas/CLICHE/00_DATA/bin_maps/daily_new/")
    files = sorted(tif_dir.glob("*.tif"))
    MAX_NODATA_GAP = 2
    
    output_dir = Path("/mnt/staas/CLICHE/00_DATA/hazard_maps/annual_daily_new/")
    output_dir.mkdir(exist_ok=True)
    
    # Agrupar por año
    files_by_year = defaultdict(list)
    print(f"There are {len(files)} files")
    print("Grouping files by year...")
    
    for f in files:
        date = extract_date_from_filename(f.name)
        files_by_year[date.year].append((date, f))
    
    for year, flist in tqdm(sorted(files_by_year.items()), desc="Processing years"):
        print(f"\nProcessing year: {year}")
        
        flist_sorted = sorted(flist, key=lambda x: x[0])
        n_days = len(flist_sorted)
        print(f"  → {n_days} daily maps")
        
        # Obtener dimensiones del primer archivo
        with rasterio.open(flist_sorted[0][1]) as src:
            nrows, ncols = src.height, src.width
            profile = src.profile.copy()
            nodata_in = src.nodata
            print(f"  → Dimensions: {nrows} x {ncols}")
            print(f"  → NoData value: {nodata_in}")
        
        # Configuración de bloques
        BLOCK_HEIGHT = 500
        n_blocks = (nrows + BLOCK_HEIGHT - 1) // BLOCK_HEIGHT
        print(f"  → Processing in {n_blocks} blocks of {BLOCK_HEIGHT} rows")
        
        # Arrays de salida completos (se llenan por bloques)
        count_flooded_full = np.zeros((nrows, ncols), dtype=np.int32)
        count_valid_full = np.zeros((nrows, ncols), dtype=np.int32)
        max_consecutive_full = np.zeros((nrows, ncols), dtype=np.int32)
        
        # =================================================================
        # PROCESAMIENTO POR BLOQUES
        # =================================================================
        
        for block_idx in tqdm(range(n_blocks), desc="Processing blocks", leave=False):
            row_start = block_idx * BLOCK_HEIGHT
            row_end = min(row_start + BLOCK_HEIGHT, nrows)
            n_rows_block = row_end - row_start
            
            # Acumuladores simples para este bloque
            count_flooded = np.zeros((n_rows_block, ncols), dtype=np.int32)
            count_valid = np.zeros((n_rows_block, ncols), dtype=np.int32)
            
            # Estado para cálculo de consecutivos (4 arrays de n_rows_block x ncols)
            current_streak = np.zeros((n_rows_block, ncols), dtype=np.int32)
            max_streak = np.zeros((n_rows_block, ncols), dtype=np.int32)
            nodata_gap_count = np.zeros((n_rows_block, ncols), dtype=np.int32)
            
            # =================================================================
            # LOOP DIARIO: Un solo raster abierto a la vez, sin acumular historial
            # =================================================================
            
            for day_idx, (date, f) in enumerate(flist_sorted):
                with rasterio.open(f) as src:
                    # Leer solo el bloque de filas necesario
                    window = rasterio.windows.Window(
                        col_off=0, row_off=row_start,
                        width=ncols, height=n_rows_block
                    )
                    arr = src.read(1, window=window)
                    
                    # Máscaras booleanas
                    valid_mask = (arr != nodata_in)
                    flood_mask = (arr == 1)
                    nodata_mask = (arr == nodata_in)
                    
                    # Acumular conteos simples
                    count_valid += valid_mask.astype(np.int32)
                    count_flooded += flood_mask.astype(np.int32)
                    
                    # =================================================================
                    # LÓGICA DE CONSECUTIVOS VECTORIZADA (equivalente a tu función)
                    # =================================================================
                    
                    # CASO 1: Inundado hoy (val == 1)
                    # current_streak += 1, max_streak update, reset nodata_gap
                    is_flood = flood_mask
                    current_streak = np.where(is_flood, current_streak + 1, current_streak)
                    max_streak = np.maximum(max_streak, current_streak)
                    nodata_gap_count = np.where(is_flood, 0, nodata_gap_count)
                    
                    # CASO 2: NoData hoy (val == nodata)
                    # incrementar gap, si excede MAX_NODATA_GAP resetear streak
                    is_nodata = nodata_mask & ~is_flood
                    nodata_gap_count = np.where(is_nodata, nodata_gap_count + 1, nodata_gap_count)
                    gap_exceeded = nodata_gap_count > MAX_NODATA_GAP
                    current_streak = np.where(gap_exceeded, 0, current_streak)
                    nodata_gap_count = np.where(gap_exceeded, 0, nodata_gap_count)
                    
                    # CASO 3: Válido pero no inundado (otros valores, tipicamente 0)
                    # resetear streak y gap
                    is_clear = valid_mask & ~is_flood & ~is_nodata
                    current_streak = np.where(is_clear, 0, current_streak)
                    nodata_gap_count = np.where(is_clear, 0, nodata_gap_count)
            
            # =================================================================
            # POST-PROCESAMIENTO DEL BLOQUE (igual que tu código original)
            # =================================================================
            
            # Manejar nodata donde no hay observaciones válidas
            max_consecutive = max_streak.copy()
            no_valid = (count_valid == 0)
            max_consecutive[no_valid] = nodata_in
            
            # Guardar en arrays completos
            count_flooded_full[row_start:row_end, :] = count_flooded
            count_valid_full[row_start:row_end, :] = count_valid
            max_consecutive_full[row_start:row_end, :] = max_consecutive
        
        # =================================================================
        # CÁLCULO DE MÉTRICAS ANUALES (igual que tu código original)
        # =================================================================
        
        flood_frequency = np.where(count_valid_full > 0, 
                                   count_flooded_full / count_valid_full, 
                                   nodata_in)
        flood_duration = np.where(count_valid_full > 0, 
                                  count_flooded_full, 
                                  nodata_in)
        
        # Guardar raster de 4 bandas
        profile.update(
            dtype="float32",
            count=4,
            nodata=nodata_in,
            compress="deflate"
        )
        
        out_file = output_dir / f"floods_annual_{year}.tif"
        
        with rasterio.open(out_file, "w", **profile) as dst:
            dst.write(flood_frequency.astype("float32"), 1)
            dst.write(flood_duration.astype("float32"), 2)
            dst.write(count_valid_full.astype("float32"), 3)
            dst.write(max_consecutive_full.astype("float32"), 4)
            
            dst.set_band_description(1, "Flood frequency (probability)")
            dst.set_band_description(2, "Flood duration (days per year)")
            dst.set_band_description(3, "Valid flood observations")
            dst.set_band_description(4, "Maximum consecutive flooded days")
            
            dst.update_tags(
                PRODUCT="Annual flood hazard metrics (from daily new)",
                DESCRIPTION="Annual flood frequency and duration derived from binary maps",
                FLOOD_FREQUENCY="Probability of pixel being inundated on a given day",
                FLOOD_DURATION="Total number of inundated days per year",
                FLOOD_VALID_OBSERVATIONS="Number of valid flood observations in the year",
                FLOOD_MAX_DURATION="Maximum consecutive flooded days",
                OBSERVED_DAYS=str(n_days),
                TEMPORAL_RESOLUTION="Daily, from binary flood diagnostics",
                PERIOD=str(year),
                REGION="South Sudan",
                SOURCE="Manual aggregation of daily flood diagnostics",
                PROCESSING="Annual aggregation from binary maps (OPTIMIZED)"
            )
        
        print(f"✔ Generado: {out_file}")
        
        # Liberar memoria antes del siguiente año
        del count_flooded_full, count_valid_full, max_consecutive_full
        del flood_frequency, flood_duration
    
    

elif args.map_type == "seasonal_2":
    print("\nGenerating SEASONAL flood hazard maps (manual check)...")

    output_dir = Path("/mnt/staas/CLICHE/00_DATA/hazard_maps/seasonal_check/")
    output_dir.mkdir(exist_ok=True)
    MAX_NODATA_GAP = 5  # days
    # Group files by (season, year)
    print("\nGenerating SEASONAL (by hand) flood hazard maps...")
    groups = defaultdict(list)

    for f in files:
        date = extract_date_from_filename(f.name)
        season, season_year = get_season(date.year, date.month)
        groups[(season_year, season)].append(f)
        
    for (year, season), flist in tqdm(groups.items(), desc="Processing seasons"):

        print(f"\n→ {season.upper()} {year} | {len(flist)} daily maps")

        arrays = []

        # ----------------------------
        # Read all daily maps
        # ----------------------------
        for f in flist:
            with rasterio.open(f) as src:
                arr = src.read(1)
                arrays.append(arr)
                NODATA = src.nodata

        stack = np.stack(arrays, axis=0)  # (time, rows, cols)
        n_time, nrows, ncols = stack.shape

        # ----------------------------
        # Accumulators
        # ----------------------------
        count_flooded = np.zeros((nrows, ncols), dtype=np.int32)
        count_valid = np.zeros((nrows, ncols), dtype=np.int32)
        max_consecutive = np.zeros((nrows, ncols), dtype=np.int32)

        # ----------------------------
        # Loop pixel-wise (manual, explicit)
        # ----------------------------
        for i in tqdm(range(nrows), desc="Rows", leave=False):
            for j in range(ncols):

                ts = stack[:, i, j]

                valid_mask = (ts != NODATA)
                flood_mask = (ts == 1)

                count_valid[i, j] = np.sum(valid_mask)
                count_flooded[i, j] = np.sum(flood_mask)

                if count_valid[i, j] == 0:
                    max_consecutive[i, j] = 0
                else:
                    max_consecutive[i, j] = max_consecutive_flood_with_nodata_tolerance(
                        ts,
                        nodata=NODATA,
                        max_nodata_gap=MAX_NODATA_GAP
                    )

        # ----------------------------
        # Save raster (3 bands)
        # ----------------------------
        with rasterio.open(flist[0]) as src_ref:
            profile = src_ref.profile

        profile.update(
            count=3,
            dtype="int32",
            nodata=0,
            compress="deflate"
        )

        out_name = f"flood_manual_{season}_{year}.tif"
        out_file = output_dir / out_name

        with rasterio.open(out_file, "w", **profile) as dst:
            dst.write(count_flooded, 1)
            dst.write(count_valid, 2)
            dst.write(max_consecutive, 3)

            dst.set_band_description(1, "count_flooded_days")
            dst.set_band_description(2, "count_valid_observations")
            dst.set_band_description(3, "max_consecutive_flood_days")

            dst.update_tags(
                season=season,
                season_year=year,
                description="Manual seasonal flood diagnostics",
                band_1="Number of days with flood detected",
                band_2="Number of valid EO observations",
                band_3="Maximum consecutive flooded days (nodata gap <= 5)"
            )

        print(f"Saved: {out_name}")