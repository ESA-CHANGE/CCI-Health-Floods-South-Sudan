# -*- coding: utf-8 -*-


__author__ = "Macarena Merida Floriano, mmef"
__maintainer__ = "Macarena Merida Floriano, mmef"
__email__ = "mmef@gmv.com"

"""
BayFloodGEN data utilities.

This script contains functions for loading, parsing, and expanding the 
input data for the BayFloodGEN model.
"""

import json
from datetime import datetime

import numpy as np
import pandas as pd


# ==========================================================
# 1. DATA EXPANSION — vectorized
# ==========================================================
def expand_series_to_df(df: pd.DataFrame) -> pd.DataFrame:
    r"""Expands the JSON time series data from the original dataframe into
    a long daily format. Each row in the output dataframe corresponds to a
    single facility-day, with columns for metadata, date, occurrence,
    percentage flooded, and distance to water.

    The function handles the following transformations:
    - Explodes the 'dates' column to create a row for each date.
    - Aligns the 'series_binary', 'series_pct_flood_pixels', and 'series_distance_m'
        columns with the exploded dates.
    - Extracts day of year and year from the date for seasonality and trend
        modeling.
    - Imputes distance to water for dry days based on buffer size.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with columns:
        - 'hf_id', 'hf_payam', 'hf_category', 'latitude', 'longitude',
          'buffer_pixels', 'dates', 'series_binary', 'series_pct_flood_pixels',
          'series_distance_m'

    Returns
    -------
    df_out : pd.DataFrame
        Expanded dataframe with columns:
        - 'hf_id', 'hf_payam', 'hf_category', 'latitude', 'longitude',
          'buffer_pixels', 'date', 'occurrence', 'pct_flooded', 'min_distance',
          'day_of_year', 'year'

    """

    # Row lengths to create repeating index for metadata
    lengths = df['dates'].apply(len).values          # (n_facilities,)
    # Repeat facility metadata for each day
    rep_idx = np.repeat(np.arange(len(df)), lengths)

    # Expand lists to flat arrays at once
    dates_flat = np.concatenate(df['dates'].values)
    occ_flat = np.concatenate(df['series_binary'].values).astype(np.float32)
    pct_flat = np.concatenate(df['series_pct_flood_pixels'].values).astype(np.float32) / 100.0
    dist_flat = np.concatenate(df['series_distance_m'].values).astype(np.float32)

    buf_flat = df['buffer_pixels'].values[rep_idx]
    doy_flat = np.array([d.timetuple().tm_yday for d in dates_flat], dtype=np.int16)
    year_flat = np.array([d.year for d in dates_flat], dtype=np.int16)

    df_out = pd.DataFrame({
        'hf_id': df['hf_id'].values[rep_idx],
        'hf_payam': df['hf_payam'].values[rep_idx],
        'hf_category': df['cat_new'].values[rep_idx],
        'latitude': df['latitude'].values[rep_idx],
        'longitude': df['longitude'].values[rep_idx],
        'buffer_pixels': buf_flat,
        'date': dates_flat,
        'occurrence': occ_flat,
        'pct_flooded': pct_flat,
        'min_distance': dist_flat,
        'day_of_year': doy_flat,
        'year': year_flat,
    })

    # Impute distance for dry days based on buffer size
    dry = df_out['occurrence'] == 0
    df_out.loc[dry & (df_out['buffer_pixels'] == 40), 'min_distance'] = 2545.5
    df_out.loc[dry & (df_out['buffer_pixels'] == 60), 'min_distance'] = 3818.0

    return df_out


def parse_series_columns(df: pd.DataFrame) -> pd.DataFrame:
    r"""Parse list-like series columns from JSON strings when needed.

    This preserves the same parsing logic used in the original script:
    - If `series_binary` is string, parse all three series columns and `dates`.
    - If already list, only parse `dates` when still string-encoded.
    """

    sample = df['series_binary'].iloc[0]

    if isinstance(sample, str):
        # strings → parse
        for col in ['series_binary', 'series_pct_flood_pixels', 'series_distance_m']:
            df[col] = df[col].apply(json.loads)
        df['dates'] = df['dates'].apply(
            lambda x: [datetime.strptime(d, '%Y-%m-%d') for d in json.loads(x)]
        )
    elif isinstance(sample, list):
        # Already lists → only parse dates if they are strings
        if isinstance(df['dates'].iloc[0], str):
            df['dates'] = df['dates'].apply(
                lambda x: [datetime.strptime(d, '%Y-%m-%d') for d in json.loads(x)]
            )

    return df
