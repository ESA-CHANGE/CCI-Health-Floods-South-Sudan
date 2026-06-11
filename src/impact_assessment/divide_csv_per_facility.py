# -*- coding: utf-8 -*-

__author__ = "Macarena Merida Floriano, mmef"
__maintainer__ = "Macarena Merida Floriano, mmef"
__email__ = "mmef@gmv.com"

"""
BayFloodGEN Generator — data preprocessing script.
=====================================================
This script reads the complete CSV of flood scenarios, fills in missing categories,
and partitions the data into separate Parquet files per health facility for more efficient
assessment. Each output file corresponds to one facility and contains all its daily records.
"""

import pandas as pd
import os

CSV_PATH   = "../data/model_output/bayfloodgen_output.csv"
PARTS_DIR  = "../data/model_output/by_facility"
os.makedirs(PARTS_DIR, exist_ok=True)

print("Reading complete CSV...")
df = pd.read_csv(
    CSV_PATH,
    parse_dates=["date"],
    dtype={"hf_category":"string","scenario_id":"Int32","hf_id":"string"},
    low_memory=False
)
print(f"  {len(df):,} rows loaded")

# Fill in hf_category
cat_map = df.dropna(subset=["hf_category"]).groupby("hf_id")["hf_category"].first()
df["hf_category"] = df["hf_category"].fillna(df["hf_id"].map(cat_map))
df["_year"] = df["date"].dt.year

print("Partitioning by facility...")
for hf_id, grp in df.groupby("hf_id", sort=False):
    safe_id = hf_id.replace("/","_").replace(" ","_")
    path = os.path.join(PARTS_DIR, f"{safe_id}.parquet")
    grp.to_parquet(path, index=False)

print(f"  Saved {df['hf_id'].nunique()} partitions in {PARTS_DIR}")