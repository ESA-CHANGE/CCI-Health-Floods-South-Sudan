"""Batch paralell processing of SWOT frames to export per-frame raster flood descriptors as GeoJSON.

This script reads SWOT frame objects from ``indir``, computes geoid-based raster
statistics with uncertainty using multiple worker processes, and writes one
GeoJSON file per input object into ``outdir``.

The script does not read the arguments, so you must modify them first by editing the file.
In line 42 can be modified the number of processes that will be created during paralell computing
Failures are logged and skipped so the batch can continue.

"""

__author__ = "Miguel González Jiménez"
__maintainer__ = "Miguel González Jiménez"
__email__ = "mgonzalez.j@gmv.com"

from multiprocessing import Pool
from tqdm import tqdm
from pathlib import Path
import gc
import SWOT as swot

import warnings
warnings.filterwarnings('ignore')

path_series = r"../data/HF_VIIRS_non-Safe_series.parquet"
path_hfs    = r"..data/HF_non-Safe_metadata.geojson"
swot.VIIRS.set_paths(path_series=path_series, path_hfs=path_hfs)

indir = Path(r"path_to_folder_where_SWOT_frames_are_stored")
outdir = Path("path_to_output_folder")

def batch_stats_wrapper(obj):
    try:
        gdf = swot.getBatchStats(
                                obj,
                                0,
                                1,
                                folder=None,
                                returns='stats',
                                stats_datarray='geoid',
                                _clean=True,  # Delete some RAM during the processing
                                _uncert=True) # Calculate uncertainty associated with results.

        return obj.path.stem, gdf

    except Exception as e:
        print(f"Failed for {obj.path.stem}: {e}")
        return None


if __name__ == "__main__":

    objects = swot.Reader(indir, remove_duplicates=True,
                                 clipToHF=True,
                                 already_done=outdir).df['object'].tolist()

    with Pool(processes=12, maxtasksperchild=10) as pool:

        for result in tqdm(pool.imap_unordered(batch_stats_wrapper,objects, chunksize=1),
                           total=len(objects)):

            if result is None:
                continue

            name, gdf = result

            gdf.to_file(outdir / f"{name}.geojson", driver="GeoJSON")

            del result
            gc.collect()