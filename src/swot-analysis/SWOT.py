r"""Utilities for SWOT flood-depth processing and health-facility analytics.

This module provides an end-to-end workflow to:
1. Read and preprocess SWOT L2 HR Raster frames.
2. Apply quality-control filters and optional DEM-based depth conversion.
3. Build per-facility flood descriptors over spatial patches.
4. Compare SWOT descriptors with VIIRS facility flood time series.
5. Aggregate, visualize, and export results.

Main components/classes:
------------------------
- ``Reader``: batch frame discovery and loading.
- ``SWOT``: per-frame filtering, reprojection, depth estimation, and export.
- ``BitWise``: inspection of ``wse_qual_bitwise`` quality categories.
- ``FloodStats`` and ``Patch``: facility patch generation and descriptor extraction.
- ``VIIRS``: access to VIIRS facility metadata and time series.
- ``Results``: post-processing and plotting utilities for output GeoJSON datasets.

Standalone helpers include frame reading, bounding-box utilities, batch statistics
execution, and memory cleanup functions.
"""

__author__ = "Miguel González Jiménez"
__maintainer__ = "Miguel González Jiménez"
__email__ = "mgonzalez.j@gmv.com"

import xarray as xr
import rioxarray as riox
from pathlib import Path
import pandas as pd
import geopandas as gpd
import numpy as np

import matplotlib.pyplot as plt

import re
import os
import logging
from tqdm.auto import tqdm

from scipy.spatial import cKDTree


def align(da, dem):
    r"""Align dem to da and clip output to da.
    Parameters
    ----------
    da: xr.DataArray
    dem: xr.DataArray
    
    Returns
    -------
    da: xr.DataArray
    """

    dem_r = dem.rio.reproject(da.rio.crs)
    return dem_r.interp(x=da.x, y=da.y, method="nearest").rio.write_crs(da.rio.crs)

def drop_wrong_frames(metadata):
    r"""Retain only frames in valid AOI, not in Antartic or Artic.
    
    Parameters
    ----------
    metadata: dict
    
    Returns
    -------
    bool
    """

    zone = int(metadata['utm_zone_num'])
    if zone == 60 or zone==1:
        return False
    else:
        return True
    
def get_url(umm):
    r"""Extract the direct download URL from UMM metadata.

    Parameters
    ----------
    umm: dict
        UMM metadata dictionary containing the ``RelatedUrls`` field.

    Returns
    -------
    str
        First URL whose ``Type`` is ``GET DATA``.
    """

    urls_list = umm['RelatedUrls']
    return [i['URL'] for i in urls_list if i['Type']=='GET DATA'][0]

def get_file(url, folder=None):
    r"""Download a remote file from URL into a local folder.

    Parameters
    ----------
    url: str
        Remote file URL.
    folder: str | pathlib.Path | None
        Destination directory where the downloaded file will be saved.
    """

    import requests

    folder = Path(folder)
    filename = folder / Path(url).name
    
    try:
        response = requests.get(url, stream=True, verify=False)

        response.raise_for_status()  # Raises an HTTPError for bad responses
        
        with open(filename, 'wb') as file:
            file.write(response.content)
        print(f'Downloaded url: {url}')

    except Exception as e:
        print(f"Error when downloading url: {url}")

def get_bbox(da):
    r"""Get bounding box as GeoDatafram from xr.DataArray
    
    Parameteres
    -----------
    da: xr.DataArray
    
    Returns
    -------
    gdf: gpd.GeoDataFrame
    """

    from shapely.geometry import Polygon

    # Extract corners
    lat_min = float(da['y'].min())
    lat_max = float(da['y'].max())
    lon_min = float(da['x'].min())
    lon_max = float(da['x'].max())

    # Define corners in (lon, lat) order
    poly = Polygon([
        (lon_min, lat_min),
        (lon_min, lat_max),
        (lon_max, lat_max),
        (lon_max, lat_min),
        (lon_min, lat_min)])

    return gpd.GeoDataFrame(index=[0], geometry=[poly], crs=da.rio.crs)

def get_swot_bbox_format(da):
    r"""Return SWOT-style bounding box tuple from a DataArray.

    Parameters
    ----------
    da: xr.DataArray

    Returns
    -------
    tuple
        Bounding box in the format ``(xmin, ymin, xmax, ymax)``.
    """
    coords = da.get_coordinates()

    return (coords['x'].min().item(), coords['y'].min().item(),
            coords['x'].max().item(), coords['y'].max().item())

class Reader:
    r"""Class that allows to read and format SWOT frames on an
    easy and structured way."""

    def __init__(self, folder,
                 already_done=None,
                 remove_duplicates=True,
                 clipToHF=True,
                 _limit=None):
        r"""Read SWOT NetCDF files and build a frame index DataFrame.

        Parameters
        ----------
        folder: pathlib.Path
            Folder containing SWOT ``.nc`` files.
        already_done: pathlib.Path | None
            Folder with already processed outputs to skip.
        _limit: int | None
            Optional limit for number of files to read.
        remove_duplicates: bool
            Whether to retain only best duplicate versions.
        """
 
        self.df = self._read(folder, already_done=already_done,
                                     _limit=_limit,
                                     remove_duplicates=remove_duplicates)
        
        if clipToHF:
            self.df = self.clipToHF(self.df)

    def _read(self, folder, already_done=None, _limit=None, remove_duplicates=False):
        if _limit is not None:
            files = list(folder.glob('*.nc'))[0:_limit]
        else:
            files = list(folder.glob('*.nc'))

        if already_done is not None:
            done_files = [i.stem for i in list(already_done.rglob("*.geojson"))]
            files = [i for i in files if not i.stem in done_files]

        df = pd.DataFrame(dict(path=files))

        df['resolution'] = df['path'].apply(lambda x: x.name.split('_')[4])
        df['datetime'] = df['path'].apply(lambda x: x.name.split('_')[-4])

        df['datetime'] = pd.to_datetime(df['datetime'])
        df['date'] = df['datetime'].apply(lambda x: x.date())
        df['year'] = df['datetime'].dt.year
        df['month'] = df['datetime'].dt.month

        df = df.set_index('datetime')

        if remove_duplicates:
            df = self._removeDuplicate(df)

        tqdm.pandas(desc=f"Reading SWOT frames")
        df['object'] = df['path'].progress_apply(read_frame)

        logging.info("All frames have been read and converted into SWOT objects.")

        return df

    def _removeDuplicate(self, df):

        df['_CRID']          = df['path'].apply(lambda x: Path(x).stem.split('_')[-2])
        df['_Version']       = df['path'].apply(lambda x: Path(x).stem.split('_')[-1]).astype(int)
        df['_FullName']      = df['path'].apply(lambda x: '_'.join(Path(x).stem.split('_')[0:-2]))
        df['_crid_priority'] = df['_CRID'].map(dict(PGD0=1, PID0=0))

        def choose_best(group):
            return group.sort_values(['_crid_priority', '_Version'], ascending=False).iloc[0]

        df_clean = df.groupby('_FullName', group_keys=False).apply(choose_best).reset_index(drop=True)
        df_clean.drop(['_CRID', '_Version', '_FullName', '_crid_priority'], axis=1, inplace=True)

        logging.info("Duplicated versiones removed from SWOT frames DataFrame")

        return df_clean
    
    def clipToHF(self, df):
        r"""Keep frames whose footprint contains at least one health facility.

        Parameters
        ----------
        df: pandas.DataFrame
            DataFrame with an ``object`` column containing SWOT objects.

        Returns
        -------
        pandas.DataFrame
            Filtered DataFrame with only frames containing health facilities.
        """
        hfs = VIIRS().gdf
        tqdm.pandas(desc='Clipping frames to those containig HFs')
        df['geometry'] = df['object'].apply(lambda x: x.footprint.to_crs(4326).geometry.iloc[0])
        hf_points = hfs.geometry
        df['Contains_HF'] = df['geometry'].progress_apply(lambda geom: hf_points.within(geom).any())
        print(df['Contains_HF'].value_counts())
        return df[df['Contains_HF']]

class SWOT:
    r"""This class modelize a SWOT-mission frame of the product L2_HR_Raster.
    Includes several method for quality filtering and visualing."""

    def __init__(self, path, crs2project=None, dem_path=None,
                 dem_type=None, dem_crs=4326):
        r"""Class for handling SWOT frames.
        
        Parameters
        ----------
        path: pathlib.Path
        crs2project: int | None. Crs to reproject the file to. If None, UTMZone of original file is kept.
        dem_path: pathlib.Path | None
            Path to DEM covering SWOT footprint to be used when converting WSE into ground-referenced
            values. DEM should be referenced to geoid EGM2008 to keep consistency with WSE values.
        dem_type: str | None
            Name of the type of DEM used to be included in output filename.
        dem_crs: int
            Crs of DEM.
        """
        self._done = False
        self._reprojected = False
        self._dem_type = dem_type
        self.path = path

        dem_type = dem_type if dem_type is not None else ''

        self.filename = path.stem
        self.name = '_'.join([self.filename.split('_')[0], *self.filename.split('_')[-8:-3],
                              f"using-{dem_type}"])

        # Removing time-vars to avoid projecting errors
        self.raw = xr.open_dataset(self.path).drop_vars(['illumination_time', 'illumination_time_tai'])

        self.metadata = self.raw.attrs.copy()
        self.da = self.raw.copy()

        if dem_path is not None:
            self._dem =  riox.open_rasterio(dem_path).sel(band=1).rio.write_crs(dem_crs)
        else:
            self._dem = None
        
        self._assign_proj(crs2project=crs2project)
    
    def reset(self, crs2project=None):
        r"""Revert all applied filters.
        
        Parameters
        ----------
        crs2project: optional"""
        self.da = self.raw
        self._assign_proj(crs2project=crs2project)
        self.name = '_'.join([*self.filename.split('_')[:5], *self.filename.split('_')[-7:-3],
                              f"using-{self._dem_type}"])

    def remove(self):
        r"""Release large in-memory data attributes from the SWOT object.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        del self.da
        self.raw.close()
        if hasattr(self, '_depth'):
            del self._depth
        del self._dem

    @property
    def shortname(self):
        pieces = self.filename.split('_')
        return '_'.join([*pieces[0:6], pieces[-4][0:8]])

    def _assign_proj(self, crs2project=None):
        r"""Asign projection obtained from UTMzone, extract corresponding EPSG and assign it.
        Optionally, reproject to crs2project if specify.
        """

        from pyproj import CRS

        hemisphere = dict(North=['N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X'],
                          South=['C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M'])
        
        is_south = self.metadata['mgrs_latitude_band'] in hemisphere['South']

        crs_current = CRS.from_dict({'proj': 'utm', 'zone': self.metadata['utm_zone_num'],
                             'south': is_south}).to_authority()
        
        if crs2project is not None:
            self.da = self.da.rio.write_crs(crs_current).rio.reproject(crs2project)
            self._crs = self.da.rio.crs
        else:
            self.da = self.da.rio.write_crs(crs_current)
            self._crs = crs_current

        self._reprojected = True
        self.name= '_'.join([self.name, self.da.rio.crs.to_string()])

    def project(self):
        r"""Write current CRS to the active DataArray and return self.

        Parameters
        ----------
        None

        Returns
        -------
        SWOT
        """
        assert self._reprojected, f"First, reproject to desired crs using reproject()"
        self.da = self.da.rio.write_crs(self._crs)
        return self

    @property
    def footprint(self):
        from shapely.geometry import Polygon

        attrs = self.raw.attrs

        poly = Polygon([ (attrs['left_first_longitude'], attrs['left_first_latitude']),
                        (attrs['left_last_longitude'], attrs['left_last_latitude']),
                        (attrs['right_last_longitude'], attrs['right_last_latitude']),
                        (attrs['right_first_longitude'], attrs['right_first_latitude']),
                        (attrs['left_first_longitude'], attrs['left_first_latitude']),
                       ])

        metadata = {key: [value] for key, value in zip(['name', 'path', 'shortname'], [self.name, str(self.path), self.shortname])}
        metadata["geometry"] = [poly]
        return gpd.GeoDataFrame(metadata, crs=4326).to_crs(self._crs)

    @property
    def bbox(self):
        r"""Get gdf of SWOT footprint."""

        assert self._reprojected, f"da needs to be reprojected first, use reproject()"

        from shapely.geometry import Polygon

        # Extract corners
        lat_min = float(self.da['y'].min())
        lat_max = float(self.da['y'].max())
        lon_min = float(self.da['x'].min())
        lon_max = float(self.da['x'].max())

        # Define corners in (lon, lat) order
        poly = Polygon([
                        (lon_min, lat_min),
                        (lon_min, lat_max),
                        (lon_max, lat_max),
                        (lon_max, lat_min),
                        (lon_min, lat_min)])

        return gpd.GeoDataFrame(index=[0], geometry=[poly], crs=self.da.rio.crs)

    @property
    def dem(self):
        return self._dem
    
    @dem.setter
    def dem(self, dem_path):
        self._dem =  riox.open_rasterio(dem_path).sel(band=1)

    def filtrate_qual(self, *args):
        r"""Filters SWOT frame according to wse_qual field and return self.

        Parameters
        ----------
        args: int.
            Numbers of quality you want to retain. From 0 (Good) to 3 (Bad).

        Example: filtrate_qual(swot, 0, 1) is equivalent to (swot['wse_qual'] == 0) | (swot['wse_qual'] == 1)

        Returns
        ------
        SWOT 
        """
        mask = self.da['wse_qual'].isin(args)
        self.da = self.da.where(mask)

        # Including in name:
        _str = ''.join([str(i) for i in args])
        self.name = '_'.join([self.name, f"QUAL{_str}"])

        return self.project()
    
    def filtrate_uncert(self, threshold):
        r"""Filters SWOT frame according to wse_uncert field and returns self.

        Parameters
        ----------
        threshold: float
            Pixels with values above this threshold will be removed.

        Returns
        -------
        SWOT
        """

        mask = self.da['wse_uncert'] < threshold
        self.da = self.da.where(mask)

        self.name = '_'.join([self.name, f"UNCER{threshold}"])

        return self.project()

    def filtrate_cross(self, min_val, max_val):
        r"""Filters SWOT frame based on cross_track field and return self.
        Retain pixels with a cross-track distance between min and max, for both sides of the
        swath (hence, absolute values)

        Parameters
        ----------
        min_val: int
            Absolute min value (in meters) of cross-track distance.
        max_val: int
            Absolute max value (in meters) of cross-track distance.
        
        Returns
        -------
        SWOT

        E.g., filtrate_cross(10,60000) will retain from -60000 to -10000 and from
        10000 to 60000
        """

        cross = self.da['cross_track']

        mask = (np.abs(cross) >= min_val) & (np.abs(cross) <= max_val)

        self.da = self.da.where(mask)

        self.name = '_'.join([self.name, f"CROSS{min_val}-{max_val}"])

        return self
    
    def filtrate_water(self, perc):
        r"""Filters SWOT frame according to water_frac field.

        Parameters
        ----------
        perc: float
            Values bellow this threshold will be removed.
        
        Returns
        -------
        SWOT
        """
        mask = self.da['water_frac'] > perc
        self.da = self.da.where(mask)

        self.name = '_'.join([self.name, f"FRAC{perc}"])

        return self.project()
    
    def filtrate_dark(self, perc):
        r"""Filters SWOT frame according to dark_frac field and returns self.

        Parameters
        ----------
        perc: float
            Pixels with a dark water fraction above this threshold will be removed"""
        mask = self.da['dark_frac'] < perc
        self.da = self.da.where(mask)

        self.name = '_'.join([self.name, f"DARK{perc}"])

        return self.project()
    
    def filtrate_layover(self, percentil):
        r"""Filters SWOT frame according to layover_impact and return self.
        
        Parameters
        ----------
        percentil: float.
            Pixels with a layover impact abs value above this percentil will be removed.
            
        Returns
        -------
        SWOT
        """

        da_lay = self.da['layover_impact']
        thr = np.nanpercentile(np.abs(da_lay), percentil)
        mask = np.abs(da_lay) < thr
        self.da = self.da.where(mask)
        
        self.name = '_'.join([self.name, f"LAYOVER{percentil}"])

        return self.project()

    def filtr_plot(self):
        r"""Plot comparing filtered vs non-filtered WSE datarray
        
        Parameteres
        ----------
        None

        Returns
        -------
        matplotlib.pyplot.axes
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 2, figsize=(15,6))

        self.raw.wse.plot(ax=ax[0])
        self.da.wse.plot(ax=ax[1])

        ax[0].set_title('Not-Filtered')
        ax[1].set_title('Filtered')
        return ax
    
    @property
    def inspect(self):
        r"""Returns a BitWise object to inspect the dataframe of bits categories.
        You can acces to this dataframe using self.inspect.lookup property.
        Inspect unique pixels categories (columns) affected by specific issues (rows).
        Make a list with those categories that flags bad quality pixels
        and pass it as args in filtr_bitwise to remove those pixels from dataset.
        
        Follow this table to assess flags (categories) gravity:
        
        Bad --> remove allways.
        Degraded --> remove almost allways.
        Suspect --> Removal depending on application.
        """
        return BitWise(self)
    
    def filtr_bitwise(self, *bits_cats):
        r"""Remove pixels of specific bit category(ies). These categories are indicated by
        wse_qual_bitwise--a variable that indicates the combination of quality categories in a bitwise
        manner. This bits_categories needs to be first identified using ``BitWise`` class, accessed
        throughout ``inspect`` property.
        
        For instance, pixels affected by 'few_pixels' variables usually removes erroneous pixels of nadir area,
        which helps reducing image noise. 'specular_ringing_prior_water_suspect' tends to remove river channels.
        
        Parameters
        ----------
        bits_cats: str
            Name of categories found to be related with bad quality data.
        """

        Inspect = self.inspect

        to_remove = list(bits_cats)

        # selecciona columnas del lookup asociadas a esos flags
        mask_flags = Inspect._lookup.loc[to_remove].any(axis=0)

        # convierte a nombres de columnas activas (bits)
        bits2remove = (Inspect._lookup.columns.to_series().iloc[1:].loc[mask_flags].astype(int).values)

        # === FIX CLAVE: construir máscara bitwise correcta ===
        bitfield = self.da.wse_qual_bitwise.values.astype("uint32")

        combined_mask = 0
        for b in bits2remove:
            combined_mask |= b

        mask_retain = (bitfield & combined_mask) == 0

        # Aplicar máscara
        self.da['wse'] = self.da.wse.where(mask_retain)

        self.name = '_'.join([self.name, "BWISE"])

        return self
    

    def toDepth(self, filtered=True, lt=None, gt=None,
                quantile_fr=False, quantile_fr_params=(0.02, 0.98)):
        r""" Convert WSE values, which are by default referenced to geoid EGM2008 to
        ground-reference values according to the DEM passed when class initialization
        or dem_path property setting.

        Parameters
        ----------
        lt: int. Optional.
            Take only those depth values lower than given value.
        gt: int. Optional.
            Take only those depth values upper than given value.
        quantile_fr: bool.
            Wether to apply a quantile-based filter of depth values based.
        quantile_fr_params: tuple. Optional.
            Tuple of quantile thresholds for quantile-based filtering. Only works if quantile_fr is True.
        
        Returns
        -------
        SWOT
        """

        assert self.dem is not None, "Associate first a dem using dem attr."
        assert isinstance(quantile_fr, bool), f"quantile_fr should be bool. Type: {type(quantile_fr)}"

        da = self.da if filtered else self.raw

        dem_aligned = align(da.wse, self.dem)

        self._depth = da.wse.copy() - dem_aligned

        if lt is not None:
            mask = self._depth < lt
            self._depth = self._depth.where(mask)
            if not self._done:
                self.name = '_'.join([self.name, f"lt{lt}"])
        
        if gt is not None:
            mask = self._depth > gt
            self._depth = self._depth.where(mask)

            if not self._done:
                self.name = '_'.join([self.name, f"gt{gt}"])
        
        if quantile_fr:
            q_low, q_high = quantile_fr_params[0], quantile_fr_params[1]
            mask = (self.depth >= q_low) & (self.depth < q_high)
            self._depth = self._depth.where(mask)
            print(f"Pixels existing above quantil {q_high}: {(self.depth > q_high).sum().item()} out of {self.depth.count().item()}")
            print(f"Pixels existing bellow quantil {q_low}: {(self.depth < q_low).sum().item()} out of {self.depth.count().item()}")

        self._done = True

        return self
    
    @property
    def depth(self):
        assert hasattr(self,'_depth'), "Should calculate first FloodDepth using toDepth"
        return self._depth
 
    def plotDepth(self):
        r"""Plot depth raster with fixed color limits.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.depth.plot(vmin=0, vmax=4, cmap='Spectral_r')
    
    def depthHist(self, **kwargs):
        r"""Plot histogram of flood depth map values
        
        Parameters
        ----------
        None
        Kwargs are passed to  pandas.DataFrame.hist() method.
        
        Returns
        -------
        df: pd.DataFrame
        """

        return pd.DataFrame(self.depth.values).melt().value.dropna().hist(**kwargs)
    
    @property
    def depth_values(self):
        return pd.DataFrame(self.depth.values).melt().value.dropna()
    
    @property
    def stats(self):
        return FloodStats(self)
    
    def save(self, folder, stats_datarray='geoid', extra=None):
        r"""Save geoid or ground raster output to GeoTIFF.

        Parameters
        ----------
        folder: pathlib.Path
            Destination output folder.
        stats_datarray: str
            Either ``'geoid'`` or ``'ground'``.
        extra: str | None
            Optional suffix appended to the output filename.

        Returns
        -------
        None
        """

        extra = f"_{extra}" if extra is not None else ''

        if stats_datarray=='ground':
            hasattr(self, 'depth'), "Should calculate first FloodDepth using toDepth"
            self.depth.rio.to_raster(folder/f"{self.name}_FloodDepth{extra}.tif", compress='LZW')
        
        elif stats_datarray=='geoid':
            self.da.wse.rio.to_raster(folder/f"{self.name}_FloodDepth{extra}.tif", compress='LZW')
        else:
            raise NameError(f"stats_datarray is not 'ground' or 'geoid")

    def _check_inspect(self, cats):
        r"""Validate requested bitwise categories against lookup table.

        Parameters
        ----------
        cats: list[str]
            Requested bitwise category names.

        Returns
        -------
        None
        """
        args_series = pd.Series(cats)
        mask = args_series.isin(self.inspect._lookup.index)

        assert mask.all(), f"Following categories do not exist: {args_series[~mask].values.tolist()}"

class BitWise:
    r"""This class allows to inspect the field wse_qual_bitwise of a raster
    SWOT dataset. This field is a bit flag that provides details on why the
    wse_qual flag is set as it is.

    The class allows to access the lookup tables and main flag categories.
    
    Inspection of bits categories is needed when aggresive filtrating
    by wse_bitwise dataset variable. Each info category is represented as a bit.
        Each bits aggregation is unique and represents a specific combination
        of categories."""

    def __init__(self, Swot):
        r"""
        Parameters
        ----------
        Swot: swot.SWOT
        """

        self.object = Swot
        self.attrs = attrs = dict(Swot.da.wse_qual_bitwise.attrs.copy())

        self.flags = pd.DataFrame({'flag_meanings': attrs.get("flag_meanings").split(" "), 
                      'flag_masks': attrs.get("flag_masks"), }).set_index("flag_meanings")

    @property
    def unique_bits(self):
        r"""Return unique integer values present in wse_qual_bitwise.
        """
        return pd.DataFrame(self.object.raw.wse_qual_bitwise.values).melt().value.unique().tolist()

    def lookup(self, drop=[]):
        r"""Disaggegrate bits structure creating lookup table of categories per pixel.
        
        Parameters
        ----------
        drop: list,
            List of flag variables to exclude from lookup table.
        """

        def qual_bits_iter(n):
            r"""Yield individual bit masks from an integer flag value.
            """
            n = int(n)

            while n:
                b = n & (~n+1)
                yield b
                n ^= b

        decomposed_qual = {}
        for i in sorted(self.unique_bits):
            if i not in decomposed_qual:
                decomposed_qual[i] = qual_bits_iter(i)

        new_columns_qual = {}
        for i, j in enumerate(list(set(decomposed_qual))):
            new_columns_qual[j] = self.flags['flag_masks'].apply(lambda x: x in [b for b in qual_bits_iter(j)])

        self._raw_lookup = self.flags.join(pd.DataFrame(new_columns_qual).sort_index(axis=1)).copy()

        return self._lookup.drop(drop).style.apply(lambda x: ['background-color:yellow' if all([i,type(i)==bool]) else '' for i in x])
    
    @property
    def _lookup(self):
        r"""Return cached lookup table, building it on first access.
        """
        if not hasattr(self, '_raw_lookup'):
            self.lookup()
        return self._raw_lookup
    
    @property
    def lookup_hist(self):
        r"""Return category occurrence counts from lookup table.
        """
        return self._lookup.drop('flag_masks', axis=1).sum(axis=1)


class FloodStats:
    r"""Class designed to create flood descriptores at the patches of
    every health facility. Creates HF patches and extract descriptors
    on each of them usin 'Patch' class."""

    def __init__(self, Scene):
        r""" Read HF point layer and clip it to SWOT's frame footprint.

        Parameters
        ----------
        Scene: swot.SWOT
        """

        self.Scene = Scene
        self.points = self._read_points()
        self.patches = self._create_patches(self.points)
    
    def _read_points(self):
        r"""Reproject health facility shp to Scene crs and spatial join."""

        points = VIIRS().gdf.to_crs(self.Scene._crs)
            
        points['patch_size'] = points['buffer_pixels'].apply(lambda x: 3600 if x==40 else 5400)

        return gpd.sjoin(points, self.Scene.footprint, predicate='within')

    def _create_patches(self, points):
        r"""Create patches around points according to square buffer indicated by patch_size field."""

        from shapely.geometry import CAP_STYLE
        points_patch = points.copy()
        points_patch['point_geometry'] = points_patch['geometry']
        points_patch['geometry'] = points_patch.apply(lambda x: x['geometry'].buffer(x['patch_size']/2, cap_style=CAP_STYLE.square), axis=1)
        return points_patch.set_crs(points_patch.crs)
    
    def get_patches(self, stats_datarray, **kwargs):
        r"""Create ``Patch`` objects for all facilities in current scene.

        Parameters
        ----------
        stats_datarray: str
            Either ``'geoid'`` or ``'ground'``.
        **kwargs: dict
            Extra arguments forwarded to ``Patch``.

        Returns
        -------
        list[Patch]
        """

        return [Patch(self.Scene, self.patches.iloc[[i]], stats_datarray, **kwargs) for i in range(self.patches.shape[0])]
    
    def get_Stats(self, stats_datarray='ground', **kwargs):
        r"""Compute per-patch flood statistics and concatenate outputs.

        Parameters
        ----------
        stats_datarray: str
            Either ``'geoid'`` or ``'ground'``.
        **kwargs: dict
            Extra arguments forwarded to ``Patch``.

        Returns
        -------
        pandas.DataFrame | geopandas.GeoDataFrame
        """

        patches = self.get_patches(stats_datarray, **kwargs)
        gdfs_rows = [i.stats for i in patches]
        return pd.concat(gdfs_rows)
    
    def saveAll(self, folder, stats_datarray='ground'):
        r"""Save patches, statistics, and footprint for a scene.

        Parameters
        ----------
        folder: str | pathlib.Path
            Base output folder.
        stats_datarray: str
            Either ``'geoid'`` or ``'ground'``.

        Returns
        -------
        None
        """

        import os

        folder = Path(folder) if isinstance(folder,str) else folder
        subfolder = folder/self.Scene.shortname
        os.makedirs(subfolder, exist_ok=True)

        self.patches.drop('point_geometry', axis=1).to_file(subfolder/f"{self.Scene.shortname}_Patches.geojson", driver='GeoJSON')
        self.get_Stats(stats_datarray=stats_datarray).to_file(subfolder/f"{self.Scene.shortname}_Stats.geojson", driver='GeoJSON')
        self.Scene.footprint.to_file(subfolder/f"{self.Scene.shortname}_Footprint.geojson", driver='GeoJSON')

class Patch:
    r"""A class to model the patch created around every HF as a square buffer,
    where SWOT-based flood descriptors will be calculated. Includes all the 
    methods to estimate these statistics.
    """

    def __init__(self, Scene, gdf, stats_datarray, uncert=False, threshold=5):
        r""" Clip SWOT's frame dataset to patch.

        Parameters
        ----------
        Scene: swot.SWOT
        gdf: gpd.GeoDataFrame
            Patch.
        stats_datarray: str, 'geoid' or 'ground'.
            Referes to the datarray considered to calculate statistics.
            When 'geoid', former values referenced respect geoid EGM2008 are taken;
            if 'ground', depth values Scene.depth property are taken.
        uncert: bool.
            Wether to calculate uncertainty propagation through analysis.
        threshold: int.
            Minimum number of flooded pixels within the buffer (HF patch) to
            consider the HF as flooded and hence computing statistics.
        """

        self.Scene = Scene
        self.gdf = gdf
        self.polygon = gdf['geometry']
        self.point   = gdf['point_geometry']
        self.stats_datarray = stats_datarray
        self._uncert = uncert
        self._threshold = threshold

        point_lat = gdf['latitude'].values[0]
        point_lon = gdf['longitude'].values[0]

        msg = f"stats_datarray arg should be 'geoid or 'ground'. Provided: {stats_datarray}"
        assert stats_datarray in ['geoid', 'ground'], msg

        da = Scene.depth if stats_datarray=='ground' else Scene.da.wse
        self.da = da.rio.clip(gdf.geometry, crs=gdf.crs, drop=True)

        # Just for consulting
        self.raw = Scene.da.rio.clip(gdf.geometry, crs=gdf.crs, drop=True)

    def isFlooded(self):
        r"""Check whether flooded-pixel count passes threshold and returns bool.
        """

        n_flooded = self.da.notnull().sum().item()
        if n_flooded >= self._threshold:
            return True
        else:
            return False

    @property
    def wse_values(self):
        return self.da.to_dataframe().wse.dropna()

    @property
    def max_depth(self):
        r"""Estimate max flooded depth using 95th percentile.
        """

        if self.isFlooded():
            return round(self.da.quantile(0.95).item(), 2)
        else:
            return np.nan

    @property
    def mean_depth(self):
        r"""Estimate mean flooded depth for the patch.
        """

        if self.isFlooded():
            return round(self.da.mean().item(),2)
        else:
            return np.nan
    
    @property
    def median_depth(self):
        r"""Estimate median flooded depth for the patch.
        """

        if self.isFlooded():
            return round(self.da.median().item(),2)
        else:
            return np.nan

    @property
    def distance(self):
        r"""Distance from Facility to pixel with maximum flooded value, calculated
        as in maxDepth property (i.e., based on q95)."""

        if self.isFlooded():
            max_depth = self.max_depth
            return self._distance(max_depth)
        else:
            return np.nan

    def _distance(self, point):
        r"""Compute distance from facility point to nearest raster value to target.

        Parameters
        ----------
        point: float
            Target value to search within the patch raster.

        Returns
        -------
        float
        """
        from shapely.geometry import Point
        diff = abs(self.da - point)
        argmin = diff.argmin(dim=("y", "x"))

        x = self.da.isel(x=argmin["x"]).x.item()
        y = self.da.isel(y=argmin["y"]).y.item()

        return self.point.distance(Point(x, y)).item()

    @property
    def flood_frac(self):
        r"""Compute fraction of flooded pixels within the patch.
        """

        non_flood = self.da.isnull().sum().item()
        total = self.da.size
        flood = total - non_flood

        return flood/total

    @property
    def max_Uncert(self):
        r"""Estimate uncertainty of max depth by bootstrap.
        """

        from scipy.stats import bootstrap

        if self.isFlooded():
            func = lambda x: np.quantile(x, 0.95)
            B = bootstrap((self.wse_values.values,), func, n_resamples=10000, random_state=42)
            return B.standard_error.item()
        else:
            return np.nan

    @property
    def mean_Uncert(self):
        r"""Propagate uncertainty for mean depth estimate.
        """

        if self.isFlooded():
            values = self.raw.wse_uncert.to_dataframe().wse_uncert.dropna()
            return np.sqrt((values**2).sum()).item()/values.shape[0]
        else:
            return np.nan

    @property
    def median_Uncert(self):
        r"""Estimate uncertainty of median depth by bootstrap.
        """

        from scipy.stats import bootstrap

        if self.isFlooded():
            B = bootstrap((self.wse_values.values,), np.median, n_resamples=10000, random_state=42)
            return B.standard_error.item()
        else:
            return np.nan

    @property
    def distance_Uncert(self):
        r"""Estimate uncertainty of distance metric by bootstrap.
        """

        def bootstrap_distance(x):
            max_depth = np.quantile(x, 0.95)
            return self._distance(max_depth)
        
        from scipy.stats import bootstrap

        if self.isFlooded():
            B = bootstrap((self.wse_values.values,), bootstrap_distance, n_resamples=2000, random_state=42)
            return B.standard_error.item()
        else:
            return np.nan
        
    @property
    def flood_frac_Uncert(self):
        r"""Estimate uncertainty of flood fraction by bootstrap.
        """

        from scipy.stats import bootstrap

        if self.isFlooded():
            values = self.da.values
            values_bool = np.where(np.isnan(values), 0, 1).flatten()
            B = bootstrap((values_bool,), np.mean, n_resamples=10000, random_state=42)
            return B.standard_error.item()

        else:
            return np.nan

    @property
    def footprint(self):
        from shapely.geometry import Polygon

        # Extract corners
        lat_min = float(self.da['y'].min())
        lat_max = float(self.da['y'].max())
        lon_min = float(self.da['x'].min())
        lon_max = float(self.da['x'].max())

        # Define corners in (lon, lat) order
        poly = Polygon([
            (lon_min, lat_min),
            (lon_min, lat_max),
            (lon_max, lat_max),
            (lon_max, lat_min),
            (lon_min, lat_min)])

        return gpd.GeoDataFrame(index=[0], geometry=[poly], crs=self.da.rio.crs)
    
    @property
    def stats(self):
        r"""Stamp patch statistics into the patch GeoDataFrame.

        Parameters
        ----------
        None

        Returns
        -------
        gdf: gpd.GeoDataFrame
            patch statistics as a single-row GeoDataFrame
        """

        gdf = self.gdf.copy().drop('geometry',axis=1)
        gdf = gdf.rename(dict(point_geometry='geometry'), axis=1)

        gdf['IsFlooded']       = self.isFlooded()
        gdf['FloodFraction']   = self.flood_frac
        gdf['Mean']            = self.mean_depth
        gdf['Median']          = self.median_depth
        gdf['Max']             = self.max_depth
        gdf['Distance']        = self.distance

        if self._uncert:
            gdf['_FloodFrac_Uncert'] = self.flood_frac_Uncert
            gdf['_Mean_Uncert']      = self.mean_Uncert
            gdf['_Median_Uncert']    = self.median_Uncert
            gdf['_Max_Uncert']       = self.max_Uncert
            gdf['_Distance_Uncert']  = self.distance_Uncert            

        gdf['DateTime']          = pd.to_datetime(self.Scene.metadata['time_granule_start'])
        gdf['SwotFile']          = self.Scene.shortname
 
        return gdf

class VIIRS:
    r""" Class to access easily to VIIRS data:
    _ Flood percentage (0-1) series at HF level contained in a parquet file.
    _ HF metadata of non-Safe facilities, stored in a geojson file. 
    """

    path_series = None
    path_hfs    = None
    
    def __init__(self):
        r"""Reads both the parquet and geojson files.
        """

        self.gdf = gpd.read_file(__class__.path_hfs)
        self.df  = pd.read_parquet(__class__.path_series)

    @classmethod
    def set_paths(cls, path_series=None, path_hfs=None):
        r"""Update class-level input paths used by new VIIRS instances.

        Parameters
        ----------
        path_series: str | pathlib.Path | None
            New path to the VIIRS parquet time-series file.
        path_hfs: str | pathlib.Path | None
            New path to the facilities GeoJSON file.

        Returns
        -------
        None
        """

        if path_series is not None:
            cls.path_series = Path(path_series)
        if path_hfs is not None:
            cls.path_hfs = Path(path_hfs)

    def getSeries(self, id):
        r"""Return VIIRS time series for a given Health Facility--indicated by its ID.

        Parameters
        ----------
        id: str

        Returns
        -------
        df : pandas.DataFrame | None
        """
        if id in self.ids:
            return self.df.loc[self.df['ID']==id].set_index('date').sort_index()
        else:
            return None
    
    @property
    def ids(self):
        r"""Return list of available facility IDs.
        """
        return self.gdf['ID'].unique().tolist()
    
    @property
    def random_id(self):
        r"""Return a random facility ID from available IDs
        """
        return pd.Series(self.ids).sample(1).values.item()

class Results:
    r"""Class designed for reading and analyzing an output dataset
    of geojson files, each containing the flood descriptors calculated
    for a SWOT-mission frame."""

    descriptors           = {'FloodFraction':'_FloodFrac_Uncert',
                             'Mean':        '_Mean_Uncert',
                             'Median':      '_Median_Uncert',
                             'Max':         '_Max_Uncert',
                             'Distance':  '_Distance_Uncert'}
    
    descriptors_anomalies = {'FloodFraction' : '_FloodFrac_Uncert',
                             'Mean_anomaly'  : 'Mean_Anomaly_Uncert',
                             'Median_anomaly': 'Median_Anomaly_Uncert',
                             'Max_anomaly'   : 'Max_Anomaly_Uncert',
                             'Distance'      : '_Distance_Uncert'}

    def __init__(self, path):
        r"""Reads flood descriptor statisctis from geojson files.
        
        Parameteres
        -----------
        path: str | pathlib.Path
            Folder where output geojson files are stored.
        """

        self.path = Path(path)
        self.raw = self._read(self.path)
        self.gdf = self.raw.copy()

        self._outlier_removed = False
    
    def _read(self, path):
        r"""Read all descriptor GeoJSON files and normalize dtypes.

        Parameters
        ----------
        path: pathlib.Path

        Returns
        -------
        geopandas.GeoDataFrame
        """
        assert path.exists(), f"Provided path does not exist: {path}"
        gdf = pd.concat([gpd.read_file(i) for i in tqdm(list(path.rglob('*.geojson')))]).set_index('DateTime')
        descriptors = [*list(__class__.descriptors.keys()), *list(__class__.descriptors.values())]
        gdf[descriptors] = gdf[descriptors].apply(pd.to_numeric)
        return gdf

    def del_outliers(self, field='Mean', z=3):
        r"""Remove outliers using a z-score filter based on a specific field.
        
        Parameters
        ----------
        field: str
            Field of geojson files to consider for outliers remotion.
        z: int.
            Z-score (ie. number of standard deviation) number to consider as outliers threshold.
        
        Returns
        -------
        Results
        """

        self.gdf['z_score_Field'] = self.gdf.groupby('ID')[field].transform(lambda x: (x - x.mean()) / x.std())
        self.gdf['IsOutlier'] = self.gdf['z_score_Field'].abs() > z

        print(f"With outliers: {self.gdf.shape[0]} entries")
        print(f"Without outliers: {self.gdf[~self.gdf['IsOutlier']].shape[0]} entries")

        self.gdf = self.gdf.loc[~self.gdf['IsOutlier']]
        self._outlier_removed = True
        return self

    def to_anomalies(self, how='min'):
        r"""Convert all series to anomalies respect certain value (how)
        
        ONLY CREATE ANOMALIES OF DEPTH-RELATED VARIABALES. Flood percentage and HF-distance to Max are
        excluded.
       
        Parameters
        ----------
        how: str. Default: 'min'
            Method for creating anomalies. Eg, 'min', 'max', 'mean', etc.
            By default 'how' is set to 'min' so that all values are levelled from zero.
        
        Returns
        -------
        Results
        """

        gdf = self.gdf.copy()

        # Anomalies of time series values
        gdf['Mean_anomaly'] = gdf['Mean'] - gdf.groupby('ID')['Mean'].transform(how)
        gdf['Median_anomaly']    = gdf['Median'] - gdf.groupby('ID')['Median'].transform(how)
        gdf['Max_anomaly']  = gdf['Max'] - gdf.groupby('ID')['Max'].transform(how)

        self.gdf = gdf
        self._get_UncertAnomaly('Mean')
        self._get_UncertAnomaly('Median')
        self._get_UncertAnomaly('Max')

        # Round all descriptors fields:
        fields = [i for i in self.getSerie('274').columns if any([j in i for j in ['Anomaly', 'anomaly', 'Uncert']])]

        self.gdf[fields] = self.gdf[fields].astype(float).round(3)

        return self

    def _get_UncertAnomaly(self, field):
        r"""Propagate uncertainty for anomaly values of a descriptor field.

        Parameters
        ----------
        field: str
            Descriptor name in ``descriptors`` mapping.

        Returns
        -------
        None
        """
        
        gdf = self.gdf
        uncert_field = __class__.descriptors[field]
        output_field = f"{field}_Anomaly_Uncert"

        argmins = gdf.groupby('ID')[field].idxmin()

        for _id in self.ids:
            mask = gdf['ID'] == _id
            series = gdf.loc[mask, uncert_field].dropna()
            if series.empty or _id not in argmins.index:
                continue

            argmin = argmins.loc[_id]
            if argmin not in series.index:
                continue

            uncert_baseline = series.loc[argmin]
            gdf.loc[mask, output_field] = np.sqrt(series**2 + uncert_baseline**2)

        self.gdf = gdf

    def get_Oscil(self, field):
        r"""Get oscillation range from specific field.
        
        Patameters
        ----------
        field: str.
            Descriptor name in ``descriptors`` mapping.
        
        Returns
        -------
        gdf: gpd.GeoDataFrame
        """

        assert field in self.gdf.columns, f"No existing field named: {field}"
        return self.gdf.groupby('ID')[field].agg(lambda x: x.max() - x.min())
    
    def getSerie(self, _id):
        r"""Return time series of all fields of given Health Facility.
        
        Parameters
        ----------
        _id: str.
            Health Facility ID.
        
        Returns
        -------
        gdf: gpd.GeoDataFrame
        """

        return self.gdf.loc[self.gdf['ID']==_id].sort_index()
    
    def plot(self, id, field='Median_anomaly'):
        r"""Plot one SWOT descriptor time series for a selected ID.

        Parameters
        ----------
        id: str
        field: str

        Returns
        -------
        matplotlib.axes.Axes
        """

        return self.gdf.loc[self.gdf['ID']==id][field].dropna().plot(title=id, figsize=(10,4))
    
    def plotWithViirs(self, id=None, field='FloodFraction', plot=True):
        r"""Compare SWOT and VIIRS series and optionally plot trends.

        Parameters
        ----------
        id: str | None
            Facility ID. If ``None``, a random ID is used.
        field: str
            SWOT field to compare against VIIRS flood fraction.
        plot: bool
            Whether to produce plots or return correlation only.

        Returns
        -------
        tuple | float
            ``(axes, correlation)`` when ``plot=True``; otherwise correlation.
        """

        def corr(_swot, viirs):
            r"""Compute trend-based Pearson correlation between series.

            Parameters
            ----------
            _swot: pandas.Series
            viirs: pandas.Series

            Returns
            -------
            tuple
                ``(correlation, swot_trend, viirs_trend_reindex)``.
            """

            from statsmodels.tsa.seasonal import STL
            from statsmodels.nonparametric.smoothers_lowess import lowess

            viirs_trend = STL(viirs, period=60).fit().trend.rename('VIIRS-trend')  # Good for regular series
            swot_trend_array  = lowess(_swot.values, _swot.index, frac=0.15)       # Good for irregulares
            swot_trend        = pd.Series(swot_trend_array[:,1],
                                        index=pd.to_datetime(swot_trend_array[:, 0]),
                                        name='SWOT-trend')
            viirs_trend_reindex = viirs_trend.reindex(swot_trend.index, method='nearest')
                
            r = np.corrcoef(viirs_trend_reindex, swot_trend)[0,1]

            return round(r, 3), swot_trend, viirs_trend_reindex

        id = id if id is not None else self.random_id

        # SWOT series
        _swot = self.getSerie(id)[field]
        _swot.index = _swot.index.normalize()
        _swot.index = _swot.index.tz_localize(None)

        viirs = VIIRS().getSeries(id)['pct_flooded']
        viirs = viirs.loc[_swot.index.min():_swot.index.max()]

        r, swot_trend, viirs_trend_reindex = corr(_swot, viirs)

        if plot:
            fig, ax = plt.subplots(3,1, figsize=(12,10), sharex=True)

            _swot.plot(ax=ax[0], label='SWOT', title=f'ID: {id}', grid=True, legend=True)
            viirs.plot(ax=ax[1], color='green', grid=True, label='VIIRS', legend=True)

            swot_trend.plot(ax=ax[2], grid=True)
            viirs_trend_reindex.plot(ax=ax[2].twinx(), color='green')
            ax[2].text(0.01,0.93, f"r: {r}", transform=ax[2].transAxes)
            ax[2].text(0.93,0.93, f"TRENDS", transform=ax[2].transAxes)

            plt.subplots_adjust(hspace=0.1)
            [i.set_xlim(_swot.index.min(), _swot.index.max()) for i in ax]

            return ax, r

        return r

    def scatter(self, id=None, return_xy=False, **kwargs):
        r"""Compare VIIRS and SWOT values with a scatter plot.

        The method merges both sources by ``ID`` and date (normalized to day),
        keeps only overlapping observations, and plots ``pct_flooded`` (VIIRS)
        against ``FloodFraction`` (SWOT) for one health facility.

        Parameters
        ----------
        id: str | None
            Facility identifier. If ``None``, a random ID is used.
        return_xy: bool
            If ``True``, return the prepared DataFrame used for the scatter
            instead of drawing the plot.
        **kwargs: dict
            Extra keyword arguments forwarded to ``matplotlib.axes.Axes.scatter``.

        Returns
        -------
        pandas.DataFrame | matplotlib.axes.Axes
            DataFrame with columns ``pct_flooded`` and ``FloodFraction`` when
            ``return_xy=True``; otherwise, the plot ``Axes`` object.
        """

        id = id if id is not None else self.random_id

        # Obtention of MERGE dataset
        one = VIIRS().df.rename(dict(date='DateTime'), axis=1)[['hf_id', 'DateTime', 'pct_flooded', 'ID']]
        other = self.gdf.reset_index()[['DateTime', 'ID', 'FloodFraction']]
        other['DateTime'] = other['DateTime'].apply(lambda x: x.tz_localize(None).normalize())
        merge = one.merge(other, on=['ID','DateTime'], how='inner')
        
        # Selection of ID and plot
        df = merge.loc[merge['ID']==id].set_index('DateTime').sort_index()[['pct_flooded', 'FloodFraction']]

        if return_xy:
            return df
        fig, ax = plt.subplots()
        ax.scatter(df['pct_flooded'], df['FloodFraction'], **kwargs)
        ax.set_xlabel('VIIRS - % flooded')
        ax.set_ylabel('SWOT % flooded')
        ax.set_title(f"ID: {id}")
        ax.set_xlim(0,1)
        ax.set_ylim(0,100)
        return ax
    
    def plotOutlier(self, id=None, field='Median', **kwargs):
        r"""Plot a series before and after outlier removal for one facility.

        Parameters
        ----------
        id: str | None
            Facility identifier. If ``None``, a random ID is used.
        field: str
            Descriptor field to plot.
        **kwargs: dict
            Extra keyword arguments forwarded to the filtered-series plot call.

        Returns
        -------
        matplotlib.axes.Axes
        """
        
        assert self._outlier_removed, f"First remove outliers using `del_outliers` method"
        id = id if id is not None else self.random_id

        original = self.raw

        ax = original.loc[original['ID']==id][field].sort_index().dropna().plot(figsize=(14,6),
                                                                   color='black',
                                                                   alpha=0.75,
                                                                   label='Original series',
                                                                   title=id,
                                                                   legend=True, zorder=0, ls='--', lw=0.8)
        self.getSerie(id)[field].sort_index().plot(ax=ax, color='cornflowerblue',
                                                            label='Filtered series', legend=True, zorder=4,
                                                            **kwargs)

        return ax
    
    def boxPlot(self, field, **kwargs):
        r"""Draw sorted boxplot of a descriptor across facility IDs.

        Parameters
        ----------
        field: str
            Descriptor field to plot.
        **kwargs: dict
            Extra arguments forwarded to ``seaborn.boxplot``.

        Returns
        -------
        matplotlib.axes.Axes
        """

        import seaborn as sns
        ax = kwargs.pop('ax', None)
        created_fig = ax is None

        if created_fig:
            _, ax = plt.subplots(figsize=(16, 6))

        # Ordering gdf according to oscilation range of filed:
        oscil_ids = self.get_Oscil(field).sort_values(ascending=False).index.tolist()
        orden = {id_: i for i, id_ in enumerate(oscil_ids)}
        gdf = self.gdf.copy().assign(_orden=self.gdf["ID"].map(orden)
                                     ).sort_values("_orden", na_position="last"
                                                   ).drop(columns="_orden").reset_index()

        sns.boxplot(x='ID',
                    y=field,
                    data=gdf,
                    whis=(0,100),
                    showfliers=False,
                    ax=ax,
                    **kwargs)
        
        if created_fig:
            plt.show()

        return ax
    
    def oscilHist(self, field, **kwargs):
        r"""Plot histogram of oscillation range for a descriptor.

        Parameters
        ----------
        field: str
            Descriptor field to evaluate.
        **kwargs: dict
            Extra plotting options. passed to seaborn plot.

        Returns
        -------
        None
        """
        import seaborn as sns
        fig, ax = plt.subplots(figsize=(10,5))
        data = self.get_Oscil(field)
        sns.histplot(data=data, bins=150, kde=True, ax=ax, **kwargs)

        q75 = data.quantile(0.75)
        q95 = data.quantile(0.95)

        ax.axvline(q75, color='orange', linestyle='--', linewidth=1.8, label='Q75')
        ax.axvline(q95, color='red', linestyle='--', linewidth=1.8, label='Q95')
        
        ax.legend()
        ax.set_title(f'Histogram of Oscillation Range - {field}')
        ax.set_xlim(0,90)
        ax.set_ylim(0,60)
        ax.text(75,30,
        f"N Series: {self.gdf.groupby('ID')[field].agg(lambda x: x.any()).sum()}")

    def oscilMap(self, field, **kwargs):
        r"""Map of oscillation range for a descriptor by facility location.

        Parameters
        ----------
        field: str
            Descriptor field to map.
        **kwargs: dict
            Extra arguments forwarded to ``GeoDataFrame.plot``.

        Returns
        -------
        matplotlib.axes.Axes
        """

        oscil_gdf = self.gdf.set_index('ID')[['geometry']].join(self.get_Oscil(field), how='left')
        ax = oscil_gdf.plot(column=field, cmap='Spectral_r', legend=True, **kwargs)
        return ax

    def reset(self):
        r"""Reset working GeoDataFrame to its raw loaded state.
        """

        self.gdf = self.raw
        print(f"Reset gdf to raw.")

    @property
    def ids(self):
        r"""Return list of IDs present in current results table.
        """

        return self.gdf['ID'].unique().tolist()
    
    @property
    def random_id(self):
        r"""Return a random ID from current results table.
        """

        return pd.Series(self.ids).sample(1).values.item()

def read_frame(path, dem_path=None):
    r"""Create SWOT object from frame path, returning None on failure.

    Parameters
    ----------
    path: pathlib.Path
        NetCDF frame path.
    dem_path: str | pathlib.Path | None
        Optional DEM path.

    Returns
    -------
    SWOT | None
    """
    try:
        return SWOT(path, dem_path=dem_path, dem_type='', dem_crs=4326)
    except:
        return None

def getBatchStats(X, *quals, folder=None, returns=None, stats_datarray='ground', _clean=True, _uncert=False):
    r"""Apply standard filtering pipeline and return or save batch outputs.

    Parameters
    ----------
    X: SWOT
        SWOT scene object.
    *quals: int
        ``wse_qual`` classes to retain. E.g., 0, 1
    folder: pathlib.Path | None
        Output folder for raster/all outputs.
    returns: str | None
        One of ``'raster'``, ``'stats'``, ``'all'``.
    stats_datarray: str
        Either ``'geoid'`` or ``'ground'``.
    _clean: bool
        Whether to free scene RAM after processing.
    _uncert: bool
        Whether to include uncertainty metrics in stats mode.

    Returns
    -------
    geopandas.GeoDataFrame | None
    """
    
    msg = f"stats_datarray arg should be 'geoid or 'ground'. Provided: {stats_datarray}"
    assert stats_datarray in ['geoid', 'ground'], msg

    try:
        X.filtrate_qual(*quals).filtrate_cross(10000,60000)\
                               .filtrate_water(0.15)\
                               .filtrate_dark(0.2)\
                               .filtrate_uncert(0.5)\
                               .filtrate_layover(95)\
                               .filtr_bitwise('geolocation_qual_degraded',
                                        'classification_qual_degraded',
                                        'value_bad')\
                        
        if stats_datarray=='ground':
            X.toDepth()

        if returns=='raster':
            if stats_datarray=='ground':
                X.depth.rio.reproject(4326)
                X.save(folder)
            elif stats_datarray=='geoid':
                depth = X.da.wse.rio.reproject(4326)
                depth.rio.to_raster(folder/fr"{X.filename}_FloodDepth.tif")

            clean(X) if _clean else None
        elif returns=='stats':
            shp = X.stats.get_Stats(stats_datarray=stats_datarray,
                                    uncert=_uncert).copy().to_crs(4326)
            clean(X) if _clean else None
            return shp

        elif returns=='all':
            assert folder is not None, f"Folder lacking"
            X.stats.saveAll(folder)
            clean(X) if _clean else None
    except Exception as e:
        logging.error(f"Error with frame {X.name}")
        raise e

def clean(Scene):
    r"""Release scene resources and trigger garbage collection.

    Parameters
    ----------
    Scene: SWOT

    Returns
    -------
    None
    """
    import gc
    import matplotlib.pyplot as plt
    # Liberar memoria de xarray
    try:
        Scene.raw.close()
    except Exception:
        pass
    try:
        Scene.da.close()
    except Exception:
        pass
    try:
        Scene.depth.close()
    except Exception:
        pass
    # Eliminar atributos grandes
    for attr in ['raw', 'da', '_depth', '_dem']:
        if hasattr(Scene, attr):
            delattr(Scene, attr)
    # Cerrar figuras de matplotlib
    plt.close('all')
    # Forzar recolección de basura
    gc.collect()
