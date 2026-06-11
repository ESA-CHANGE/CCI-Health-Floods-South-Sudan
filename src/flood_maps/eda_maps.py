# -*- coding: utf-8 -*-

__author__ = "Macarena Merida Floriano, mmef"
__maintainer__ = "Macarena Merida Floriano, mmef"
__email__ = "mmef@gmv.com"

"""
CHANGE: data processing script
=============================================================================
Script for exploratory data analysis (EDA) of annual flood maps. Before
running this script, ensure that the maps have been binarized using the
binarize_maps.py and hazard_maps.py scripts. In addition, a CSV file with Health
Facilities with locations is required for spatial analysis (generated in 
get_valid_facilities.py).

Inputs are annual flood maps for South Sudan, with the following bands:
- flood_frequency: Proportion of days inundated in the year (0-1) 
- flood_duration: Total number of days inundated per year 
- count_valid_full: Number of valid observed days per pixel
- max_consecutive_full: Maximum number of consecutive inundated days

The script performs the following analyses:
1. Basic statistics: mean, std, min, max, median, percentiles, and percentage
of never flooded pixels (0% frequency).
2. Temporal trends: Analyzes trends over years for each variable.
3. Spatial consistency: Evaluates spatial patterns and consistency across years.
4. Correlation analysis: Examines correlations between variables and across years.
5. Data quality assessment: Checks for inconsistencies and data quality issues.
6. Visualizations: Generates maps, boxplots, and trend plots to illustrate findings.

"""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import os
import pandas as pd
import rasterio
from scipy import stats
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Style configuration
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class FloodMapEDA:
    r"""Class for exploratory data analysis (EDA) of annual flood maps.
    
    Methods
    -------
    load_data(): Load data from raster files.
    get_valid_mask(year): Get mask of valid pixels for a given year.
    get_flooded_mask(year, threshold=0): Get mask of flooded pixels for a given year.
    basic_statistics(): Calculate basic descriptive statistics by year and band.
    temporal_trends_analysis(): Analyze temporal trends in the variables.
    spatial_consistency_analysis(): Analyze spatial consistency across years.
    correlation_analysis(): Analyze correlations between bands and across years.
    data_quality_assessment(): Assess data quality and identify inconsistencies.
    visualize_basic_maps(): Visualize basic maps for each year and band.
    visualize_statistics(): Visualize descriptive statistics and trends.
    visualize_spatial_analysis(): Visualize spatial consistency and patterns.
    visualize_correlations(): Visualize correlation matrices.   
    """
    
    def __init__(self, file_paths=None, plot_dir=None):
        r""" Initilialize the analyzer.
        
        Parameters
        ----------
        file_paths: dict or None
            Dictionary with year as key and file path as value, 
            e.g. {2012: 'path/to/2012.tif', ...}. If None, synthetic data 
            will be generated for testing purposes.
        plot_dir: str or None
            Directory where to save the generated plots. If None, plots will not be saved.
        
        Arguments
        ---------
        data : dict
            Dictionary to store data arrays by year and band.
        metadata : dict
            Dictionary to store metadata for each year.
        years : list
            List of years available in the data.
        bands : list
            List of bands in the data (e.g. flood_frequency, flood_duration, etc.).
        nodata_value : int
            Value used to represent no-data in the raster files.
        """
        
        self.file_paths = file_paths
        self.plot_dir = plot_dir
        self.data = {}  
        self.metadata = {}
        self.years = []
        self.bands = ['flood_frequency', 'flood_duration', 'count_valid_full', 'max_consecutive_full']
        self.nodata_value = -9999
        
    def load_data(self):
        r"""It loads data from raster files.
        If file_paths is None, it generates synthetic data for testing purposes.
        The method reads each raster file, extracts the data for each band, and
        stores it in the data dictionary. It also extracts metadata such as
        observed days, coordinate reference system (CRS), transform, and bounds.
        """
        
        for year, path in self.file_paths.items():
            with rasterio.open(path) as src:
                # Read all bands
                self.data[year] = {
                    'flood_frequency': src.read(1),
                    'flood_duration': src.read(2),
                    'count_valid_full': src.read(3),
                    'max_consecutive_full': src.read(4)
                }
                # Extract metadata
                self.metadata[year] = {
                    'OBSERVED_DAYS': src.tags().get('OBSERVED_DAYS', 365),
                    'crs': src.crs,
                    'transform': src.transform,
                    'bounds': src.bounds
                }
        
        self.years = sorted(self.data.keys())
        print(f"Data loaded for years: {self.years}")
    
    def get_valid_mask(self, year):
        r"""This method gets the mask of valid pixels (not no-data) 
        for a given year.
        
        Parameters
        ----------
        year: int
            Year for which to get the valid mask.
        
        Returns
        -------
        numpy.ndarray
            Boolean mask where True indicates valid pixels and False indicates no-data.
        """
        
        freq = self.data[year]['flood_frequency']
        return freq != self.nodata_value
    
    def get_flooded_mask(self, year, threshold=0):
        r"""This method gets the mask of pixels that have been flooded.
        
        Parameters
        ----------
        year: int
            Year for which to get the flooded mask.
        threshold: float
            Threshold for flood frequency to consider a pixel as flooded 
            (default: 0).
        
        Returns
        -------
        numpy.ndarray
            Boolean mask where True indicates flooded pixels and False 
            indicates non-flooded pixels.
        """
        
        valid = self.get_valid_mask(year)
        freq = self.data[year]['flood_frequency']
        return valid & (freq > threshold)
    
    def basic_statistics(self):
        r"""This method calculates basic descriptive statistics by year and band.
        
        Returns
        -------
        pandas.DataFrame
            DataFrame containing basic statistics for each year and band.
        """
        
        stats_dict = []
        
        for year in self.years:
            valid_mask = self.get_valid_mask(year)
            
            for band in self.bands:
                # Assure float type for calculations
                data = self.data[year][band][valid_mask].astype(np.float64)
                
                if len(data) == 0:
                    continue
                
                # DEBUG
                if band == 'flood_frequency':
                    print(f"\nDEBUG {year} - {band}:")
                    print(f"  Total píxeles válidos: {len(data)}")
                    print(f"  Píxeles > 0: {np.sum(data > 0)}")
                    print(f"  Suma total: {np.sum(data)}")
                    print(f"  Media (float64): {np.mean(data)}")
                    print(f"  Max: {np.max(data)}")
                    print(f"  Tipo original: {self.data[year][band].dtype}")
                
                # Basic statistics
                stat = {
                    'year': year,
                    'band': band,
                    'count': len(data),
                    'mean': float(np.mean(data)), 
                    'std': float(np.std(data)),
                    'min': float(np.min(data)),
                    'max': float(np.max(data)),
                    'median': float(np.median(data)),
                    'q25': float(np.percentile(data, 25)),
                    'q75': float(np.percentile(data, 75)),
                    'zeros': int(np.sum(data == 0)),
                    'zeros_pct': float(np.sum(data == 0) / len(data) * 100)
                }
                stats_dict.append(stat)
        
        self.stats_df = pd.DataFrame(stats_dict)
        
        # DEBUG
        print("\nDEBUG DataFrame medias flood_frequency:")
        print(self.stats_df[self.stats_df['band'] == 'flood_frequency'][['year', 'mean', 'count', 'zeros_pct']])
        
        return self.stats_df
    
    def basic_statistics_flood(self):
        r"""This method calculates basic descriptive statistics for 
        flooded pixels by year and band.
        
        Returns
        -------
        pandas.DataFrame
            DataFrame containing basic statistics for flooded pixels for each 
            year and band.
        """
        
        stats_dict = []
        
        for year in self.years:
            valid_mask = self.get_valid_mask(year)
            
            for band in self.bands:
                # Assure float type for calculations
                data = self.data[year][band][valid_mask].astype(np.float64)
                data_flood = data[data>0]   # Filter only flooded pixels
                if len(data) == 0:
                    continue
                
                # DEBUG
                if band == 'flood_frequency':
                    print(f"\nDEBUG {year} - {band}:")
                    print(f"  Total píxeles inundados: {len(data_flood)}")
                    print(f"  Media (float64): {np.mean(data_flood)}")
                    print(f"  Max: {np.max(data_flood)}")
                
                # Basic statistics for flooded pixels
                stat = {
                    'year': year,
                    'band': band,
                    'count': len(data_flood),
                    'mean': float(np.mean(data_flood)),  
                    'std': float(np.std(data_flood)),
                    'min': float(np.min(data_flood)),
                    'max': float(np.max(data_flood)),
                    'median': float(np.median(data_flood)),
                    'q25': float(np.percentile(data_flood, 25)),
                    'q75': float(np.percentile(data_flood, 75)),
                    'zeros': int(np.sum(data == 0)),
                    'zeros_pct': float(np.sum(data == 0) / len(data) * 100)
                }
                stats_dict.append(stat)
        
        self.stats_df_flood = pd.DataFrame(stats_dict)
        
        # DEBUG
        print("\nDEBUG DataFrame medias flood_frequency:")
        print(self.stats_df[self.stats_df['band'] == 'flood_frequency'][['year', 'mean', 'count', 'zeros_pct']])
        
        return self.stats_df_flood
    
    
    def temporal_trends_analysis(self):
        r"""It analyzes temporal trends in the variables.
        For each band, it calculates the yearly mean and performs a linear 
        regression to identify trends over time. It returns a dictionary with
        the slope, R-squared, p-value, and trend direction for each band.
        
        Returns
        -------
        trends: dict
            Dictionary containing the slope, R-squared, p-value, and trend 
            direction for each band.
        """
        
        trends = {}
        
        for band in self.bands:
            yearly_means = []
            for year in self.years:
                valid_mask = self.get_valid_mask(year)
                data = self.data[year][band][valid_mask]
                yearly_means.append(np.mean(data))
            
            # Linear regression for trend analysis
            slope, intercept, r_value, p_value, std_err = stats.linregress(self.years, yearly_means)
            
            trends[band] = {
                'slope': slope,
                'r_squared': r_value**2,
                'p_value': p_value,
                'trend': 'increasing' if slope > 0 else 'decreasing',
                'yearly_means': yearly_means
            }
        
        self.trends = trends
        return trends
    
    def spatial_consistency_analysis(self):
        r"""Analyzes spatial consistency across years.
        It creates a data stack for each band across years and calculates 
        the mean, standard deviation, coefficient of variation, and maximum 
        change for each pixel. It also identifies pixels with valid data 
        across all years. The results are stored in a dictionary for each band.
        
        Returns
        -------
        consistency_maps: dict
            Dictionary containing the mean, standard deviation, coefficient of variation,
            maximum change, and valid mask for each band.
        """
        
        # Create data stack for spatial analysis
        consistency_maps = {}
        
        for band in self.bands:
            stack = np.stack([self.data[year][band] for year in self.years])
            valid_mask = np.stack([self.get_valid_mask(year) for year in self.years]).all(axis=0)
            
            # Compute spatial coefficient of variation
            mean_map = np.mean(stack, axis=0)
            std_map = np.std(stack, axis=0)
            cv_map = np.zeros_like(mean_map)
            cv_map[valid_mask] = std_map[valid_mask] / (mean_map[valid_mask] + 1e-10)
            
            # Detect abrupt changes in pixels
            max_change = np.max(stack, axis=0) - np.min(stack, axis=0)
            
            consistency_maps[band] = {
                'mean': mean_map,
                'std': std_map,
                'cv': cv_map,
                'max_change': max_change,
                'valid_mask': valid_mask
            }
        
        self.consistency = consistency_maps
        return consistency_maps
    
    def correlation_analysis(self):
        r"""Analyzes correlations between bands and across years.
        It calculates the correlation matrix between bands for each year 
        and the temporal correlation for each band across years. The results 
        are stored in a dictionary.
        
        Returns
        -------
        correlations: dict
            Dictionary containing the correlation matrices for each year 
            and the temporal correlation for each band.
        """
        
        correlations = {}
        
        # Correlation between bands for each year
        for year in self.years:
            valid_mask = self.get_valid_mask(year)
            df_year = pd.DataFrame({
                band: self.data[year][band][valid_mask] 
                for band in self.bands
            })
            correlations[f'year_{year}'] = df_year.corr()
        
        # Temporal correlation for each band
        temporal_corr = {}
        for band in self.bands:
            band_data = {year: self.data[year][band][self.get_valid_mask(year)] for year in self.years}
            # Use intersection of valid pixels
            min_len = min(len(v) for v in band_data.values())
            band_matrix = np.array([v[:min_len] for v in band_data.values()])
            temporal_corr[band] = np.corrcoef(band_matrix)
        
        correlations['temporal'] = temporal_corr
        self.correlations = correlations
        return correlations
    
    def data_quality_assessment(self):
        r"""Evaluates the quality of the data.
        It checks for inconsistencies such as pixels with count_valid_full
        greater than OBSERVED_DAYS, and logical relationships between bands.
        
        Returns
        -------
        quality_report: dict
            Dictionary containing the quality assessment results for each year.
        
        """
        
        quality_report = {}
        
        for year in self.years:
            valid_mask = self.get_valid_mask(year)
            total_pixels = valid_mask.size
            valid_pixels = np.sum(valid_mask)
            
            # Convert OBSERVED_DAYS to int, with error handling
            obs_days_raw = self.metadata[year].get('OBSERVED_DAYS', 365)
            try:
                obs_days = int(obs_days_raw)
            except (ValueError, TypeError):
                # If conversion fails, use default value and warning
                print(f"Warning: OBSERVED_DAYS invalid for {year}: '{obs_days_raw}'. Using 365.")
                obs_days = 365
            
            count_valid = self.data[year]['count_valid_full'][valid_mask]
            
            #  PPixels where count_valid > OBSERVED_DAYS (error)
            invalid_count = np.sum(count_valid > obs_days)
            
            # Check logical relationships between bands
            freq = self.data[year]['flood_frequency'][valid_mask]
            dur = self.data[year]['flood_duration'][valid_mask]
            max_consec = self.data[year]['max_consecutive_full'][valid_mask]
            
            if year == 2012:
                print(f"\nDEBUG 2012 - max_consecutive_full:")
                print(f"  dtype: {max_consec.dtype}")
                print(f"  min: {np.min(max_consec)}")
                print(f"  max: {np.max(max_consec)}")
                print(f"  unique values (top 10): {np.unique(max_consec)[-10:]}")
                print(f"  count of -9999: {np.sum(max_consec == -9999)}")
                print(f"  count of 0: {np.sum(max_consec == 0)}")
                print(f"  count of >0: {np.sum(max_consec > 0)}")
                
                # Specific case: freq=0 but consecutives>0
                mask_problem = (freq == 0) & (max_consec > 0)
                print(f"  freq=0 & consecutives>0: {np.sum(mask_problem)}")
                if np.sum(mask_problem) > 0:
                    print(f"  Consecutive values in these cases: {np.unique(max_consec[mask_problem])}")
            
            # Inconsistency: duration > frequency * observed_days
            obs_days_float = float(obs_days)
            inconsistent_dur = np.sum(dur > (freq * obs_days_float + 1e-5))
            
            # Inconsistency: max_consecutive > duration
            inconsistent_consec = np.sum(max_consec > (dur + 1e-5))
            
            # Inconsistency: freq=0 but dur>0 or consec>0
            zero_freq_incons = np.sum((freq == 0) & ((dur > 0) | (max_consec > 0)))
            
            quality_report[year] = {
                'total_pixels': int(total_pixels),
                'valid_pixels': int(valid_pixels),
                'coverage_pct': valid_pixels / total_pixels * 100,
                'pixels_count_gt_observed': int(invalid_count),
                'duration_inconsistencies': int(inconsistent_dur),
                'consecutive_inconsistencies': int(inconsistent_consec),
                'zero_freq_inconsistencies': int(zero_freq_incons),
                'observed_days_metadata': obs_days,
                'mean_count_valid': float(np.mean(count_valid)),
                'completeness': float(np.mean(count_valid)) / obs_days_float * 100
            }
        
        self.quality = quality_report
        return quality_report
    
    
    def visualize_basic_maps(self):
        r"""Basic visualization of the maps - a large image with all the 
        years.
        It creates and saves a figure with subplots for each year and band, 
        showing the spatial distribution of the variables. The maps are 
        displayed using a consistent color scale, and no-data values are 
        masked out.
        """
        
        n_years = len(self.years)
        n_bands = len(self.bands)
        
        # Proportional size: 5 inches per year, 4 per band
        fig, axes = plt.subplots(n_years, n_bands, figsize=(n_bands * 5, n_years * 4))
        
        # Ensure axes is 2D even if there is only one year
        if n_years == 1:
            axes = axes.reshape(1, -1)
        if n_bands == 1:
            axes = axes.reshape(-1, 1)
        
        for i, year in enumerate(self.years):
            for j, band in enumerate(self.bands):
                ax = axes[i, j]
                data = self.data[year][band].copy()
                data[data == self.nodata_value] = np.nan
                
                im = ax.imshow(data, cmap='viridis', interpolation='nearest')
                ax.set_title(f'{year} - {band}', fontsize=10)
                ax.axis('off')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        fig_name = self.plot_dir + '01_basic_maps.png'
        if self.plot_dir is not None:
            plt.savefig(fig_name, dpi=150, bbox_inches='tight')
        plt.show()
        if self.plot_dir is not None:
            print(f"Basic maps saved: {fig_name} ({n_years}x{n_bands} subplots)")
    
    def visualize_statistics(self):
        r"""Visualize descriptive statistics.
        
        It creates and saves figures showing the temporal evolution of means,
        the percentage of never flooded pixels, and boxplots for each band. 
        The plots are designed to clearly illustrate trends and differences 
        between years.
        
        """
        
        if not hasattr(self, 'stats_df'):
            self.basic_statistics()
            self.basic_statistics_flood()
        
        # =============================================================================
        # FIGURE 1: Detailed boxplots by band (ONLY FLOODED PIXELS >0)
        # =============================================================================
        fig_box, axes_box = plt.subplots(2, 2, figsize=(16, 12))
        axes_box = axes_box.flatten()
        
        for idx, band in enumerate(self.bands):
            ax = axes_box[idx]
            band_data_by_year = []
            year_labels = []
            
            for year in self.years:
                valid_mask = self.get_valid_mask(year)
                data = self.data[year][band][valid_mask]
                
                # Filter: only flooded pixels (>0)
                data_inundados = data[data > 0]
                
                print(f"{year} - {band}: {len(data_inundados)} flooded pixels out of {len(data)} valid ({len(data_inundados)/len(data)*100:.2f}%)")
                
                if len(data_inundados) > 0:
                    if len(data_inundados) > 50000:
                        sample = np.random.choice(data_inundados, 50000, replace=False)
                    else:
                        sample = data_inundados
                    band_data_by_year.append(sample)
                    year_labels.append(str(year))
                else:
                    band_data_by_year.append(np.array([0]))
                    year_labels.append(f"{year}\n(no flooded pixels)")
            
            bp = ax.boxplot(band_data_by_year, patch_artist=True, widths=0.6,
                            showfliers=False)
            
            colors = plt.cm.viridis(np.linspace(0, 1, len(self.years)))
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax.set_xticklabels(year_labels, rotation=45, ha='right', fontsize=9)
            ax.set_title(f'{band}\n(Just flooded pixels)', fontsize=11, fontweight='bold')
            ax.set_xlabel('Year')
            ax.set_ylabel('Value')
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('Distribution by Band - Just Flooded Pixels (>0)', fontsize=14, y=1.02)
        plt.tight_layout()
        if self.plot_dir is not None:
            plt.savefig(self.plot_dir + '02_boxplots_inundados.png', dpi=150, bbox_inches='tight')
        plt.show()
        if self.plot_dir is not None:
            print("Flooded pixels Boxplots saved")
        
        # =============================================================================
        # FIGURE 2: General statistical summary (2x2 subplots)
        # =============================================================================
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Temporal evolution of means - TWO Y AXES
        ax1 = axes[0, 0]
        ax1_right = ax1.twinx()  # Right Y axis

        # Left: flood_frequency (0-0.12 approx)
        subset_freq = self.stats_df_flood[self.stats_df_flood['band'] == 'flood_frequency']
        line1 = ax1.plot(subset_freq['year'], subset_freq['mean'], 
                        marker='o', color='crimson', label='flood_frequency', linewidth=2, markersize=6)
        ax1.set_ylabel('Mean Flood Frequency (0-1) only flooded pixels', color='crimson')
        ax1.tick_params(axis='y', labelcolor='crimson')
        ax1.set_ylim(0, 1)  # 0.12 Appropriate scale for frequency

        # Right: duration and consecutive (0-40 approx)
        for band, color in [('flood_duration', 'darkgoldenrod'), ('max_consecutive_full', 'forestgreen')]:
            subset = self.stats_df_flood[self.stats_df_flood['band'] == band]
            line = ax1_right.plot(subset['year'], subset['mean'], 
                                marker='s', color=color, label=band, linewidth=2, markersize=6)

        ax1_right.set_ylabel('Mean (days)', color='black')
        ax1_right.tick_params(axis='y', labelcolor='black')
        ax1_right.set_ylim(0, 365) 

        # Combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1_right.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

        ax1.set_xlabel('Year')
        ax1.set_title('Temporal Evolution of Means for Flooded Pixels')
        ax1.grid(True, alpha=0.3)
                
        # 2. Spatial extent of flooding (count of pixels >0)
        ax2 = axes[0, 1]
        counts = []
        for year in self.years:
            valid_mask = self.get_valid_mask(year)
            data = self.data[year]['flood_frequency'][valid_mask]
            counts.append(np.sum(data > 0))
        
        ax2.bar(self.years, counts, color='steelblue', edgecolor='black')
        ax2.set_xlabel('Year')
        ax2.set_ylabel('Number of Flooded Pixels')
        ax2.set_title('Spatial Extent of Flooding\n(Pixels >0)')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 3. Percentage of zeros (not flooded)
        ax3 = axes[1, 0]
        zero_pivot = self.stats_df.pivot(index='year', columns='band', values='zeros_pct')
        zero_pivot.plot(kind='bar', ax=ax3, width=0.8)
        ax3.set_title('Percentage of Never Flooded Pixels (0%)')
        ax3.set_ylabel('Percentage (%)')
        ax3.legend(title='Band', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax3.tick_params(axis='x', rotation=0)
        
        # 4. Flood frequency histograms
        ax4 = axes[1, 1]
        for year in self.years:
            valid_mask = self.get_valid_mask(year)
            freq_data = self.data[year]['flood_frequency'][valid_mask]
            freq_positive = freq_data[freq_data > 0]
            if len(freq_positive) > 0:
                ax4.hist(freq_positive, bins=30, alpha=0.5, label=f'{year}', density=True)
        ax4.set_xlabel('Flood Frequency (0-1)')
        ax4.set_ylabel('Density')
        ax4.set_title('Flood Frequency Distribution (>0)')
        ax4.legend()
        
        plt.tight_layout()
        if self.plot_dir is not None:
            plt.savefig(self.plot_dir + '02_statistical_analysis_flooded_means.png', dpi=150, bbox_inches='tight')
            print("Statistical analysis saved")        
        plt.show()
        
    
    def visualize_spatial_analysis(self):
        r"""Visualize spatial analysis and temporal consistency.
        
        It creates and saves a figure showing the spatial consistency analysis
        for each band. The first row shows the temporal mean for each pixel, 
        while the second row shows the coefficient of variation (CV) to identify 
        areas with high variability. No-data values are masked out, and consistent 
        areas across years will be highlighted.
        """
        
        if not hasattr(self, 'consistency'):
            self.spatial_consistency_analysis()
        
        fig, axes = plt.subplots(2, len(self.bands), figsize=(20, 10))
        
        # First row: Temporal Mean
        for j, band in enumerate(self.bands):
            ax = axes[0, j]
            mean_map = self.consistency[band]['mean']
            valid = self.consistency[band]['valid_mask']
            display = np.where(valid, mean_map, np.nan)
            
            im = ax.imshow(display, cmap='plasma')
            ax.set_title(f'{band}\nTemporal Mean')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046)
        
        # Second row: Coefficient of Variation
        for j, band in enumerate(self.bands):
            ax = axes[1, j]
            cv_map = self.consistency[band]['cv']
            valid = self.consistency[band]['valid_mask']
            display = np.where(valid, cv_map, np.nan)
            
            im = ax.imshow(display, cmap='RdYlBu_r', vmin=0, vmax=2)
            ax.set_title(f'{band}\nVariation Coef. (σ/μ)')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046)
        
        plt.tight_layout()
        if len(self.years)>1:
            fig_name = self.plot_dir + '03_spatial_consistency.png'
        else:
            fig_name = self.plot_dir + f'03_spatial_consistency_{self.years[0]}.png'
        if self.plot_dir is not None:
            plt.savefig(fig_name, dpi=150, bbox_inches='tight')
            print("Spatial analysis saved")
        plt.show()
        
    
    def visualize_correlations(self):
        r"""Visualize correlation matrices."""
        
        if not hasattr(self, 'correlations'):
            self.correlation_analysis()
        
        fig, axes = plt.subplots(1, len(self.years) + 1, figsize=(20, 5))
        
        # Correlations between bands by year
        for i, year in enumerate(self.years):
            ax = axes[i]
            corr_matrix = self.correlations[f'year_{year}']
            sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                       center=0, ax=ax, square=True, cbar_kws={'shrink': 0.8})
            ax.set_title(f'Band Correlation\n{year}')
        
        # Mean temporal correlation
        ax = axes[-1]
        temporal_corr = np.zeros((len(self.bands), len(self.bands)))
        for i, band1 in enumerate(self.bands):
            for j, band2 in enumerate(self.bands):
                if i == j:
                    temporal_corr[i, j] = 1.0
                else:
                    corrs = []
                    for year in self.years:
                        valid_mask = self.get_valid_mask(year)
                        d1 = self.data[year][band1][valid_mask]
                        d2 = self.data[year][band2][valid_mask]
                        min_len = min(len(d1), len(d2))
                        if min_len > 10:
                            corrs.append(np.corrcoef(d1[:min_len], d2[:min_len])[0, 1])
                    temporal_corr[i, j] = np.mean(corrs) if corrs else 0
        
        sns.heatmap(temporal_corr, annot=True, fmt='.2f', cmap='coolwarm',
                   xticklabels=self.bands, yticklabels=self.bands,
                   center=0, ax=ax, square=True)
        ax.set_title('Average Temporal Correlation')
        
        plt.tight_layout()
        if len(self.years)>1:
            fig_name = self.plot_dir + '04_correlations.png'
        else:
            fig_name = self.plot_dir + f'04_correlations_{self.years[0]}.png'
        if self.plot_dir is not None:
            plt.savefig(fig_name, dpi=150, bbox_inches='tight')
            print("Correlation analysis saved")
        plt.show()
    
    def monte_carlo_preparation_optimized(self):
        r"""This method prepare specific analysis for Monte Carlo simulation - OPTIMIZED.
        It uses stratified subsampling (flooded + non-flooded) for speed.
        It calculates global statistics (mean, std, percentiles) for each 
        band and year, and identifies the best-fitting distribution for 
        the positive values using a representative sample.
        
        Returns
        -------
        mc_analysis: dict
            Dictionary containing the best distribution, zero probability, 
            variability, and temporal autocorrelation for each band
        """
        
        mc_analysis = {}
        
        for band in self.bands:
            # Strategy: calculate global statistics with all data (fast: mean, std, percentiles)
            # but fit distributions with a representative sample (max 50k)
            
            yearly_stats = []
            all_data_sample = []  # Accumulator for global sample
            
            for year in self.years:
                valid_mask = self.get_valid_mask(year)
                data_full = self.data[year][band][valid_mask]
                
                # Exact statistics (fast, O(n))
                yearly_stats.append({
                    'mean': np.mean(data_full),
                    'std': np.std(data_full),
                    'zeros_pct': np.sum(data_full == 0) / len(data_full) * 100,
                    'count': len(data_full)
                })
                
                # STRATIFIED SAMPLING for distribution analysis
                # Separate zeros and positives to preserve the zero-inflated structure
                data_0 = data_full[data_full == 0]
                data_pos = data_full[data_full > 0]
                
                # Sample from each stratum proportionally, max 5000 per year
                max_per_year = 5000
                n_0 = min(len(data_0), int(max_per_year * 0.3))  # 30% 0
                n_pos = min(len(data_pos), int(max_per_year * 0.7))  # 70% positives
                
                if n_0 > 0:
                    sample_0 = np.random.choice(data_0, n_0, replace=False)
                    all_data_sample.extend(sample_0)
                if n_pos > 0:
                    sample_pos = np.random.choice(data_pos, n_pos, replace=False)
                    all_data_sample.extend(sample_pos)
            
            all_data_sample = np.array(all_data_sample)
            
            # Identify best distribution ONLY for positives (>0)
            positive_data = all_data_sample[all_data_sample > 0]
            best_dist = None
            
            if len(positive_data) > 100:
                distributions = [stats.beta, stats.gamma, stats.lognorm, stats.weibull_min, stats.invgauss]
                best_ks = np.inf
                
                for dist in distributions:
                    try:
                        if dist == stats.beta:
                            # Beta requires scaling to [0,1] if the data is not in that range
                            if band == 'flood_frequency':
                                params = dist.fit(positive_data, floc=0, fscale=1)
                                ks_stat, _ = stats.kstest(positive_data, lambda x: dist.cdf(x, *params))
                            else:
                                continue  # Skip Beta for unbounded variables
                        else:
                            params = dist.fit(positive_data, floc=0)
                            ks_stat, _ = stats.kstest(positive_data, lambda x: dist.cdf(x, *params))
                        
                        if ks_stat < best_ks:
                            best_ks = ks_stat
                            best_dist = (dist.name, params)
                    except:
                        continue
            
            # Temporal autocorrelation between years (using annual means, not all data)
            yearly_means = [s['mean'] for s in yearly_stats]
            temporal_autocorr = np.corrcoef(yearly_means[:-1], yearly_means[1:])[0,1] if len(yearly_means) > 1 else 0
            
            mc_analysis[band] = {
                'best_distribution': best_dist,
                'total_samples_estimated': sum(s['count'] for s in yearly_stats),
                'zero_probability': np.mean([s['zeros_pct']/100 for s in yearly_stats]),
                'yearly_variability': {
                    'mean_std': np.std([s['mean'] for s in yearly_stats]),
                    'std_std': np.std([s['std'] for s in yearly_stats])
                },
                'temporal_autocorr': temporal_autocorr,
                'sample_for_fitting': len(all_data_sample)
            }
        
        self.mc_analysis = mc_analysis
        return mc_analysis

    def visualize_monte_carlo_insights_optimized(self):
        r"""Visualizes insigihts for Monte Carlo - OPTIMIZED VERSION.
        It creates and saves a figure with multiple subplots showing the 
        probability of zero values, the interannual variability of means, 
        and the distribution fit for flood frequency.
        
        """
        if not hasattr(self, 'mc_analysis'):
            self.monte_carlo_preparation_optimized()
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Zero probability by band
        ax1 = axes[0, 0]
        bands = list(self.mc_analysis.keys())
        zero_probs = [self.mc_analysis[b]['zero_probability'] for b in bands]
        bars = ax1.bar(range(len(bands)), zero_probs, color='steelblue', edgecolor='black')
        ax1.set_xticks(range(len(bands)))
        ax1.set_xticklabels(bands, rotation=45, ha='right')
        ax1.set_ylabel('Probability')
        ax1.set_title('P(Value = 0) by Band\n(Null Flood)')
        ax1.set_ylim(0, 1)
        for bar, prob in zip(bars, zero_probs):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height, f'{prob:.3f}', 
                    ha='center', va='bottom', fontsize=9)
        
        # 2. Interannual variability of means
        ax2 = axes[0, 1]
        ax2_right = ax2.twinx()  # Right Y-axis

        # Left: flood_frequency (0-0.12 approx)
        subset_freq = self.stats_df[self.stats_df['band'] == 'flood_frequency']
        line1 = ax2.plot(subset_freq['year'], subset_freq['mean'], 
                        marker='o', color='crimson', label='flood_frequency', linewidth=2, markersize=6)
        ax2.set_ylabel('Mean Flood Frequency (0-1)', color='crimson')
        ax2.tick_params(axis='y', labelcolor='crimson')
        ax2.set_ylim(0, 0.12)  # Appropriate scale for frequency

        # Right: duration and consecutive (0-40 approx)
        for band, color in [('flood_duration', 'darkgoldenrod'), ('max_consecutive_full', 'forestgreen')]:
            subset = self.stats_df[self.stats_df['band'] == band]
            line = ax2_right.plot(subset['year'], subset['mean'], 
                                marker='s', color=color, label=band, linewidth=2, markersize=6)

        ax2_right.set_ylabel('Mean (days)', color='black')
        ax2_right.tick_params(axis='y', labelcolor='black')
        ax2_right.set_ylim(0, 40)

        # Combined legend
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_right.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

        ax2.set_xlabel('Year')
        ax2.set_title('Mean temporal evolution')
        ax2.grid(True, alpha=0.3)
        
        """
        # 3. Distribution fit (flood_frequency, only positive values)
        ax3 = axes[1, 0]
        band = 'flood_frequency'
        
        # Collect stratified sample quickly (same method as in preparation)
        all_pos_data = []
        for year in self.years:
            valid_mask = self.get_valid_mask(year)
            data = self.data[year][band][valid_mask]
            pos_data = data[data > 0]
            # Take max 3000 per year for smooth but fast histogram
            if len(pos_data) > 3000:
                pos_data = np.random.choice(pos_data, 3000, replace=False)
            all_pos_data.extend(pos_data)
        
        all_pos_data = np.array(all_pos_data)
        
        ax3.hist(all_pos_data, bins=50, density=True, alpha=0.6, label='Datos (>0)', color='gray')
        
        # Adjust and plot distributions
        x = np.linspace(0.001, 1, 200)
        colors = {'beta': 'red', 'gamma': 'green', 'lognorm': 'blue'}
        
        for dist_name in ['beta', 'gamma', 'lognorm']:
            try:
                if dist_name == 'beta':
                    params = stats.beta.fit(all_pos_data, floc=0, fscale=1)
                    pdf_vals = stats.beta.pdf(x, *params)
                    label = f'Beta (a={params[0]:.2f}, b={params[1]:.2f})'
                elif dist_name == 'gamma':
                    params = stats.gamma.fit(all_pos_data, floc=0)
                    pdf_vals = stats.gamma.pdf(x, *params)
                    label = f'Gamma (a={params[0]:.2f})'
                else:  # lognorm
                    params = stats.lognorm.fit(all_pos_data, floc=0)
                    pdf_vals = stats.lognorm.pdf(x, *params)
                    label = f'Lognormal'
                
                ax3.plot(x, pdf_vals, color=colors[dist_name], linewidth=2, label=label)
            except:
                continue
        
        ax3.set_xlabel('Flood Frequency (0-1)')
        ax3.set_ylabel('Density')
        ax3.set_title('Distribution Fit\n(Only Flooded Pixels)')
        ax3.legend(fontsize=9)
        """
        # 3. Distribution fit (flood_duration, only positive values)
        ax3 = axes[1, 0]
        band = 'flood_duration'

        # Collect stratified sample
        all_pos_data = []
        for year in self.years:
            valid_mask = self.get_valid_mask(year)
            data = self.data[year][band][valid_mask]
            pos_data = data[data > 0]
            # Take max 3000 per year for smooth but fast histogram
            if len(pos_data) > 3000:
                pos_data = np.random.choice(pos_data, 3000, replace=False)
            all_pos_data.extend(pos_data)

        all_pos_data = np.array(all_pos_data)

        # Histogram
        ax3.hist(all_pos_data, bins=50, density=True, alpha=0.6, label='Data (>0)', color='gray', edgecolor='black')

        # Adjust and plot distributions
        x = np.linspace(0.001, np.percentile(all_pos_data, 99), 200)  # Up to 99th percentile, not extreme max
        colors = {
            'gamma': 'green',
            'lognorm': 'blue',
            'weibull': 'orange',
            'invgauss': 'purple'
        }

        for dist_name in ['gamma', 'lognorm', 'weibull', 'invgauss']:
            try:
                if dist_name == 'gamma':
                    params = stats.gamma.fit(all_pos_data, floc=0)
                    pdf_vals = stats.gamma.pdf(x, *params)
                    label = f'Gamma (a={params[0]:.2f})'
                    color = colors['gamma']
                    
                elif dist_name == 'lognorm':
                    params = stats.lognorm.fit(all_pos_data, floc=0)
                    pdf_vals = stats.lognorm.pdf(x, *params)
                    label = f'Lognormal'
                    color = colors['lognorm']
                    
                elif dist_name == 'weibull':
                    # Weibull: useful for modeling durations (time to failure)
                    params = stats.weibull_min.fit(all_pos_data, floc=0)
                    pdf_vals = stats.weibull_min.pdf(x, *params)
                    label = f'Weibull (c={params[0]:.2f})'
                    color = colors['weibull']
                    
                elif dist_name == 'invgauss':
                    # Inverse Gaussian: for positive data with right tail
                    params = stats.invgauss.fit(all_pos_data, floc=0)
                    pdf_vals = stats.invgauss.pdf(x, *params)
                    label = f'Inv. Gauss'
                    color = colors['invgauss']
                
                ax3.plot(x, pdf_vals, color=color, linewidth=2, label=label)
                
            except Exception as e:
                print(f"Failed to fit {dist_name}: {e}")
                continue

        # ECDF (Empirical CDF) for non-parametric comparison
        #from statsmodels.distributions.empirical_distribution import ECDF
        #ecdf = ECDF(all_pos_data)
        # Plot ECDF on secondary axis or as step (optional, uncomment if desired)
        #ax3_twin = ax3.twinx()
        #ax3_twin.step(ecdf.x, ecdf.y, color='black', linestyle='--', alpha=0.5, label='ECDF')
        #ax3_twin.set_ylabel('ECDF', color='black')

        ax3.set_xlabel('Flood Duration (days)')
        ax3.set_ylabel('Density')
        ax3.set_title('Distribution Fit\n(Only Flooded Pixels)')
        ax3.legend(fontsize=8, loc='upper right')
        ax3.set_xlim(0, np.percentile(all_pos_data, 95))  # Limit X-axis for visualization
        
        # 4. ULTRA-OPTIMIZED VARIOGRAM
        ax4 = axes[1, 1]
        
        # Use only a representative year (the middle one) and a small sample
        year = self.years[len(self.years)//2]
        data = self.data[year]['flood_frequency'].copy()
        valid = self.get_valid_mask(year)
        
        # SPATIAL STRATIFIED SUBSAMPLE: regular grid + random within
        coords = np.argwhere(valid)
        values = data[valid]
        
        # Reduce to a maximum of 1500 points using systematic spatial sampling
        n_points = min(1500, len(coords))
        if len(coords) > n_points:
            # Systematic sampling: take every k-th point to cover the space
            step = len(coords) // n_points
            idx = np.arange(0, len(coords), step)[:n_points]
            coords = coords[idx]
            values = values[idx]
        
        # Compute distance matrix with pdist (optimized C implementation)
        from scipy.spatial.distance import pdist
        
        dists = pdist(coords, metric='euclidean')
        # Semivariance: (z_i - z_j)^2 / 2
        diffs = pdist(values[:, np.newaxis], metric='sqeuclidean') / 2
        
        # Vectorized binning with np.histogram
        max_dist = np.percentile(dists, 60)  # Up to 60th percentile (sufficient for range)
        n_bins = 12
        bin_edges = np.linspace(0, max_dist, n_bins + 1)
        
        # Use np.digitize to assign to bins
        bin_indices = np.digitize(dists, bin_edges) - 1
        
        semivariances = []
        bin_centers = []
        for i in range(n_bins):
            mask = bin_indices == i
            if np.sum(mask) > 5:  # Minimum 5 pairs
                semivariances.append(np.mean(diffs[mask]))
                bin_centers.append((bin_edges[i] + bin_edges[i+1]) / 2)
        
        if len(bin_centers) > 3:
            ax4.scatter(bin_centers, semivariances, s=80, alpha=0.7, c='darkblue', edgecolors='black', zorder=3)
            ax4.plot(bin_centers, semivariances, 'b-', alpha=0.6, linewidth=2, zorder=2)
            
            # Fit simple spherical model (optional, for visualization)
            try:
                from scipy.optimize import curve_fit
                def spherical(h, nugget, sill, range_val):
                    h = np.array(h)
                    result = np.zeros_like(h)
                    mask = h <= range_val
                    result[mask] = nugget + sill * (1.5 * h[mask] / range_val - 0.5 * (h[mask] / range_val)**3)
                    result[~mask] = nugget + sill
                    return result
                
                popt, _ = curve_fit(spherical, bin_centers, semivariances, 
                                p0=[0.01, np.max(semivariances), max_dist/2],
                                maxfev=2000)
                h_fit = np.linspace(0, max_dist, 100)
                ax4.plot(h_fit, spherical(h_fit, *popt), 'r--', linewidth=2, 
                        label=f'Spherical\n(nugget={popt[0]:.3f}, sill={popt[1]:.3f})')
                ax4.legend(fontsize=8)
            except:
                pass
        
        ax4.set_xlabel('Lag Distance (pixels)')
        ax4.set_ylabel('Semivariance γ(h)')
        ax4.set_title(f'Empirical Variogram ({year})\n(n={n_points} points)')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if self.plot_dir is not None:
            plt.savefig(f'{self.plot_dir}/05_monte_carlo_analysis.png', 
                        dpi=150, bbox_inches='tight')
            print("Monte Carlo analysis saved")

        plt.show()
    
    
    def plot_flood_persistence(self):
        r"""Persistence maps. It creates and saves a figure showing the flood
        persistence across years. It answers the question: how many years did 
        each pixel flood? The resulting map highlights areas that are consistently 
        flooded versus those that are rarely or never flooded, providing insights 
        into spatial patterns of flooding over the 14-year period.
        """
        
        # Frequency stack  > 0 (binary) per year
        flood_binary = np.stack([
            (self.data[year]['flood_frequency'] > 0).astype(np.uint8)
            for year in self.years
        ], axis=0)
        
        # Count flooded years per pixel
        persistence = np.sum(flood_binary, axis=0)  # 0-14
        valid_mask = self.get_valid_mask(self.years[0])  # Assume same mask for all years
        
        fig, ax = plt.subplots(figsize=(12, 10))
        im = ax.imshow(np.where(valid_mask, persistence, np.nan), 
                    cmap='YlOrRd', vmin=0, vmax=14)
        plt.colorbar(im, label='Number of Flooded Years (2012-2025)')
        ax.set_title('Spatial Flood Persistence\n(14 = always flooded, 0 = never flooded)')
        plt.tight_layout()
        if self.plot_dir is not None:
            plt.savefig(f'{self.plot_dir}/06_flood_persistence.png', 
                    dpi=150, bbox_inches='tight')
        return fig
    
    def plot_hf_on_persistence(self, hf_csv_path):
        r"""This method plots health facilities (HF) on top of the flood 
        persistence map. It filters out HF that are outside the valid map 
        area. This allows us to visualize the relationship between HF 
        locations and flood persistence.
        
        Returns
        -------
        fig: matplotlib.figure.Figure
            The figure object containing the plot.
        hf_valid_df: pandas.DataFrame
            DataFrame containing the valid health facilities with their 
            persistence values.
        """
        
        # Load HF data
        hf_df = pd.read_csv(hf_csv_path)
        
        # Calculate persistence
        flood_binary = np.stack([
            (self.data[year]['flood_frequency'] > 0).astype(np.uint8)
            for year in self.years
        ], axis=0)
        persistence = np.sum(flood_binary, axis=0)
        valid_mask = self.get_valid_mask(self.years[0])
        
        # Geotransform
        transform = self.metadata[self.years[0]]['transform']
        
        def coord_to_pixel(lat, lon, transform):
            col, row = ~transform * (lon, lat)
            col, row = int(round(col)), int(round(row))
            return row, col
        
        # Filter: Only HF within the valid map
        hf_valid = []
        
        for idx, row in hf_df.iterrows():
            r, c = coord_to_pixel(row['latitude'], row['longitude'], transform)
            row_min_clip = max(r, 0)
            col_min_clip = max(c, 0)
            row_max_clip = min(r, persistence.shape[0])
            col_max_clip = min(c, persistence.shape[1])
            
            # Verify limits and valid mask
            if (row_min_clip <= row_max_clip and col_min_clip <= col_max_clip and 
                valid_mask[r, c]):
                
                if 'PHCC' in row['Health Facility Name'].upper() or 'HOSPITAL' in row['Health Facility Name'].upper():
                    buffer = 40
                else:
                    buffer = 60

                persistence_hf_patch = persistence[max(0, r-buffer):min(persistence.shape[0], r+buffer),
                                                   max(0, c-buffer):min(persistence.shape[1], c+buffer)]
                mean_persistence_hf_patch = np.mean(persistence_hf_patch)
                
                hf_valid.append({
                    'hf_id': row['Health Facility Name'],
                    'latitude': row['latitude'],
                    'longitude': row['longitude'],
                    'row': r,
                    'col': c,
                    'persistence_years': int(mean_persistence_hf_patch),
                    'persistence_pct': mean_persistence_hf_patch / len(self.years) * 100
                })
                    
        # Convert to DataFrame
        hf_valid_df = pd.DataFrame(hf_valid)
        
        print(f"Total HF in CSV: {len(hf_df)}")
        print(f"HF within valid map: {len(hf_valid_df)}")
        print(f"HF discarded: {len(hf_df) - len(hf_valid_df)}")
        
        if len(hf_valid_df) == 0:
            print("ERROR: No valid HF to plot. Check coordinates and valid mask.")
            return None, None
        
        # Plot
        fig, ax = plt.subplots(figsize=(14, 10))
        # Flood persistence map
        im = ax.imshow(np.where(valid_mask, persistence, np.nan),
                    cmap='YlOrRd', vmin=0, vmax=len(self.years), alpha=0.7)
        
        # Valid HF (convert lat/lon to indices for scatter in image coordinates)
        # Or use geographic coordinates if the plot allows, but here we use pixel indices for accurate overlay
        rows = hf_valid_df['row'].values
        cols = hf_valid_df['col'].values
        
        scatter = ax.scatter(cols, rows,  # Use pixel indices, not lat/lon
                            c=hf_valid_df['persistence_years'], 
                            cmap='YlOrRd', vmin=0, vmax=len(self.years),
                            edgecolors='black', s=80, zorder=5, marker='o')
        
        plt.colorbar(im, label='Number of Flooded Years')
        plt.colorbar(scatter, label='HF: Flooded Years')
        ax.set_title(f'Health Facilities over Flood Persistence Map\n(n={len(hf_valid_df)} valid HF)')
        plt.tight_layout()
        if self.plot_dir is not None:
            plt.savefig(f'{self.plot_dir}/07_hf_persistence.png', 
                        dpi=150, bbox_inches='tight')
        # Save valid HF data for further analysis
        hf_valid_df.to_csv(f'{self.plot_dir}/hf_persistence_valid.csv', index=False)
        return fig, hf_valid_df
    
    def plot_persistence_variability_analysis(self):
        r"""Joint analysis of persistence vs. variability. It creates and
        saves a figure showing the relationship between flood persistence 
        (number of years flooded) and variability (coefficient of variation 
        of flood duration) for each pixel. The plot identifies critical zones 
        where low persistence coincides with high variability, as well as
        stable zones with high persistence and low variability. This analysis 
        helps to understand the spatial patterns of flood risk and stability 
        across the study area.
        
        Returns
        -------
        fig_a: matplotlib.figure.Figure
            The figure object containing the scatter plot of persistence vs. variability.
        fig_b: matplotlib.figure.Figure
            The figure object containing the bicolor map classification of zones 
            based on persistence and variability.
        critic_area : np.ndarray
            A boolean array indicating the critical zone (low persistence, high 
            variability) for further analysis.
        """
                
        # Compute persistence: number of years with flooding
        flood_binary = np.stack([
            (self.data[year]['flood_duration'] > 0).astype(np.uint8)
            for year in self.years
        ], axis=0)
        persistence = np.sum(flood_binary, axis=0)  # 0-14
        
        # Compute variability: coefficient of variation of duration
        duration_stack = np.stack([
            self.data[year]['flood_duration'] 
            for year in self.years
        ], axis=0)
        
        mean_duration = np.mean(duration_stack, axis=0)
        std_duration = np.std(duration_stack, axis=0)
        
        # CV only where mean > 0 (avoid division by zero)
        cv = np.zeros_like(mean_duration)
        valid_mean = mean_duration > 0
        cv[valid_mean] = std_duration[valid_mean] / mean_duration[valid_mean]
        cv = np.clip(cv, 0, 5)  # Limit extreme outliers
        
        valid_mask = self.get_valid_mask(self.years[0])
        
        # =================================================================
        # OPTION A: SCATTER PLOT (pixel sample)
        # =================================================================
        fig_a, ax_a = plt.subplots(figsize=(10, 8))
        
        # Subsample for visualization (max 5000 pixels)
        y_coords, x_coords = np.where(valid_mask)
        n_pixels = len(y_coords)
        
        if n_pixels > 5000:
            idx = np.random.choice(n_pixels, 5000, replace=False)
            y_sample, x_sample = y_coords[idx], x_coords[idx]
        else:
            y_sample, x_sample = y_coords, x_coords
        
        pers_sample = persistence[y_sample, x_sample]
        cv_sample = cv[y_sample, x_sample]
        
        # Color by density (where there are more pixels)
        scatter = ax_a.scatter(pers_sample, cv_sample, 
                            c=pers_sample, cmap='viridis', 
                            alpha=0.6, s=20, edgecolors='none')
        
        # Zones of interest
        ax_a.axvline(7, color='red', linestyle='--', alpha=0.5, label='Mean persistence')
        ax_a.axhline(1.5, color='red', linestyle='--', alpha=0.5, label='High CV')
        
        # Annotate quadrants
        ax_a.text(2, 4.5, 'LOW persistence\nHIGH variability\n← CRITICAL ZONE', 
                bbox=dict(boxstyle='round,pad=0.5', facecolor='red', alpha=0.3),
                fontsize=10, ha='center', fontweight='bold',verticalalignment='top')
        ax_a.text(12, 4.5, 'HIGH persistence\nHIGH variability\n← UNSTABLE ZONE', 
                bbox=dict(boxstyle='round,pad=0.5', facecolor='orange', alpha=0.3),
                fontsize=10, ha='center', verticalalignment='top')
        ax_a.text(2, 0.5, 'LOW persistence\nLOW variability\n(Safe Zone)', 
                bbox=dict(boxstyle='round,pad=0.5', facecolor='green', alpha=0.3),
                fontsize=10, ha='center', verticalalignment='bottom')
        ax_a.text(12, 0.5, 'HIGH persistence\nLOW variability\n(Chronic Zone)', 
                bbox=dict(boxstyle='round,pad=0.5', facecolor='blue', alpha=0.3),
                fontsize=10, ha='center', verticalalignment='bottom')
        
        ax_a.set_xlabel('Persistence (years flooded out of 14)')
        ax_a.set_ylabel('Variability (Coefficient of Variation σ/μ)')
        ax_a.set_title('Persistence vs. Variability per Pixel')
        ax_a.set_xlim(0, 14)
        ax_a.set_ylim(0, 5)
        ax_a.legend(loc='upper right')
        plt.colorbar(scatter, label='Persistence (years)')
        if self.plot_dir is not None:
            plt.savefig(f'{self.plot_dir}/08_persistence_variability_scatter.png', 
                        dpi=150, bbox_inches='tight')
        
        # =================================================================
        # OPTION B: BICOLOR MAP (spatial classification)
        # =================================================================
        fig_b, ax_b = plt.subplots(figsize=(14, 10))
        
        # Create 4-zone classification based on persistence and variability
        # zone 0 = No data (masked out), 1 = Safe, 2 = Chronic, 3 = Unstable, 4 = CRITICAL
        zona = np.zeros_like(persistence, dtype=np.uint8)
        zona[(persistence <= 6) & (cv < 1.25)] = 1   # Low pers, low var: Safe
        zona[(persistence > 6) & (cv < 1.25)] = 2  # High pers, low var: Chronic
        zona[(persistence >= 6) & (cv >= 1.25)] = 3 # High pers, high var: Unstable
        zona[(persistence < 6) & (cv >= 1.25)] = 4  # Low pers, high var: CRITICAL
        
        # Harmonious colors (soft palette)
        colors = ['#2c3e50',  # No data: dark gray
                "#efeff1",  # Safe: soft mint green
                '#0072c6',  # Chronic: medium turquoise
                "#a3c7e3",  # Unstable: soft orange
                '#fd4f00']  # CRITICAL: terracotta red
        
        cmap = mcolors.ListedColormap(colors)
        im = ax_b.imshow(np.where(valid_mask, zona, 0), 
                        cmap=cmap, vmin=0, vmax=4)
        
        # Manual legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#2c3e50', label='No data'),
            Patch(facecolor='#efeff1', label='Safe (low pers, low var)'),
            Patch(facecolor='#0072c6', label='Chronic (high pers, low var)'),
            Patch(facecolor="#a3c7e3", label='Unstable (high pers, high var)'),
            Patch(facecolor='#fd4f00', label='CRITICAL (low pers, high var)')
        ]
        ax_b.legend(handles=legend_elements, loc='upper left', 
                    bbox_to_anchor=(1.02, 1), frameon=True, 
                    fancybox=True, shadow=True)
        plt.tight_layout()
        plt.subplots_adjust(right=0.75)  # Leave space on the right
        ax_b.set_title('Areas classification: Persistence vs. Variability')
        if self.plot_dir is not None:
            plt.savefig(f'{self.plot_dir}/08_persistence_variability_map.png', 
                        dpi=150, bbox_inches='tight')
        # Save critical zone mask for further analysis
        critic_area = (zona == 4) & valid_mask
        if self.plot_dir is not None:
            np.save(f'{self.plot_dir}/08_zona_critica_mask.npy', critic_area)
        print(f"CRITICAL pixels in CRITICAL zone: {np.sum(critic_area)} ({np.sum(critic_area)/np.sum(valid_mask)*100:.1f}%)")
        return fig_a, fig_b, critic_area
    
    def plot_hf_on_persistence_variability(self, hf_csv_path):
        r"""This method plots health facilities (HF) on top of the combined 
        map of persistence vs. variability. Each HF is colored according to 
        its zone classification.
        """
        
        # Compute persistence and variability (same code as before)
        flood_binary = np.stack([
            (self.data[year]['flood_duration'] > 0).astype(np.uint8)
            for year in self.years
        ], axis=0)
        persistence = np.sum(flood_binary, axis=0)
        
        duration_stack = np.stack([
            self.data[year]['flood_duration'] 
            for year in self.years
        ], axis=0)
        mean_duration = np.mean(duration_stack, axis=0)
        std_duration = np.std(duration_stack, axis=0)
        
        cv = np.zeros_like(mean_duration)
        valid_mean = mean_duration > 0
        cv[valid_mean] = std_duration[valid_mean] / mean_duration[valid_mean]
        cv = np.clip(cv, 0, 5)
        
        valid_mask = self.get_valid_mask(self.years[0])
        transform = self.metadata[self.years[0]]['transform']
        
        # Create zone classification (same code as before)
        zona = np.zeros_like(persistence, dtype=np.uint8)
        zona[(persistence <= 6) & (cv < 1.25)] = 1   # Safe
        zona[(persistence > 6) & (cv < 1.25)] = 2  # Chronic
        zona[(persistence > 6) & (cv >= 1.25)] = 3 # Unstable
        zona[(persistence <= 6) & (cv >= 1.25)] = 4  # CRITICAL
        
        # Harmonized colors
        colors_map = ['#2c3e50',  # No data: dark gray
                "#efeff1",  # Safe: soft mint green
                '#0072c6',  # Chronic: medium turquoise
                "#a3c7e3",  # Unstable: soft orange
                '#fd4f00']  # Critical: terracotta red
        cmap = mcolors.ListedColormap(colors_map)
        
        # Load HF
        hf_df = pd.read_csv(hf_csv_path)
        
        def coord_to_pixel(lat, lon, transform):
            col, row = ~transform * (lon, lat)
            col, row = int(round(col)), int(round(row))
            return row, col
        
        # Filter valid HF and classify
        hf_valid = []
        
        for idx, row in hf_df.iterrows():
            r, c = coord_to_pixel(row['latitude'], row['longitude'], transform)
            row_min_clip = max(r, 0)
            col_min_clip = max(c, 0)
            row_max_clip = min(r, persistence.shape[0])
            col_max_clip = min(c, persistence.shape[1])
            
            # Check bounds and valid mask
            if (row_min_clip <= row_max_clip and col_min_clip <= col_max_clip and 
                valid_mask[r, c]):
                
                zona_hf = int(zona[r, c])
                if 'PHCC' in row['Health Facility Name'].upper() or 'HOSPITAL' in row['Health Facility Name'].upper():
                    buffer = 40
                else:
                    buffer = 60
                
                persistence_hf_patch = persistence[max(0, r-buffer):min(persistence.shape[0], r+buffer),
                                                   max(0, c-buffer):min(persistence.shape[1], c+buffer)]
                mean_persistence_hf_patch = np.mean(persistence_hf_patch)
                cv_hf_patch = cv[max(0, r-buffer):min(cv.shape[0], r+buffer),
                                 max(0, c-buffer):min(cv.shape[1], c+buffer)]
                mean_cv_hf_patch = np.mean(cv_hf_patch)
                zona_hf_patch = zona[max(0, r-buffer):min(zona.shape[0], r+buffer),
                                     max(0, c-buffer):min(zona.shape[1], c+buffer)]
                mean_zona_hf_patch = np.round(np.mean(zona_hf_patch)).astype(int)
                
                hf_valid.append({
                    'hf_id': row['Health Facility Name'],
                    'lat': row['latitude'],
                    'lon': row['longitude'],
                    'row': r,
                    'col': c,
                    'zona': mean_zona_hf_patch,
                    'zona_nombre': ['No data', 'Safe', 'Chronic', 'Unstable', 'CRITICAL'][mean_zona_hf_patch],
                    'persistence': mean_persistence_hf_patch,
                    'cv': mean_cv_hf_patch,
                    'color': colors_map[mean_zona_hf_patch]
                })
        
        hf_valid_df = pd.DataFrame(hf_valid)
        
        if len(hf_valid_df) == 0:
            print("ERROR: No valid HF to plot. Check coordinates and valid mask.")
            return None, None
        
        print(f"\nTotal HF: {len(hf_df)}")
        print(f"Valid HF: {len(hf_valid_df)}")
        print(f"Discarded HF: {len(hf_df) - len(hf_valid_df)}")
        print(f"\nDistribution by zone:")
        print(hf_valid_df['zona_nombre'].value_counts())
        
        # Plot
        fig, ax = plt.subplots(figsize=(14, 10))
        # Background map
        im = ax.imshow(np.where(valid_mask, zona, 0), 
                    cmap=cmap, vmin=0, vmax=4, alpha=0.6)
        
        # HF with colors according to zone
        for zona_id in range(1, 5):
            subset = hf_valid_df[hf_valid_df['zona'] == zona_id]
            if len(subset) > 0:
                ax.scatter(subset['col'], subset['row'],
                        c=colors_map[zona_id],
                        edgecolors='black',
                        s=120,
                        linewidths=1.5,
                        zorder=5,
                        label=f"{subset['zona_nombre'].iloc[0]} (n={len(subset)})")
        
        # Legend
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1),
                title='Health Facilities',
                frameon=True, fancybox=True, shadow=True)
        
        plt.tight_layout()
        plt.subplots_adjust(right=0.75)
        ax.set_title(f'Health Facilities over Map of Zones\n(n={len(hf_valid_df)} valid HF)', 
                    pad=20, fontsize=12)
        if self.plot_dir is not None:
            plt.savefig(f'{self.plot_dir}/09_hf_persistence_variability.png', 
                        dpi=150, bbox_inches='tight')
        
        # Save
        hf_valid_df.to_csv(f'{self.plot_dir}/hf_areas_persistence_variability.csv', 
                        index=False)
        
        return fig, hf_valid_df
    
    def generate_report(self):
        r"""Generates a complete EDA report."""
        
        print("=" * 80)
        print("EXPLORATORY DATA ANALYSIS - FLOOD MAPS")
        print("=" * 80)
        
        # 1. Basic Statistics
        print("\n1. DESCRIPTIVE STATISTICS")
        print("-" * 40)
        if not hasattr(self, 'stats_df'):
            self.basic_statistics()
        print(self.stats_df.round(3).to_string())
        
        # 2. Temporal Trends
        print("\n2. TEMPORAL TRENDS")
        print("-" * 40)
        if not hasattr(self, 'trends'):
            self.temporal_trends_analysis()
        for band, trend in self.trends.items():
            print(f"\n{band}:")
            print(f"  Slope: {trend['slope']:.6f} units/year")
            print(f"  R²: {trend['r_squared']:.3f}")
            print(f"  p-value: {trend['p_value']:.3f}")
            print(f"  Direction: {trend['trend']}")
        
        # 3. Data Quality
        print("\n3. DATA QUALITY ASSESSMENT")
        print("-" * 40)
        if not hasattr(self, 'quality'):
            self.data_quality_assessment()
        for year, q in self.quality.items():
            print(f"\nYear {year}:")
            print(f"  Valid coverage: {q['coverage_pct']:.1f}%")
            print(f"  Average completeness: {q['completeness']:.1f}%")
            print(f"  Duration inconsistencies: {q['duration_inconsistencies']}")
            print(f"  Consecutive inconsistencies: {q['consecutive_inconsistencies']}")
            print(f"  Zero frequency inconsistencies: {q['zero_freq_inconsistencies']}")
        
        # 4. Monte Carlo Preparation
        print("\n4. INSIGHTS FOR MONTE CARLO SIMULATION")
        print("-" * 40)
        if not hasattr(self, 'mc_analysis'):
            self.monte_carlo_preparation()
        for band, analysis in self.mc_analysis.items():
            print(f"\n{band}:")
            print(f"  Zero probability: {analysis['zero_probability']:.3f}")
            print(f"  Best distribution: {analysis['best_distribution'][0] if analysis['best_distribution'] else 'N/A'}")
            print(f"  Yearly variability (mean): {analysis['yearly_variability'].get('mean', 'N/A')}")
            print(f"  Temporal autocorrelation: {analysis['temporal_autocorr']:.3f}")
        
        return {
            'statistics': self.stats_df,
            'trends': self.trends,
            'quality': self.quality,
            'monte_carlo': self.mc_analysis
        }


# =============================================================================
# ANALYSIS EXECUTION
# =============================================================================

# To use with real data, uncomment and modify the file paths and run the analysis:
file_paths = {
    2012: '../data/hazard_maps/annual_3/floods_annual_2012.tif',
    2013: '../data/hazard_maps/annual_3/floods_annual_2013.tif',
    2014: '../data/hazard_maps/annual_3/floods_annual_2014.tif',
    2015: '../data/hazard_maps/annual_3/floods_annual_2015.tif',
    2016: '../data/hazard_maps/annual_3/floods_annual_2016.tif',
    2017: '../data/hazard_maps/annual_3/floods_annual_2017.tif',
    2018: '../data/hazard_maps/annual_3/floods_annual_2018.tif',
    2019: '../data/hazard_maps/annual_3/floods_annual_2019.tif',
    2020: '../data/hazard_maps/annual_3/floods_annual_2020.tif',
    2021: '../data/hazard_maps/annual_3/floods_annual_2021.tif',
    2022: '../data/hazard_maps/annual_3/floods_annual_2022.tif',
    2023: '../data/hazard_maps/annual_3/floods_annual_2023.tif',
    2024: '../data/hazard_maps/annual_3/floods_annual_2024.tif',
    2025: '../data/hazard_maps/annual_3/floods_annual_2025.tif'
}

plot_dir = "../data/EDA/"
os.makedirs(plot_dir, exist_ok=True)
hf_csv_list = "../data/valid_facilities.csv"
eda = FloodMapEDA(file_paths, plot_dir=plot_dir)

# Execute full analysis
eda.load_data()
eda.visualize_basic_maps()
eda.visualize_statistics()
eda.visualize_spatial_analysis()
eda.visualize_monte_carlo_insights_optimized()
eda.plot_flood_persistence()
eda.plot_hf_on_persistence(hf_csv_list)
eda.plot_persistence_variability_analysis()
eda.plot_hf_on_persistence_variability(hf_csv_list)
results = eda.generate_report()