import numpy as np
import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Configuración de estilo
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class FloodMapEDA:
    """
    Clase para análisis exploratorio de datos de mapas de inundación anuales.
    """
    
    def __init__(self, file_paths=None):
        """
        Inicializa el analizador.
        
        Args:
            file_paths: Dict {año: ruta_al_archivo} o None para usar datos sintéticos
        """
        self.file_paths = file_paths
        self.data = {}  # Almacenará los datos por año
        self.metadata = {}
        self.years = []
        self.bands = ['flood_frequency', 'flood_duration', 'count_valid_full', 'max_consecutive_full']
        self.nodata_value = -9999
        
    def load_data(self):
        """Carga los datos desde archivos raster."""
        for year, path in self.file_paths.items():
            with rasterio.open(path) as src:
                # Leer todas las bandas
                self.data[year] = {
                    'flood_frequency': src.read(1),
                    'flood_duration': src.read(2),
                    'count_valid_full': src.read(3),
                    'max_consecutive_full': src.read(4)
                }
                # Extraer metadata
                self.metadata[year] = {
                    'OBSERVED_DAYS': src.tags().get('OBSERVED_DAYS', 365),
                    'crs': src.crs,
                    'transform': src.transform,
                    'bounds': src.bounds
                }
        
        self.years = sorted(self.data.keys())
        print(f"Datos cargados para años: {self.years}")
    
    def get_valid_mask(self, year):
        """Obtiene máscara de píxeles válidos (no no-data)."""
        freq = self.data[year]['flood_frequency']
        return freq != self.nodata_value
    
    def get_flooded_mask(self, year, threshold=0):
        """Obtiene máscara de píxeles que han sido inundados."""
        valid = self.get_valid_mask(year)
        freq = self.data[year]['flood_frequency']
        return valid & (freq > threshold)
    
    def basic_statistics(self):
        """Calcula estadísticas descriptivas básicas por año y banda."""
        stats_dict = []
        
        for year in self.years:
            valid_mask = self.get_valid_mask(year)
            
            for band in self.bands:
                # ASEGURAR TIPO FLOAT para cálculos
                data = self.data[year][band][valid_mask].astype(np.float64)
                
                if len(data) == 0:
                    continue
                
                # DEBUG: imprimir info para flood_frequency
                if band == 'flood_frequency':
                    print(f"\nDEBUG {year} - {band}:")
                    print(f"  Total píxeles válidos: {len(data)}")
                    print(f"  Píxeles > 0: {np.sum(data > 0)}")
                    print(f"  Suma total: {np.sum(data)}")
                    print(f"  Media (float64): {np.mean(data)}")
                    print(f"  Max: {np.max(data)}")
                    print(f"  Tipo original: {self.data[year][band].dtype}")
                
                # Estadísticas básicas
                stat = {
                    'year': year,
                    'band': band,
                    'count': len(data),
                    'mean': float(np.mean(data)),  # ASEGURAR QUE ES FLOAT PYTHON
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
        
        # DEBUG: mostrar DataFrame resultante
        print("\nDEBUG DataFrame medias flood_frequency:")
        print(self.stats_df[self.stats_df['band'] == 'flood_frequency'][['year', 'mean', 'count', 'zeros_pct']])
        
        return self.stats_df
    
    """
    def basic_statistics(self):
        #Calcula estadísticas descriptivas básicas por año y banda.
        stats_dict = []
        
        for year in self.years:
            valid_mask = self.get_valid_mask(year)
            
            for band in self.bands:
                data = self.data[year][band][valid_mask]
                
                if len(data) == 0:
                    continue
                    
                # Estadísticas básicas
                stat = {
                    'year': year,
                    'band': band,
                    'count': len(data),
                    'mean': np.mean(data),
                    'std': np.std(data),
                    'min': np.min(data),
                    'max': np.max(data),
                    'median': np.median(data),
                    'q25': np.percentile(data, 25),
                    'q75': np.percentile(data, 75),
                    'zeros': np.sum(data == 0),
                    'zeros_pct': np.sum(data == 0) / len(data) * 100
                }
                stats_dict.append(stat)
        
        self.stats_df = pd.DataFrame(stats_dict)
        return self.stats_df
    """
    def temporal_trends_analysis(self):
        """Analiza tendencias temporales en las variables."""
        trends = {}
        
        for band in self.bands:
            yearly_means = []
            for year in self.years:
                valid_mask = self.get_valid_mask(year)
                data = self.data[year][band][valid_mask]
                yearly_means.append(np.mean(data))
            
            # Regresión lineal para tendencia
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
        """Analiza consistencia espacial entre años."""
        # Crear stack de datos para análisis espacial
        consistency_maps = {}
        
        for band in self.bands:
            stack = np.stack([self.data[year][band] for year in self.years])
            valid_mask = np.stack([self.get_valid_mask(year) for year in self.years]).all(axis=0)
            
            # Calcular coeficiente de variación espacial
            mean_map = np.mean(stack, axis=0)
            std_map = np.std(stack, axis=0)
            cv_map = np.zeros_like(mean_map)
            cv_map[valid_mask] = std_map[valid_mask] / (mean_map[valid_mask] + 1e-10)
            
            # Detectar píxeles con cambios abruptos
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
        """Analiza correlaciones entre bandas y entre años."""
        correlations = {}
        
        # Correlación entre bandas para cada año
        for year in self.years:
            valid_mask = self.get_valid_mask(year)
            df_year = pd.DataFrame({
                band: self.data[year][band][valid_mask] 
                for band in self.bands
            })
            correlations[f'year_{year}'] = df_year.corr()
        
        # Correlación temporal para cada banda
        temporal_corr = {}
        for band in self.bands:
            band_data = {year: self.data[year][band][self.get_valid_mask(year)] for year in self.years}
            # Usar intersección de píxeles válidos
            min_len = min(len(v) for v in band_data.values())
            band_matrix = np.array([v[:min_len] for v in band_data.values()])
            temporal_corr[band] = np.corrcoef(band_matrix)
        
        correlations['temporal'] = temporal_corr
        self.correlations = correlations
        return correlations
    
    def data_quality_assessment(self):
        """Evalúa la calidad de los datos."""
        quality_report = {}
        
        for year in self.years:
            valid_mask = self.get_valid_mask(year)
            total_pixels = valid_mask.size
            valid_pixels = np.sum(valid_mask)
            
            # CORRECCIÓN: Convertir OBSERVED_DAYS a int, con manejo de errores
            obs_days_raw = self.metadata[year].get('OBSERVED_DAYS', 365)
            try:
                obs_days = int(obs_days_raw)
            except (ValueError, TypeError):
                # Si no se puede convertir, usar valor por defecto y advertir
                print(f"Advertencia: OBSERVED_DAYS no válido para {year}: '{obs_days_raw}'. Usando 365.")
                obs_days = 365
            
            count_valid = self.data[year]['count_valid_full'][valid_mask]
            
            # Píxeles donde count_valid > OBSERVED_DAYS (error)
            invalid_count = np.sum(count_valid > obs_days)
            
            # Verificar relaciones lógicas entre bandas
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
                
                # Caso específico: freq=0 pero consecutivos>0
                mask_problem = (freq == 0) & (max_consec > 0)
                print(f"  freq=0 & consecutivos>0: {np.sum(mask_problem)}")
                if np.sum(mask_problem) > 0:
                    print(f"  Valores de consecutivos en estos casos: {np.unique(max_consec[mask_problem])}")
            
            # Inconsistencias: duración > frecuencia * observed_days
            # CORRECCIÓN: Asegurar que obs_days sea float para la multiplicación
            obs_days_float = float(obs_days)
            inconsistent_dur = np.sum(dur > (freq * obs_days_float + 1e-5))
            
            # Inconsistencias: max_consecutive > duration
            inconsistent_consec = np.sum(max_consec > (dur + 1e-5))
            
            # Inconsistencias: freq=0 pero dur>0 o consec>0
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
        """Visualización básica de los mapas - una imagen grande con todos los años."""
        n_years = len(self.years)
        n_bands = len(self.bands)
        
        # Tamaño proporcional: 5 pulgadas por año, 4 por banda
        fig, axes = plt.subplots(n_years, n_bands, figsize=(n_bands * 5, n_years * 4))
        
        # Asegurar que axes sea 2D incluso si hay un solo año
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
        fig_name = '/mnt/staas/CLICHE/00_DATA/EDA/annual_floods/01_basic_maps.png'
        plt.savefig(fig_name, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"Mapas básicos guardados: {fig_name} ({n_years}x{n_bands} subplots)")
    
    def visualize_statistics(self):
        """Visualiza estadísticas descriptivas."""
        if not hasattr(self, 'stats_df'):
            self.basic_statistics()
        
        # =============================================================================
        # FIGURA 1: Boxplots detallados por banda (SOLO PIXELES INUNDADOS >0)
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
                
                # FILTRAR: solo píxeles inundados (>0)
                data_inundados = data[data > 0]
                
                print(f"{year} - {band}: {len(data_inundados)} píxeles inundados de {len(data)} válidos ({len(data_inundados)/len(data)*100:.2f}%)")
                
                if len(data_inundados) > 0:
                    if len(data_inundados) > 50000:
                        sample = np.random.choice(data_inundados, 50000, replace=False)
                    else:
                        sample = data_inundados
                    band_data_by_year.append(sample)
                    year_labels.append(str(year))
                else:
                    band_data_by_year.append(np.array([0]))
                    year_labels.append(f"{year}\n(sin inund)")
            
            bp = ax.boxplot(band_data_by_year, patch_artist=True, widths=0.6,
                            showfliers=False)
            
            colors = plt.cm.viridis(np.linspace(0, 1, len(self.years)))
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax.set_xticklabels(year_labels, rotation=45, ha='right', fontsize=9)
            ax.set_title(f'{band}\n(Solo píxeles >0)', fontsize=11, fontweight='bold')
            ax.set_xlabel('Año')
            ax.set_ylabel('Valor')
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('Distribución por Banda - Solo Píxeles Inundados (>0)', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig('/mnt/staas/CLICHE/00_DATA/EDA/annual_floods/02_boxplots_inundados.png', 
                    dpi=150, bbox_inches='tight')
        plt.show()
        print("Boxplots de píxeles inundados guardados")
        
        # =============================================================================
        # FIGURA 2: Resumen estadístico general (2x2 subplots)
        # =============================================================================
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Evolución temporal de medias - DOS EJES Y
        ax1 = axes[0, 0]
        ax1_right = ax1.twinx()  # Eje Y derecho

        # Izquierda: flood_frequency (0-0.12 aprox)
        subset_freq = self.stats_df[self.stats_df['band'] == 'flood_frequency']
        line1 = ax1.plot(subset_freq['year'], subset_freq['mean'], 
                        marker='o', color='crimson', label='flood_frequency', linewidth=2, markersize=6)
        ax1.set_ylabel('Media Flood Frequency (0-1)', color='crimson')
        ax1.tick_params(axis='y', labelcolor='crimson')
        ax1.set_ylim(0, 0.12)  # Escala apropiada para frecuencia

        # Derecha: duration y consecutive (0-40 aprox)
        for band, color in [('flood_duration', 'darkgoldenrod'), ('max_consecutive_full', 'forestgreen')]:
            subset = self.stats_df[self.stats_df['band'] == band]
            line = ax1_right.plot(subset['year'], subset['mean'], 
                                marker='s', color=color, label=band, linewidth=2, markersize=6)

        ax1_right.set_ylabel('Media (días)', color='black')
        ax1_right.tick_params(axis='y', labelcolor='black')
        ax1_right.set_ylim(0, 40)

        # Leyenda combinada
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1_right.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

        ax1.set_xlabel('Año')
        ax1.set_title('Evolución Temporal de Medias')
        ax1.grid(True, alpha=0.3)
                
        # 2. Extensión espacial de inundación (count de píxeles >0)
        ax2 = axes[0, 1]
        counts = []
        for year in self.years:
            valid_mask = self.get_valid_mask(year)
            data = self.data[year]['flood_frequency'][valid_mask]
            counts.append(np.sum(data > 0))
        
        ax2.bar(self.years, counts, color='steelblue', edgecolor='black')
        ax2.set_xlabel('Año')
        ax2.set_ylabel('Número de píxeles inundados')
        ax2.set_title('Extensión espacial de inundación\n(píxeles >0)')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 3. Porcentaje de ceros (no inundado)
        ax3 = axes[1, 0]
        zero_pivot = self.stats_df.pivot(index='year', columns='band', values='zeros_pct')
        zero_pivot.plot(kind='bar', ax=ax3, width=0.8)
        ax3.set_title('Porcentaje de Píxeles Nunca Inundados (0%)')
        ax3.set_ylabel('Porcentaje (%)')
        ax3.legend(title='Banda', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax3.tick_params(axis='x', rotation=0)
        
        # 4. Histogramas de frecuencia de inundación
        ax4 = axes[1, 1]
        for year in self.years:
            valid_mask = self.get_valid_mask(year)
            freq_data = self.data[year]['flood_frequency'][valid_mask]
            freq_positive = freq_data[freq_data > 0]
            if len(freq_positive) > 0:
                ax4.hist(freq_positive, bins=30, alpha=0.5, label=f'{year}', density=True)
        ax4.set_xlabel('Flood Frequency (0-1)')
        ax4.set_ylabel('Densidad')
        ax4.set_title('Distribución de Frecuencia de Inundación (>0)')
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig('/mnt/staas/CLICHE/00_DATA/EDA/annual_floods/02_statistical_analysis.png', 
                    dpi=150, bbox_inches='tight')
        plt.show()
        print("Análisis estadístico guardado")
    
    def visualize_spatial_analysis(self):
        """Visualiza análisis espacial y consistencia temporal."""
        if not hasattr(self, 'consistency'):
            self.spatial_consistency_analysis()
        
        fig, axes = plt.subplots(2, len(self.bands), figsize=(20, 10))
        
        # Primera fila: Media temporal
        for j, band in enumerate(self.bands):
            ax = axes[0, j]
            mean_map = self.consistency[band]['mean']
            valid = self.consistency[band]['valid_mask']
            display = np.where(valid, mean_map, np.nan)
            
            im = ax.imshow(display, cmap='plasma')
            ax.set_title(f'{band}\nMedia Temporal')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046)
        
        # Segunda fila: Coeficiente de variación
        for j, band in enumerate(self.bands):
            ax = axes[1, j]
            cv_map = self.consistency[band]['cv']
            valid = self.consistency[band]['valid_mask']
            display = np.where(valid, cv_map, np.nan)
            
            im = ax.imshow(display, cmap='RdYlBu_r', vmin=0, vmax=2)
            ax.set_title(f'{band}\nCoef. Variación (σ/μ)')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046)
        
        plt.tight_layout()
        if len(self.years)>1:
            fig_name = '/mnt/staas/CLICHE/00_DATA/EDA/annual_floods/03_spatial_consistency.png'
        else:
            fig_name = f'/mnt/staas/CLICHE/00_DATA/EDA/annual_floods/03_spatial_consistency_{self.years[0]}.png'
        plt.savefig(fig_name, dpi=150, bbox_inches='tight')
        plt.show()
        print("Análisis espacial guardado")
    
    def visualize_correlations(self):
        """Visualiza matrices de correlación."""
        if not hasattr(self, 'correlations'):
            self.correlation_analysis()
        
        fig, axes = plt.subplots(1, len(self.years) + 1, figsize=(20, 5))
        
        # Correlaciones entre bandas por año
        for i, year in enumerate(self.years):
            ax = axes[i]
            corr_matrix = self.correlations[f'year_{year}']
            sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                       center=0, ax=ax, square=True, cbar_kws={'shrink': 0.8})
            ax.set_title(f'Correlación Bandas\n{year}')
        
        # Correlación temporal promedio
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
        ax.set_title('Correlación Promedio\nTemporal')
        
        plt.tight_layout()
        if len(self.years)>1:
            fig_name = '/mnt/staas/CLICHE/00_DATA/EDA/annual_floods/04_correlations.png'
        else:
            fig_name = f'/mnt/staas/CLICHE/00_DATA/EDA/annual_floods/04_correlations_{self.years[0]}.png'
        plt.savefig(fig_name, dpi=150, bbox_inches='tight')
        plt.show()
        print("Análisis de correlaciones guardado")
    
    def monte_carlo_preparation_optimized(self):
        """
        Prepara análisis específicos para diseño de simulación Monte Carlo - OPTIMIZADO.
        Usa subsampling estratificado (inundados + no inundados) para velocidad.
        """
        mc_analysis = {}
        
        for band in self.bands:
            # Estrategia: calcular estadísticas globales con todos los datos (rápido: mean, std, percentiles)
            # pero ajuste de distribuciones con muestra representativa (máx 50k)
            
            yearly_stats = []
            all_data_sample = []  # Acumulador para muestra global
            
            for year in self.years:
                valid_mask = self.get_valid_mask(year)
                data_full = self.data[year][band][valid_mask]
                
                # Estadísticas exactas (rápidas, O(n))
                yearly_stats.append({
                    'mean': np.mean(data_full),
                    'std': np.std(data_full),
                    'zeros_pct': np.sum(data_full == 0) / len(data_full) * 100,
                    'count': len(data_full)
                })
                
                # MUESTRA ESTRATIFICADA para análisis de distribución
                # Separar ceros y positivos para no perder la estructura zero-inflated
                data_0 = data_full[data_full == 0]
                data_pos = data_full[data_full > 0]
                
                # Muestrear de cada estrato proporcionalmente, máx 5000 por año
                max_per_year = 5000
                n_0 = min(len(data_0), int(max_per_year * 0.3))  # 30% ceros
                n_pos = min(len(data_pos), int(max_per_year * 0.7))  # 70% positivos
                
                if n_0 > 0:
                    sample_0 = np.random.choice(data_0, n_0, replace=False)
                    all_data_sample.extend(sample_0)
                if n_pos > 0:
                    sample_pos = np.random.choice(data_pos, n_pos, replace=False)
                    all_data_sample.extend(sample_pos)
            
            all_data_sample = np.array(all_data_sample)
            
            # Identificar mejor distribución SOLO en los positivos (>0)
            positive_data = all_data_sample[all_data_sample > 0]
            best_dist = None
            
            if len(positive_data) > 100:
                distributions = [stats.beta, stats.gamma, stats.lognorm]
                best_ks = np.inf
                
                for dist in distributions:
                    try:
                        if dist == stats.beta:
                            # Beta requiere escalar a [0,1] si los datos no están en ese rango
                            if band == 'flood_frequency':
                                params = dist.fit(positive_data, floc=0, fscale=1)
                                ks_stat, _ = stats.kstest(positive_data, lambda x: dist.cdf(x, *params))
                            else:
                                continue  # Saltar Beta para variables no acotadas
                        else:
                            params = dist.fit(positive_data, floc=0)
                            ks_stat, _ = stats.kstest(positive_data, lambda x: dist.cdf(x, *params))
                        
                        if ks_stat < best_ks:
                            best_ks = ks_stat
                            best_dist = (dist.name, params)
                    except:
                        continue
            
            # Autocorrelación temporal entre años (usando medias anuales, no todos los datos)
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
        """
        Visualiza insights para Monte Carlo - VERSIÓN OPTIMIZADA.
        Sustituye completamente la función anterior.
        """
        if not hasattr(self, 'mc_analysis'):
            self.monte_carlo_preparation_optimized()
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Probabilidad de cero por banda
        ax1 = axes[0, 0]
        bands = list(self.mc_analysis.keys())
        zero_probs = [self.mc_analysis[b]['zero_probability'] for b in bands]
        bars = ax1.bar(range(len(bands)), zero_probs, color='steelblue', edgecolor='black')
        ax1.set_xticks(range(len(bands)))
        ax1.set_xticklabels(bands, rotation=45, ha='right')
        ax1.set_ylabel('Probabilidad')
        ax1.set_title('P(Valor = 0) por Banda\n(Inundación nula)')
        ax1.set_ylim(0, 1)
        for bar, prob in zip(bars, zero_probs):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height, f'{prob:.3f}', 
                    ha='center', va='bottom', fontsize=9)
        
        # 2. Variabilidad interanual de medias
        ax2 = axes[0, 1]
        ax2_right = ax2.twinx()  # Eje Y derecho

        # Izquierda: flood_frequency (0-0.12 aprox)
        subset_freq = self.stats_df[self.stats_df['band'] == 'flood_frequency']
        line1 = ax2.plot(subset_freq['year'], subset_freq['mean'], 
                        marker='o', color='crimson', label='flood_frequency', linewidth=2, markersize=6)
        ax2.set_ylabel('Media Flood Frequency (0-1)', color='crimson')
        ax2.tick_params(axis='y', labelcolor='crimson')
        ax2.set_ylim(0, 0.12)  # Escala apropiada para frecuencia

        # Derecha: duration y consecutive (0-40 aprox)
        for band, color in [('flood_duration', 'darkgoldenrod'), ('max_consecutive_full', 'forestgreen')]:
            subset = self.stats_df[self.stats_df['band'] == band]
            line = ax2_right.plot(subset['year'], subset['mean'], 
                                marker='s', color=color, label=band, linewidth=2, markersize=6)

        ax2_right.set_ylabel('Mean (days)', color='black')
        ax2_right.tick_params(axis='y', labelcolor='black')
        ax2_right.set_ylim(0, 40)

        # Leyenda combinada
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_right.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

        ax2.set_xlabel('Year')
        ax2.set_title('Mean temporal evolution')
        ax2.grid(True, alpha=0.3)
        
        
        # 3. Ajuste de distribuciones (flood_frequency, solo positivos)
        ax3 = axes[1, 0]
        band = 'flood_frequency'
        
        # Recolectar muestra estratificada rápidamente (mismo método que en preparation)
        all_pos_data = []
        for year in self.years:
            valid_mask = self.get_valid_mask(year)
            data = self.data[year][band][valid_mask]
            pos_data = data[data > 0]
            # Tomar máx 3000 por año para histograma suave pero rápido
            if len(pos_data) > 3000:
                pos_data = np.random.choice(pos_data, 3000, replace=False)
            all_pos_data.extend(pos_data)
        
        all_pos_data = np.array(all_pos_data)
        
        ax3.hist(all_pos_data, bins=50, density=True, alpha=0.6, label='Datos (>0)', color='gray')
        
        # Ajustar y plotear distribuciones
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
        
        # 4. VARIOGRAMA ULTRA-OPTIMIZADO
        ax4 = axes[1, 1]
        
        # Usar solo un año representativo (el del medio) y muestra pequeña
        year = self.years[len(self.years)//2]
        data = self.data[year]['flood_frequency'].copy()
        valid = self.get_valid_mask(year)
        
        # SUBSAMPLE ESTRATIFICADO ESPACIAL: grid regular + aleatorio dentro
        coords = np.argwhere(valid)
        values = data[valid]
        
        # Reducir a máximo 1500 puntos usando muestreo espacial sistemático
        n_points = min(1500, len(coords))
        if len(coords) > n_points:
            # Muestreo sistemático: tomar cada k-ésimo punto para cubrir el espacio
            step = len(coords) // n_points
            idx = np.arange(0, len(coords), step)[:n_points]
            coords = coords[idx]
            values = values[idx]
        
        # Calcular matriz de distancias con pdist (C optimizado)
        from scipy.spatial.distance import pdist
        
        dists = pdist(coords, metric='euclidean')
        # Semivarianza: (z_i - z_j)^2 / 2
        diffs = pdist(values[:, np.newaxis], metric='sqeuclidean') / 2
        
        # Binning vectorizado con np.histogram
        max_dist = np.percentile(dists, 60)  # Hasta percentil 60 (suficiente para rango)
        n_bins = 12
        bin_edges = np.linspace(0, max_dist, n_bins + 1)
        
        # Usar np.digitize para asignar a bins
        bin_indices = np.digitize(dists, bin_edges) - 1
        
        semivariances = []
        bin_centers = []
        for i in range(n_bins):
            mask = bin_indices == i
            if np.sum(mask) > 5:  # Mínimo 5 pares
                semivariances.append(np.mean(diffs[mask]))
                bin_centers.append((bin_edges[i] + bin_edges[i+1]) / 2)
        
        if len(bin_centers) > 3:
            ax4.scatter(bin_centers, semivariances, s=80, alpha=0.7, c='darkblue', edgecolors='black', zorder=3)
            ax4.plot(bin_centers, semivariances, 'b-', alpha=0.6, linewidth=2, zorder=2)
            
            # Ajustar modelo esférico simple (opcional, para visualización)
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
                        label=f'Esférico\n(nugget={popt[0]:.3f}, sill={popt[1]:.3f})')
                ax4.legend(fontsize=8)
            except:
                pass
        
        ax4.set_xlabel('Distancia lag (píxeles)')
        ax4.set_ylabel('Semivarianza γ(h)')
        ax4.set_title(f'Variograma Empírico ({year})\n(n={n_points} puntos)')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/mnt/staas/CLICHE/00_DATA/EDA/annual_floods/05_monte_carlo_analysis.png', 
                    dpi=150, bbox_inches='tight')
        plt.show()
        print("Análisis Monte Carlo optimizado guardado")
    
    def generate_report(self):
        """Genera un reporte completo del EDA."""
        print("=" * 80)
        print("EXPLORATORY DATA ANALYSIS - MAPAS DE INUNDACIÓN")
        print("=" * 80)
        
        # 1. Estadísticas Básicas
        print("\n1. ESTADÍSTICAS DESCRIPTIVAS")
        print("-" * 40)
        if not hasattr(self, 'stats_df'):
            self.basic_statistics()
        print(self.stats_df.round(3).to_string())
        
        # 2. Tendencias Temporales
        print("\n2. TENDENCIAS TEMPORALES")
        print("-" * 40)
        if not hasattr(self, 'trends'):
            self.temporal_trends_analysis()
        for band, trend in self.trends.items():
            print(f"\n{band}:")
            print(f"  Pendiente: {trend['slope']:.6f} unidades/año")
            print(f"  R²: {trend['r_squared']:.3f}")
            print(f"  p-valor: {trend['p_value']:.3f}")
            print(f"  Dirección: {trend['trend']}")
        
        # 3. Calidad de Datos
        print("\n3. EVALUACIÓN DE CALIDAD")
        print("-" * 40)
        if not hasattr(self, 'quality'):
            self.data_quality_assessment()
        for year, q in self.quality.items():
            print(f"\nAño {year}:")
            print(f"  Cobertura válida: {q['coverage_pct']:.1f}%")
            print(f"  Completitud media: {q['completeness']:.1f}%")
            print(f"  Inconsistencias duración: {q['duration_inconsistencies']}")
            print(f"  Inconsistencias consecutivos: {q['consecutive_inconsistencies']}")
            print(f"  Inconsistencias freq=0: {q['zero_freq_inconsistencies']}")
        
        # 4. Preparación Monte Carlo
        print("\n4. INSIGHTS PARA SIMULACIÓN MONTE CARLO")
        print("-" * 40)
        if not hasattr(self, 'mc_analysis'):
            self.monte_carlo_preparation()
        for band, analysis in self.mc_analysis.items():
            print(f"\n{band}:")
            print(f"  Probabilidad de cero: {analysis['zero_probability']:.3f}")
            print(f"  Mejor distribución: {analysis['best_distribution'][0] if analysis['best_distribution'] else 'N/A'}")
            print(f"  Variabilidad interanual (mean): {analysis['yearly_variability'].get('mean', 'N/A')}")
            print(f"  Autocorrelación temporal: {analysis['temporal_autocorr']:.3f}")
        
        print("\n" + "=" * 80)
        print("RECOMENDACIONES PARA MONTE CARLO:")
        print("=" * 80)
        print("""
1. MODELADO DE CEROS: La alta probabilidad de píxeles con valor 0 sugiere usar 
   modelos inflados en cero (Zero-Inflated) o dos pasos: (1) probabilidad de 
   inundación, (2) magnitud condicional a inundación.
   
2. DEPENDENCIA TEMPORAL: Las correlaciones entre años deben modelarse con 
   procesos autorregresivos o copulas temporales.
   
3. ESTRUCTURA ESPACIAL: Usar campos aleatorios gaussianos (GRF) o simulación 
   secuencial para mantener la autocorrelación espacial observada.
   
4. VARIABLES DERIVADAS: flood_duration y max_consecutive_full deben generarse 
   condicionalmente a flood_frequency para mantener consistencia física.
   
5. INCERTIDUMBRE METODOLÓGICA: Incorporar la variabilidad en OBSERVED_DAYS 
   como fuente de incertidumbre en la simulación.
        """)
        
        return {
            'statistics': self.stats_df,
            'trends': self.trends,
            'quality': self.quality,
            'monte_carlo': self.mc_analysis
        }


# =============================================================================
# EJECUCIÓN DEL ANÁLISIS
# =============================================================================

# Para usar con datos reales, descomenta y modifica:
file_paths = {
    2012: '/mnt/staas/CLICHE/00_DATA/hazard_maps/annual_daily_new/floods_annual_2012.tif',
    2013: '/mnt/staas/CLICHE/00_DATA/hazard_maps/annual_daily_new/floods_annual_2013.tif',
    2014: '/mnt/staas/CLICHE/00_DATA/hazard_maps/annual_daily_new/floods_annual_2014.tif',
    2015: '/mnt/staas/CLICHE/00_DATA/hazard_maps/annual_daily_new/floods_annual_2015.tif',
    2016: '/mnt/staas/CLICHE/00_DATA/hazard_maps/annual_daily_new/floods_annual_2016.tif',
    2017: '/mnt/staas/CLICHE/00_DATA/hazard_maps/annual_daily_new/floods_annual_2017.tif',
    2018: '/mnt/staas/CLICHE/00_DATA/hazard_maps/annual_daily_new/floods_annual_2018.tif',
    2019: '/mnt/staas/CLICHE/00_DATA/hazard_maps/annual_daily_new/floods_annual_2019.tif',
    2020: '/mnt/staas/CLICHE/00_DATA/hazard_maps/annual_daily_new/floods_annual_2020.tif',
    2021: '/mnt/staas/CLICHE/00_DATA/hazard_maps/annual_daily_new/floods_annual_2021.tif',
    2022: '/mnt/staas/CLICHE/00_DATA/hazard_maps/annual_daily_new/floods_annual_2022.tif',
    2023: '/mnt/staas/CLICHE/00_DATA/hazard_maps/annual_daily_new/floods_annual_2023.tif',
    2024: '/mnt/staas/CLICHE/00_DATA/hazard_maps/annual_daily_new/floods_annual_2024.tif',
    2025: '/mnt/staas/CLICHE/00_DATA/hazard_maps/annual_daily_new/floods_annual_2025.tif'
}
eda = FloodMapEDA(file_paths)

# Ejecutar análisis completo
eda.load_data()
eda.visualize_basic_maps()
eda.visualize_statistics()
eda.visualize_spatial_analysis()
eda.visualize_monte_carlo_insights_optimized()
results = eda.generate_report()