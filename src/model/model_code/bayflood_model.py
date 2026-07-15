# -*- coding: utf-8 -*-

__author__ = "Macarena Merida Floriano, mmef"
__maintainer__ = "Macarena Merida Floriano, mmef"
__email__ = "mmef@gmv.com"

"""
BayFloodGEN Model and scenario generation .

This script contains the implementation of the PatchGENBayGEN class, which defines the
hierarchical Bayesian model for flood occurrence and severity at health facilities in South Sudan.
It also includes the method for generating synthetic scenarios in parallel using joblib, based
on the posterior distribution of the model parameters. The model captures temporal trends,
seasonality, and facility-specific effects, allowing for realistic scenario generation for risk
assessment and decision-making. The scenario generation method is designed to efficiently
produce a large number of synthetic scenarios while ensuring reproducibility and independence
of the generated data.
"""

import os
import warnings

import arviz as az
from joblib import Parallel, delayed, cpu_count
import numpy as np
import pandas as pd
import pymc as pm
from pymc.sampling import jax as pmjax
import pytensor.tensor as pt

from model_code.bayflood_config import (
    INCLUDE_DISTANCE,
    INCLUDE_BREAKPOINT,
    MU_TBREAK_PRIOR_MEAN,
    MU_TBREAK_PRIOR_SD,
    TBREAK_FACILITY_SPREAD_SCALE,
    KAPPA_ALPHA,
    KAPPA_BETA,
    N_CHAINS,
    N_DRAWS,
    N_JOBS_SCENARIOS,
    N_TUNE,
    RANDOM_SEED,
    TARGET_ACCEPT,
    MAX_SAMPLING_CORES,
)
from model_code.bayflood_runtime import select_sampler


# Canonical column order for the final combined output (observed + synthetic).
# Used to keep every batch file's columns aligned when streaming them
# together — see stream_concat_csv.
FINAL_OUTPUT_COLUMNS = [
    'hf_id', 'hf_payam', 'hf_category', 'latitude', 'longitude',
    'buffer_pixels', 'date', 'occurrence', 'pct_flooded', 'min_distance',
    'day_of_year', 'year', 'scenario_id', 'is_synthetic',
]


# ==========================================================
# 2. MODEL
# ==========================================================
class PatchGENBayGEN:
    r"""BayGEN Flood Generator adapted for South Sudan Health Facilities.

    This class implements a hierarchical Bayesian model to simulate flood
    occurrence and severity at health facilities in South Sudan. The model
    captures temporal trends, seasonality, and facility-specific effects,
    allowing us to generate realistic synthetic scenarios for risk assessment
    and decision-making.
    
    The structural-break component (temporal trend modeled as a sigmoid
    transition between a pre- and post-break regime) is optional, controlled
    by the module-level INCLUDE_BREAKPOINT flag, so the same class can be
    applied to regions with a confirmed regime change and to regions with
    none, without forcing break parameters onto data that can't identify
    them.

    Methods
    -------
    - __init__(df): Initializes the model with the expanded dataframe.
    - _build_lags(): Creates lagged features for occurrence and percentage flooded.
    - build_model(): Defines the PyMC model structure.
    - sample(): Runs MCMC sampling to estimate the posterior distribution of
        the model parameters, with automatic backend selection for GPU acceleration.
    - _print_diagnostics(): Prints convergence diagnostics after sampling.

    """

    def __init__(self, df: pd.DataFrame):
        r"""
        Initializes the model with the expanded dataframe.

        Parameters
        ----------
        df : pd.DataFrame
            Expanded dataframe with columns:
            - 'hf_id', 'hf_payam', 'hf_category', 'latitude', 'longitude',
              'buffer_pixels', 'date', 'occurrence', 'pct_flooded', 'min_distance',
              'day_of_year', 'year'

        Arguments
        ---------
        - df_raw: Original expanded dataframe (for reference)
        - df: Dataframe with lags built and NaNs dropped
        - O: Binary occurrence array
        - P: Percentage flooded array (clipped to avoid extremes)
        - D_raw: Raw distance to water array
        - D_scaled: Scaled distance to water (0-1) based on buffer size
        - cos_t, sin_t: Seasonal covariates based on day of year
        - t_year: Scaled year for temporal trend
        - facility_ids, facility_idx: Unique facility IDs and their corresponding indices
        - wet_mask, wet_idx, fac_idx_wet: Masks and indices for flooded days
        - category_ids, cat_idx: Unique category IDs and their corresponding indices
        - fac_cat_idx: Category index for each facility
        - include_breakpoint: Whether the structural-break block is active
          (mirrors the module-level INCLUDE_BREAKPOINT at the time the model
          was built; used by build_model, _print_diagnostics, and
          generate_scenarios so they all stay consistent with each other)
        """

        self.df_raw = df.copy()
        self.df = self._build_lags(df)
        self.df = self.df.dropna(
            subset=['occurrence', 'pct_flooded', 'min_distance', 'O_lag', 'P_lag']
        ).copy().reset_index(drop=True)

        self.O = self.df['occurrence'].values.astype(int)
        eps = 1e-6
        self.P = np.clip(self.df['pct_flooded'].values, eps, 1 - eps)
        self.D_raw = np.maximum(self.df['min_distance'].values, 1e-3)

        D_max = np.where(self.df['buffer_pixels'].values == 40, 2545.5, 3818.0)
        self.D_scaled = self.D_raw / D_max

        t = self.df['day_of_year'].values
        self.cos_t = np.cos(2 * np.pi * t / 365).astype(np.float32)
        self.sin_t = np.sin(2 * np.pi * t / 365).astype(np.float32)

        year_min = self.df['year'].min()
        year_max = self.df['year'].max()
        self.year_min = year_min
        self.year_max = year_max
        self.t_year = np.clip(
            (self.df['year'].values - year_min) / max(year_max - year_min, 1),
            0.0, 1.0
        ).astype(np.float32)

        self.O_lag = self.df['O_lag'].values.astype(np.float32)
        self.P_lag = self.df['P_lag'].values.astype(np.float32)

        self.facility_ids, self.facility_idx = np.unique(
            self.df['hf_id'], return_inverse=True
        )
        self.n_facilities = len(self.facility_ids)

        self.wet_mask = (self.O == 1)
        self.wet_idx = np.where(self.wet_mask)[0]
        self.fac_idx_wet = self.facility_idx[self.wet_idx]
        
        # Mirrors the module-level toggle at construction time; build_model()
        # re-confirms this and generate_scenarios()/_print_diagnostics() read
        # it from here so everything downstream stays consistent even if the
        # global is changed later in the same session.
        self.include_breakpoint = INCLUDE_BREAKPOINT

        print(f"  Facilities : {self.n_facilities}")
        print(f"  Total obs  : {len(self.df)}")
        print(f"  Wet days   : {self.wet_mask.sum()} ({100 * self.wet_mask.mean():.1f}%)")
        print(f"  Year range : {year_min}–{year_max}")
        print(f"  Breakpoint modeling : {'ENABLED' if self.include_breakpoint else 'DISABLED'}")

        # ────────────────────── Category index as a fixed covariate ──────────────────────
        cat_order = ['SAFE', 'CHRONIC', 'UNSTABLE', 'CRITICAL']
        self.category_ids = np.array(cat_order)
        self.n_categories = len(cat_order)
        cat_map = {c: i for i, c in enumerate(cat_order)}

        self.cat_idx = np.array([
            cat_map.get(c, 0) for c in self.df['hf_category'].values
        ], dtype=int)

        self.cat_idx_wet = self.cat_idx[self.wet_idx]

        # Category index per facility (shape: n_facilities)
        self.fac_cat_idx = np.array([
            cat_map.get(
                self.df[self.df['hf_id'] == fid]['hf_category'].iloc[0], 0
            )
            for fid in self.facility_ids
        ], dtype=int)

    @staticmethod
    def _build_lags(df: pd.DataFrame) -> pd.DataFrame:
        r"""This function builds lagged features for occurrence and
        percentage flooded. It groups the data by facility, sorts by date,
        and creates lagged columns 'O_lag' and 'P_lag' for the previous day's
        occurrence and percentage flooded. This is done using a groupby-apply
        pattern, which is efficient for this type of operation.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe with columns:
            - 'hf_id', 'hf_payam', 'hf_category', 'latitude', 'longitude',
              'buffer_pixels', 'date', 'occurrence', 'pct_flooded', 'min_distance',
              'day_of_year', 'year'

        Returns
        -------
        df_lags : pd.DataFrame
            Dataframe with additional columns 'O_lag' and 'P_lag' for lagged features.
        """

        out_parts = []
        # observed=True: hf_id is categorical (see expand_series_to_df) — this
        # avoids pandas iterating over any unused category levels.
        for fid, group in df.groupby('hf_id', sort=False, observed=True):
            g = group.sort_values('date').copy()
            g['O_lag'] = g['occurrence'].shift(1)
            g['P_lag'] = g['pct_flooded'].shift(1)
            out_parts.append(g)
        return pd.concat(out_parts, ignore_index=True).sort_values(
            ['hf_id', 'date']
        ).reset_index(drop=True)

    def build_model(self):
        r"""This method defines the PyMC model structure.
        It includes:
        - Facility-specific random effects to capture unobserved heterogeneity,
            sampled with a non-centered parameterization.
        - An OPTIONAL temporal trend modeled with a sigmoid function to capture
            a structural break between a pre- and post-break regime, controlled
            by INCLUDE_BREAKPOINT. When disabled, no t_break / kappa / delta /
            gamma_cat terms are created at all.
        - Seasonality captured with sine and cosine terms based on the day of year.
        - Lagged effects of previous day's occurrence and percentage flooded.
        - A category effect on the baseline occurrence (always present), and,
            when the breakpoint block is active, a separate category effect on
            the magnitude of the post-break change.
        The model is hierarchical and allows for partial pooling of information
        across facilities, which can improve estimates for facilities with less data.
        The distance to water is optionally included as a separate likelihood if
        INCLUDE_DISTANCE is True.

        Returns
        -------
        model : pm.Model
            The defined PyMC model ready for sampling.
        """

        coords = {"facility": self.facility_ids, "category": self.category_ids}
        # Re-confirm at build time in case INCLUDE_BREAKPOINT was changed
        # after __init__ ran.
        self.include_breakpoint = INCLUDE_BREAKPOINT
        
        with pm.Model(coords=coords) as self.model:
            # ────────────────────── Facility random effects ──────────────────────
            # NON-CENTERED PARAMETERIZATION — WHY THIS MATTERS:
            # The "centered" form, facility_effect ~ Normal(0, sigma_fac), creates
            # a joint posterior between sigma_fac and facility_effect shaped like
            # Neal's funnel: when sigma_fac is small, the sampler is forced into a
            # very narrow region for facility_effect; when sigma_fac is large, the
            # geometry opens back up. NUTS struggles to move efficiently through
            # that funnel regardless of how well-identified kappa/t_break are, so
            # this fix is independent of — and complementary to — the breakpoint
            # changes below. It is worth keeping even when INCLUDE_BREAKPOINT is
            # False, since the funnel comes from the facility hierarchy itself,
            # not from the structural-break block.
            # Non-centering reparameterizes as effect = raw * sigma, with raw ~
            # Normal(0, 1) sampled independently of sigma, removing the funnel.
            sigma_fac = pm.HalfNormal("sigma_fac", 0.5)
            facility_effect_raw = pm.Normal("facility_effect_raw", mu=0, sigma=1, 
                                            dims="facility")
            facility_effect     = pm.Deterministic(
                "facility_effect", facility_effect_raw * sigma_fac, dims="facility"
            )
            
            if self.include_breakpoint:
                # ────────────────────── Structural break (optional) ──────────────────────
                # mu_tbreak / sigma_tbreak here come from MU_TBREAK_PRIOR_MEAN /
                # MU_TBREAK_PRIOR_SD at the top of the script — set these from
                # your own EDA per the comment block above CONFIGURATION.    
                mu_tbreak = pm.Normal("mu_tbreak", mu=MU_TBREAK_PRIOR_MEAN, 
                                      sigma=MU_TBREAK_PRIOR_SD)
                sigma_tbreak = pm.HalfNormal("sigma_tbreak", TBREAK_FACILITY_SPREAD_SCALE)
                # Non-centered per-facility break location (same funnel rationale
                # as facility_effect above).
                t_break_raw = pm.Normal("t_break_raw", mu=0, sigma=1, dims="facility")
                t_break     = pm.Deterministic(
                    "t_break", mu_tbreak + sigma_tbreak * t_break_raw, dims="facility"
                )
                sigma_delta = pm.HalfNormal("sigma_delta", 0.5)
                delta_raw = pm.Normal("delta_raw", mu=0, sigma=1, dims="facility")
                delta       = pm.Deterministic("delta", delta_raw * sigma_delta,
                                               dims="facility")
                # kappa: tight Gamma prior — see the CONFIGURATION comment block
                # for the rationale (keeps it a genuine prior, not a fixed value,
                # while removing the near-flat tail that drove the v2.2 SD blow-up).
                kappa = pm.Gamma("kappa", alpha = KAPPA_ALPHA, beta=KAPPA_BETA)

                transition_all = pm.math.sigmoid(
                    kappa * (self.t_year - t_break[self.facility_idx])
                )
                transition_wet = pm.math.sigmoid(
                    kappa * (self.t_year[self.wet_idx] - t_break[self.fac_idx_wet])
                )
                # Category effect on the MAGNITUDE of the post-break change.
                # Only meaningful if there is a post-break change to begin with,
                # so this is defined only inside the INCLUDE_BREAKPOINT branch.
                gamma_cat = pm.Normal("gamma_cat", mu=0, sigma=0.5, dims="category")
                
            # Priors for fixed effects
            beta0 = pm.Normal("beta0", 0, 1)
            beta_lag = pm.Normal("beta_lag", 0, 1)
            beta_cos = pm.Normal("beta_cos", 0, 0.5)
            beta_sin = pm.Normal("beta_sin", 0, 0.5)

            # Fixed effect of category on the baseline occurrence intercept.
            # Prior centered at 0: the model learns the difference between
            # categories without forcing direction. This is independent of the
            # breakpoint and stays in the model whether or not a break is modeled.
            beta_cat = pm.Normal("beta_cat", mu=0, sigma=1, dims="category")

            # Linear predictor for occurrence with interaction between category and transition
            logit_p = (
                beta0
                + facility_effect[self.facility_idx]
                + beta_cat[self.cat_idx]
                + beta_lag * self.O_lag
                + beta_cos * self.cos_t
                + beta_sin * self.sin_t
            )
            if self.include_breakpoint:
                # Interaction between category and the post-break transition
                logit_p = logit_p + (
                    delta[self.facility_idx] + gamma_cat[self.cat_idx]
                ) * transition_all
            
            # Likelihood for occurrence
            pm.Bernoulli("O_obs", p=pm.math.sigmoid(logit_p), observed=self.O)

            # Linear predictor for percentage flooded on flooded days
            beta0_P = pm.Normal("beta0_P", -1, 1)
            beta_lag_P = pm.Normal("beta_lag_P", 0, 1)
            beta_cos_P = pm.Normal("beta_cos_P", 0, 1)
            beta_sin_P = pm.Normal("beta_sin_P", 0, 1)
            
            mu_P_lin = (
                beta0_P
                + facility_effect[self.fac_idx_wet]
                + beta_lag_P * self.P_lag[self.wet_idx]
                + beta_cos_P * self.cos_t[self.wet_idx]
                + beta_sin_P * self.sin_t[self.wet_idx]
            )
            if self.include_breakpoint:
                sigma_delta_P = pm.HalfNormal("sigma_delta_P", 0.5)
                delta_P_raw   = pm.Normal("delta_P_raw", mu=0, sigma=1, dims="facility")
                delta_P       = pm.Deterministic("delta_P", delta_P_raw * sigma_delta_P,
                                                 dims="facility")
                mu_P_lin = mu_P_lin + delta_P[self.fac_idx_wet] * transition_wet
            
            mu_P  = pm.math.sigmoid(mu_P_lin)
            phi = pm.Gamma("phi", alpha=6, beta=1)
            alpha = pm.math.clip(mu_P * phi, 0.1, 100)
            beta_ = pm.math.clip((1 - mu_P) * phi, 0.1, 100)
            # Likelihood for percentage flooded on flooded days
            pm.Beta("P_obs", alpha=alpha, beta=beta_, observed=self.P[self.wet_idx])

            if INCLUDE_DISTANCE:
                # Modeling distance to water on flooded days as a separate likelihood
                beta0_D = pm.Normal("beta0_D", 2, 1)
                beta_lag_D = pm.Normal("beta_lag_D", 0, 1)
                
                mu_D_lin = (
                    beta0_D
                    + facility_effect[self.fac_idx_wet]
                    + beta_lag_D * pt.math.log1p(self.P_lag[self.wet_idx] * 100)
                )
                if self.include_breakpoint:
                    sigma_delta_D = pm.HalfNormal("sigma_delta_D", 0.5)
                    delta_D_raw   = pm.Normal("delta_D_raw", mu=0, sigma=1, dims="facility")
                    delta_D       = pm.Deterministic("delta_D", delta_D_raw * sigma_delta_D,
                                                     dims="facility")
                    mu_D_lin = mu_D_lin + delta_D[self.fac_idx_wet] * transition_wet
                
                mu_D = pm.math.clip(mu_D_lin, -5, 5)
                sigma_D = pm.HalfNormal("sigma_D", 0.5)
                # Likelihood for log distance to water on flooded days
                pm.LogNormal("D_obs", mu=mu_D, sigma=sigma_D,
                             observed=self.D_scaled[self.wet_idx])

        return self.model

    # ────────────────────── Sampling with automatic backend selection ──────────────────────
    def sample(self, draws=N_DRAWS, tune=N_TUNE, chains=N_CHAINS,
               target_accept=TARGET_ACCEPT, random_seed=RANDOM_SEED):
        r"""This method runs MCMC sampling to estimate the posterior
        distribution of the model parameters. It automatically selects
        the best available sampler based on the presence of JAX and user
        preference. The sampling process is configured with parameters
        for the number of draws, tuning steps, chains, target acceptance
        rate, and random seed for reproducibility.
        The sampling is performed within the context of the defined PyMC
        model, and the resulting trace is stored in the instance for later
        analysis and scenario generation.
        
        Note on chains: chains defaults to N_CHAINS = 4. Running at least 4
        chains gives R-hat a much better chance of catching multimodal or
        poorly identified posteriors — with only 2 chains, both can land in
        the same mode by chance and look converged when they are not.

        Note on memory: immediately after sampling, the non-centered "_raw"
        variables (facility_effect_raw, t_break_raw, delta_raw, delta_P_raw,
        delta_D_raw) are dropped from the trace, before diagnostics are
        computed and before the trace is saved. They are sampling-time
        machinery only — generate_scenarios() and the diagnostics summary
        both use the transformed (non-raw) variables — so keeping both
        copies around was pure overhead, roughly doubling the storage cost
        of every facility-level parameter.

        Parameters
        ----------
        draws : int
            Number of posterior samples to draw per chain (after tuning).
        tune : int
            Number of tuning steps (burn-in) to discard before collecting
            samples.
        chains : int
            Number of MCMC chains to run in parallel.
        target_accept : float
            Target acceptance rate for the sampler; higher values can
            improve convergence for complex models but may increase sampling
            time.
        random_seed : int
            Random seed for reproducibility of the sampling process.

        Returns
        -------
        trace : pm.backends.base.MultiTrace or pm.backends.base.JAXTrace
            The trace object containing the sampled posterior distribution
            of the model parameters (with non-centered raw variables removed).
        """

        sampler = select_sampler()
        # Cap cores for the CPU-fallback sampler specifically — JAX backends
        # parallelize chains within one process and aren't affected by this.
        cores   = min(chains, MAX_SAMPLING_CORES, os.cpu_count() or 1)

        print(f"\nSampling: {chains} chains × {draws} draws ({tune} tune)")
        print(f"  Backend : {sampler} | Cores : {cores}")

        with self.model:
            if sampler in ("numpyro", "blackjax"):
                # Import JAX module from PyMC
                if sampler == "numpyro":
                    self.trace = pmjax.sample_numpyro_nuts(
                        draws=draws,
                        tune=tune,
                        chains=chains,
                        target_accept=target_accept,
                        random_seed=random_seed,
                        progressbar=True,
                        chain_method="parallel",
                        # postprocessing_backend="cpu"  # free GPU after sampling
                    )
                else:
                    self.trace = pmjax.sample_blackjax_nuts(
                        draws=draws,
                        tune=tune,
                        chains=chains,
                        target_accept=target_accept,
                        random_seed=random_seed,
                        progressbar=True,
                    )
            else:
                self.trace = pm.sample(
                    draws=draws,
                    tune=tune,
                    chains=chains,
                    cores=cores,
                    target_accept=target_accept,
                    random_seed=random_seed,
                    progressbar=True,
                )
        # ── Drop non-centered "_raw" variables from the trace (memory) ──
        raw_var_names = ["facility_effect_raw"]
        if self.include_breakpoint:
            raw_var_names += ["t_break_raw", "delta_raw", "delta_P_raw"]
            if INCLUDE_DISTANCE:
                raw_var_names += ["delta_D_raw"]
        present = [v for v in raw_var_names if v in self.trace.posterior.data_vars]
        if present:
            self.trace.posterior = self.trace.posterior.drop_vars(present)
            print(f"  Dropped {len(present)} non-centered raw variable(s) from "
                  f"trace to reduce memory: {', '.join(present)}")
        
        self._print_diagnostics()
        return self.trace

    def _print_diagnostics(self):
        base_vars = [
            "beta0", "beta_lag", "beta_cos", "beta_sin",
            "beta0_P", "beta_lag_P", "phi", "sigma_fac",
        ]
        if self.include_breakpoint:
            base_vars += ["mu_tbreak", "sigma_tbreak", "kappa"]

        summary = az.summary(self.trace, var_names=base_vars)
        print("\n── Convergence diagnostics ──")
        print(summary.to_string())
        # Check for potential convergence issues based on R-hat and ESS
        full_summary = az.summary(self.trace)
        bad_rhat = full_summary[full_summary['r_hat'] > 1.05]
        bad_ess = full_summary[full_summary['ess_bulk'] < 400]
        if len(bad_rhat):
            # Warn about parameters with R-hat > 1.05, which may indicate lack of convergence
            warnings.warn(f"\n⚠  {len(bad_rhat)} parameters with R-hat > 1.05\n"
                          f"{bad_rhat[['r_hat']].head(10)}")
        if len(bad_ess):
            # Warn about parameters with low effective sample size, which may indicate poor mixing or insufficient sampling
            warnings.warn(f"\n⚠  {len(bad_ess)} parameters with ESS < 400\n"
                          f"{bad_ess[['ess_bulk']].head(10)}")
        if not len(bad_rhat) and not len(bad_ess):
            print("\n✓  R-hat ≤ 1.05, ESS ≥ 400 in all parameters")

    # ────────────────────── Scenario generation (parallel) ──────────────────────
    def generate_scenarios(
        self,
        n_scenarios: int = 50,
        days: int = 365,
        start_date: str = "2026-01-01",
        t_year_sim: float = 1.0,
        n_jobs: int = N_JOBS_SCENARIOS,
        output_dir: str = None,
    ) -> list[str]:
        r"""Generate synthetic scenarios in parallel and stream batches to disk."""

        if n_jobs == -1:
            warnings.warn(
                "generate_scenarios called with n_jobs=-1 (all visible CPU "
                "cores). This is not recommended on a shared machine — "
                "consider passing an explicit, smaller n_jobs instead."
            )

        if output_dir is None:
            output_dir = os.path.join(os.getcwd(), "synthetic_scenarios_tmp")
        os.makedirs(output_dir, exist_ok=True)

        posterior = self.trace.posterior

        # Extract posterior to dicts of numpy arrays — serializable by joblib
        param_names_scalar = [
            "beta0", "beta_lag", "beta_cos", "beta_sin",
            "beta0_P", "beta_lag_P", "beta_cos_P", "beta_sin_P",
            "phi",
        ]
        param_names_fac = ["facility_effect", "beta_cat"]

        if self.include_breakpoint:
            param_names_scalar += ["kappa"]
            param_names_fac += ["t_break", "delta", "delta_P", "gamma_cat"]

        if INCLUDE_DISTANCE:
            param_names_scalar += ["beta0_D", "beta_lag_D", "sigma_D"]
            if self.include_breakpoint:
                param_names_fac += ["delta_D"]

        posterior_dict = {}
        for name in param_names_scalar + param_names_fac:
            posterior_dict[name] = posterior[name].values

        # Facilities Metadata for scenario generation
        meta = (
            self.df.groupby('hf_id', observed=True)
            .agg(
                latitude=('latitude', 'first'),
                longitude=('longitude', 'first'),
                buffer_pixels=('buffer_pixels', 'first'),
                hf_payam=('hf_payam', 'first'),
                last_O=('occurrence', 'last'),
                last_P=('pct_flooded', 'last'),
            )
            .loc[self.facility_ids]
            .reset_index()
        )

        meta_dict = {
            'hf_id': meta['hf_id'].values,
            'hf_payam': meta['hf_payam'].values,
            'hf_category': self.category_ids[self.fac_cat_idx],
            'latitude': meta['latitude'].values,
            'longitude': meta['longitude'].values,
            'buffer_pixels': meta['buffer_pixels'].values,
            'last_O': meta['last_O'].values,
            'last_P': meta['last_P'].values,
            'fac_cat_idx': self.fac_cat_idx,
        }
        n_fac = self.n_facilities
        D_max_fac = np.where(meta['buffer_pixels'].values == 40, 2545.5, 3818.0)

        dates = pd.date_range(start=start_date, periods=days)
        doys = dates.dayofyear.values
        cos_t_sim = np.cos(2 * np.pi * doys / 365).astype(np.float32)
        sin_t_sim = np.sin(2 * np.pi * doys / 365).astype(np.float32)

        if t_year_sim is None:
            sim_year = pd.Timestamp(start_date).year
            t_year_sim = float(np.clip(
                (sim_year - self.year_min) / max(self.year_max - self.year_min, 1),
                0.0, 2.0
            ))
        print(f"\nt_year_sim = {t_year_sim:.3f}  "
              f"({'extrapolating' if t_year_sim > 1.0 else 'within training range'})")

        # split scenarios between workers
        n_workers = cpu_count() if n_jobs == -1 else abs(n_jobs)
        n_workers = min(n_workers, n_scenarios)
        scen_batches = np.array_split(np.arange(n_scenarios), n_workers)
        seeds = np.random.SeedSequence(RANDOM_SEED).spawn(n_workers)
        seeds_int = [int(s.generate_state(1)[0]) for s in seeds]

        print(f"Generating {n_scenarios} scenarios on {n_workers} worker(s) "
              f"(joblib, streamed to {output_dir})...")

        batch_paths = Parallel(n_jobs=n_jobs, backend="loky", verbose=5)(
            delayed(_generate_scenario_batch)(
                scen_ids=list(batch),
                posterior_dict=posterior_dict,
                meta_dict=meta_dict,
                cos_t_sim=cos_t_sim,
                sin_t_sim=sin_t_sim,
                dates=dates,
                doys=doys,
                t_year_sim=t_year_sim,
                n_fac=n_fac,
                D_max_fac=D_max_fac,
                include_distance=INCLUDE_DISTANCE,
                include_breakpoint=self.include_breakpoint,
                output_dir=output_dir,
                batch_id=i,
                seed=seeds_int[i],
            )
            for i, batch in enumerate(scen_batches)
        )

        print(f"\nWrote {len(batch_paths)} batch file(s) "
              f"({n_scenarios} scenarios × {n_fac} facilities × {days} days) "
              f"to {output_dir}")
        return batch_paths


# ==========================================================
# 3. SCENARIO GENERATION  — parallelized with joblib, streamed to disk
# ==========================================================
def _generate_scenario_batch(
    scen_ids,
    posterior_dict,
    meta_dict,
    cos_t_sim,
    sin_t_sim,
    dates,
    doys,
    t_year_sim,
    n_fac,
    D_max_fac,
    include_distance,
    include_breakpoint,
    output_dir,
    batch_id,
    seed,
):
    r"""Generate one worker batch and stream it to one CSV file."""

    rng = np.random.default_rng(seed)
    days = len(dates)

    n_chains = posterior_dict['beta0'].shape[0]
    n_post = posterior_dict['beta0'].shape[1]

    def draw_scalar(name):
        c = rng.integers(n_chains)
        d = rng.integers(n_post)
        return float(posterior_dict[name][c, d])

    def draw_fac(name):
        c = rng.integers(n_chains)
        d = rng.integers(n_post)
        return posterior_dict[name][c, d]

    batch_path = os.path.join(output_dir, f"scenario_batch_{batch_id:03d}.csv")

    with open(batch_path, "w", newline="") as f:
        for j, scen in enumerate(scen_ids):
            beta0 = draw_scalar("beta0")
            beta_lag = draw_scalar("beta_lag")
            beta_cos = draw_scalar("beta_cos")
            beta_sin = draw_scalar("beta_sin")
            beta0_P = draw_scalar("beta0_P")
            beta_lag_P = draw_scalar("beta_lag_P")
            beta_cos_P = draw_scalar("beta_cos_P")
            beta_sin_P = draw_scalar("beta_sin_P")
            phi = draw_scalar("phi")
            beta_cat = draw_fac("beta_cat")

            if include_distance:
                beta0_D = draw_scalar("beta0_D")
                beta_lag_D = draw_scalar("beta_lag_D")
                sigma_D = draw_scalar("sigma_D")
                if include_breakpoint:
                    delta_D = draw_fac("delta_D")

            fac_eff = draw_fac("facility_effect")

            if include_breakpoint:
                t_break = draw_fac("t_break")
                delta = draw_fac("delta")
                delta_P = draw_fac("delta_P")
                kappa = draw_scalar("kappa")
                gamma_cat = draw_fac("gamma_cat")
                transition_fac = 1.0 / (1.0 + np.exp(-kappa * (t_year_sim - t_break)))
            else:
                transition_fac = np.zeros(n_fac, dtype=np.float32)

            O_sim = np.zeros((n_fac, days), dtype=np.int8)
            P_sim = np.zeros((n_fac, days), dtype=np.float32)
            D_sim = np.zeros((n_fac, days), dtype=np.float32)

            lag_O = meta_dict['last_O'].astype(np.float32)
            lag_P = meta_dict['last_P'].astype(np.float32)

            for d in range(days):
                cos_d = cos_t_sim[d]
                sin_d = sin_t_sim[d]

                logit = (
                    beta0
                    + fac_eff
                    + beta_cat[meta_dict['fac_cat_idx']]
                    + beta_lag * lag_O
                    + beta_cos * cos_d
                    + beta_sin * sin_d
                )
                if include_breakpoint:
                    logit = logit + (
                        delta + gamma_cat[meta_dict['fac_cat_idx']]
                    ) * transition_fac
                p_occ = 1.0 / (1.0 + np.exp(-logit))
                O_d = (rng.random(n_fac) < p_occ).astype(np.int8)

                wet = O_d == 1
                P_d = np.zeros(n_fac, dtype=np.float32)
                if wet.any():
                    mu_P_lin = (
                        beta0_P
                        + fac_eff[wet]
                        + beta_lag_P * lag_P[wet]
                        + beta_cos_P * cos_d
                        + beta_sin_P * sin_d
                    )
                    if include_breakpoint:
                        mu_P_lin = mu_P_lin + delta_P[wet] * transition_fac[wet]
                    mu_P = 1.0 / (1.0 + np.exp(-mu_P_lin))
                    a = np.clip(mu_P * phi, 0.1, 100)
                    b = np.clip((1 - mu_P) * phi, 0.1, 100)
                    P_d[wet] = rng.beta(a, b).astype(np.float32)

                D_d = np.where(O_d == 0, D_max_fac, 0.0).astype(np.float32)

                if include_distance and wet.any():
                    mu_D_lin = (
                        beta0_D
                        + fac_eff[wet]
                        + beta_lag_D * np.log1p(lag_P[wet] * 100)
                    )
                    if include_breakpoint:
                        mu_D_lin = mu_D_lin + delta_D[wet] * transition_fac[wet]
                    mu_D = np.clip(mu_D_lin, -5, 5)
                    D_scaled_d = rng.lognormal(mu_D, sigma_D)
                    D_d[wet] = (np.clip(D_scaled_d, 0.0, 1.0) * D_max_fac[wet]).astype(np.float32)

                O_sim[:, d] = O_d
                P_sim[:, d] = P_d
                D_sim[:, d] = D_d
                lag_O = O_d.astype(np.float32)
                lag_P = P_d

            fac_rep = np.repeat(np.arange(n_fac), days)
            date_til = np.tile(dates, n_fac)
            doy_til = np.tile(doys, n_fac)

            df_scen = pd.DataFrame({
                'hf_id': meta_dict['hf_id'][fac_rep],
                'hf_payam': meta_dict['hf_payam'][fac_rep],
                'hf_category': meta_dict['hf_category'][fac_rep],
                'latitude': meta_dict['latitude'][fac_rep],
                'longitude': meta_dict['longitude'][fac_rep],
                'buffer_pixels': meta_dict['buffer_pixels'][fac_rep],
                'date': date_til,
                'occurrence': O_sim.ravel(),
                'pct_flooded': P_sim.ravel(),
                'min_distance': D_sim.ravel(),
                'day_of_year': doy_til,
                'year': pd.Timestamp(dates[0]).year,
                'scenario_id': scen,
                'is_synthetic': 1,
            })
            for col in ('hf_id', 'hf_payam', 'hf_category'):
                df_scen[col] = df_scen[col].astype('category')

            df_scen.to_csv(f, header=(j == 0), index=False)
            del df_scen

    return batch_path


def stream_concat_csv(input_paths, output_path, columns=None, chunksize=200_000):
    r"""Append CSV files into one output by streaming fixed-size chunks."""

    for p in input_paths:
        for chunk in pd.read_csv(p, chunksize=chunksize):
            if columns is not None:
                chunk = chunk.reindex(columns=columns)
            chunk.to_csv(output_path, mode='a', header=False, index=False)


def load_scenarios_sample(batch_paths, max_files=None):
    r"""Load only a bounded subset of scenario batch files into memory."""

    paths = batch_paths if max_files is None else batch_paths[:max_files]
    return pd.concat((pd.read_csv(p) for p in paths), ignore_index=True)
