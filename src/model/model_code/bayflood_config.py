# -*- coding: utf-8 -*-

__author__ = "Macarena Merida Floriano, mmef"
__maintainer__ = "Macarena Merida Floriano, mmef"
__email__ = "mmef@gmv.com"

"""
BayFloodGEN configuration constants.

This script defines global configuration parameters for the BayFloodGEN model, such as sampling
settings and sampler preferences. It is imported by other modules to ensure consistent
configuration across the codebase.
"""

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
INCLUDE_DISTANCE = True  # if False, distance to water is not modeled, only occurrence and pct_flooded
# Set this to True only if your own EDA / a statistical test (Chow test,
# CUSUM, Bai-Perron, Mann-Kendall change point, a simple before/after
# proportions test, etc.) supports an actual regime shift in the observed
# series for this region. If a region shows no such shift, set it to False:
# the model then reduces to a plain hierarchical seasonal + lag model, with
# no transition function, t_break, kappa, delta, or gamma_cat terms defined
# or sampled at all. This is what makes the *same* script usable for both
# "broken" and "unbroken" regions, instead of forcing break machinery onto
# data that doesn't support it.
INCLUDE_BREAKPOINT = True # if False, breakpoint is not modeled, only occurrence and pct_flooded

# ── Breakpoint location prior (only used if INCLUDE_BREAKPOINT = True) ────
# mu_tbreak ~ Normal(MU_TBREAK_PRIOR_MEAN, MU_TBREAK_PRIOR_SD) is the prior
# on the AVERAGE breakpoint location across facilities, expressed as a
# *fraction of the observed time range* — recall t_year is scaled so that
# 0 = year_min and 1 = year_max in the training data. sigma_tbreak (set via
# TBREAK_FACILITY_SPREAD_SCALE, below) separately controls how much each
# individual facility's own break is allowed to drift away from that
# average; it is a hyperprior scale, not a break location, and rarely needs
# changing per region.
#
# HOW TO SET MU_TBREAK_PRIOR_MEAN / MU_TBREAK_PRIOR_SD FROM YOUR OWN EDA
# -----------------------------------------------------------------------
#   1) Identify the calendar year (or fractional year) your statistical
#      test flags as the most likely break point.
#   2) Convert it to a fraction of the observed range:
#         fraction = (break_year - year_min) / (year_max - year_min)
#   3) Set MU_TBREAK_PRIOR_MEAN = fraction.
#   4) Set MU_TBREAK_PRIOR_SD according to how confident the test is in
#      that location (see the examples below — smaller SD = tighter /
#      more confident prior).
#
# WORKED EXAMPLES (assuming a 2015–2024 observed range, a 9-year span)
# -----------------------------------------------------------------------
#   Observed break pattern                  fraction   MEAN   SD
#   ----------------------------------------------------------------
#   Early, sharp, high-confidence break       ~2017     0.22   0.05
#   (e.g. a Chow/CUSUM test significant
#   at one specific, narrow year)
#
#   Mid-series break, moderate confidence    ~2020-21   0.61   0.10–0.15
#   (break detected within a ~1-2 year
#   window, not pinned to a single year)
#
#   Late, low-confidence / break only          ~2023     0.89   0.20–0.25
#   suspected visually, not confirmed by
#   a formal test
#
#   No break detected at all                    —         —      —
#   → set INCLUDE_BREAKPOINT = False instead of inflating SD further.
#     A very wide SD does not cleanly "turn off" the break term — it just
#     lets kappa and t_break wander with little data to anchor them,
#     which reintroduces the same convergence failure this change set is
#     meant to fix.
#
# South Sudan health-facility default keptfor a confirmed
# regime shift detected around 2021 in a ~2015-2024 series:
MU_TBREAK_PRIOR_MEAN = 0.7
MU_TBREAK_PRIOR_SD   = 0.15

# Spread of each facility's own break around the average mu_tbreak. This is
# a hyperprior scale (sigma_tbreak ~ HalfNormal(TBREAK_FACILITY_SPREAD_SCALE)),
# not a break location, so it rarely needs to be re-derived from EDA the way
# MU_TBREAK_PRIOR_MEAN/SD do. Increase it only if you expect facilities to
# break at noticeably different times from one another.
TBREAK_FACILITY_SPREAD_SCALE = 0.10

# ── kappa (transition speed) prior ─────────────────────────────────────────
# kappa controls how quickly the sigmoid transition moves from the
# pre-break to the post-break regime. HalfNormal(0, 5) put a lot of mass
# both near 0 (effectively "no transition") AND out in a long flat tail —
# once kappa is large enough that the sigmoid is already behaving like a
# near-perfect step function, increasing kappa further barely changes the
# likelihood at all. A Gamma(alpha, beta) prior with alpha > 1 moves the
# mode away from zero and has a lighter tail, ruling out both the "no
# transition" region and the unbounded "instant step" tail.
#   mean = alpha / beta,  sd = sqrt(alpha) / beta
KAPPA_ALPHA = 8.0
KAPPA_BETA  = 1.2     # mean ≈ 6.7, sd ≈ 2.4 (relative sd ≈ 36%, vs ≈125% before)

N_CHAINS = 4  # Number of MCMC chains; adjust depending on resources
N_DRAWS = 1500  # Number of posterior samples to draw per chain (after tuning)
N_TUNE = 3000  # Number of tuning steps (burn-in); adjust for better convergence if needed
TARGET_ACCEPT = 0.92  # Higher target_accept can improve convergence for complex models but increases sampling time
RANDOM_SEED = 42

# Caps multiprocessing cores used by PyMC's CPU-fallback sampler (only
# relevant if neither numpyro nor blackjax is installed — the JAX backends
# parallelize chains within a single process instead of spawning one OS
# process per chain, so this cap doesn't apply to them).
MAX_SAMPLING_CORES = 4

N_JOBS_SCENARIOS = 4  # To use all cores keep -1 value
# How many synthetic-scenario batch files (see section 3) get loaded into
# memory for validation plots, instead of the entire synthetic dataset.
# Roughly proportional to scenario count (batch size ≈ n_scenarios / n_jobs).
# Set to None to load every batch file (only if you have RAM to spare).
N_BATCH_FILES_FOR_VALIDATION = 4
# Sampler preference: "numpyro" > "blackjax" > "pymc" (automatic fallback)
PREFERRED_SAMPLER = "numpyro"
