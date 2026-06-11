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
N_CHAINS = 2  # Number of MCMC chains; adjust depending on resources
N_DRAWS = 1500  # Number of posterior samples to draw per chain (after tuning)
N_TUNE = 3000  # Number of tuning steps (burn-in); adjust for better convergence if needed
TARGET_ACCEPT = 0.92  # Higher target_accept can improve convergence for complex models but increases sampling time
RANDOM_SEED = 42

N_JOBS_SCENARIOS = -1  # To use all cores keep -1 value
# Sampler preference: "numpyro" > "blackjax" > "pymc" (automatic fallback)
PREFERRED_SAMPLER = "numpyro"
