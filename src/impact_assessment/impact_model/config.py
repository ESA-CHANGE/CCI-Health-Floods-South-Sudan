# -*- coding: utf-8 -*-
__author__ = "Macarena Merida Floriano, mmef"
__maintainer__ = "Macarena Merida Floriano, mmef"
__email__ = "mmef@gmv.com"

import os

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

SYNTHETIC_CSV = "../data/model_output/bayfloodgen_output.csv"
OUTPUT_DIR    = "../data/impact_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
PLOT_DIR      = os.path.join(OUTPUT_DIR, "impact_plots")
os.makedirs(PLOT_DIR, exist_ok=True)

OBS_CUTOFF = 2025

# ─────────────────────── RULE PARAMETERS ───────────────────────
# Based on empirical calibration:
# p50=5d, p75=13d, p90=33d, p95=60d, p99=132d

# Spell duration thresholds (in days) and corresponding losses for R1
SPELL_THRESHOLDS = [
    (100, 1.00),
    (60,  0.75),
    (33,  0.50),
    (13,  0.25),
    (5,   0.10),
]

# p75=0.53, p90=2.61, p95=7.55, p99=47.52
# Intensity thresholds (mean_pct_flooded * spell_duration) and corresponding losses for R2
INTENSITY_THRESHOLDS = [
    (47.0, 0.80),
    (7.5,  0.50),
    (2.6,  0.25),
    (0.5,  0.10),
]

# p10=255m, p25=774m, p50=1527m
# Proximity thresholds for R3: <500m = 0.5 loss, 500-1200m = 0.25 loss, >1200m = no loss
DISTANCE_CRITICAL_M = 500.0
DISTANCE_MODERATE_M = 1200.0

# R4: if merged spell duration >= thresholds, apply same losses as R1
RECOVERY_GAP_DAYS = 4

# Multipliers for R5 based on facility category; if category is missing, assume multiplier=1.0 (no additional loss)
CATEGORY_MULTIPLIERS = {
    "SAFE"    : 1.00,
    "CHRONIC" : 1.20,
    "UNSTABLE": 1.40,
    "CRITICAL": 1.60,
}
CATEGORY_COLORS = {
    "SAFE"    : "#4CAF50",
    "CHRONIC" : "#2196F3",
    "UNSTABLE": "#FF9800",
    "CRITICAL": "#F44336",
}
CAT_ORDER = ["SAFE", "CHRONIC", "UNSTABLE", "CRITICAL"]

# To specify which rules are active in the impact calculation
ACTIVE_RULES = {
    "R1_spell"    : True,
    "R2_intensity": True,
    "R3_proximity": True,
    "R4_fatigue"  : True,
    "R5_category" : True,
}
# To specify the relative importance of each rule in the combined loss calculation
RULE_WEIGHTS = {
    "R1_spell"    : 0.45,
    "R2_intensity": 0.25,
    "R3_proximity": 0.20,
    "R4_fatigue"  : 0.10,
}

# Paralelization: adjust according to available CPUs
N_JOBS     = 4     # -1 = all cores
BATCH_SIZE = 50    # series per batch; reduce if RAM issues

# Exceedance thresholds for summary tables (proportion of scenarios exceeding these loss levels)
EXCEEDANCE_THRESHOLDS = [0.10, 0.25, 0.50, 0.75, 1.00]

# Sensitivity R1
SENSITIVITY_SETS = [
    [(130, 1.00), (70, 0.75), (40, 0.50), (16, 0.25), (6, 0.10)],   # conservative
    SPELL_THRESHOLDS,                                                   # base
    [(70,  1.00), (45, 0.75), (25, 0.50), (10, 0.25), (3, 0.10)],   # aggressive
]
SENSITIVITY_LABELS = ["Conservative", "Base", "Aggressive"]
