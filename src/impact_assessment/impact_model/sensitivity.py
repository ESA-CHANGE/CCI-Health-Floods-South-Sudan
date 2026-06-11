# -*- coding: utf-8 -*-
__author__ = "Macarena Merida Floriano, mmef"
__maintainer__ = "Macarena Merida Floriano, mmef"
__email__ = "mmef@gmv.com"

import numpy as np
import pandas as pd

from impact_model.config import ACTIVE_RULES, CAT_ORDER, SENSITIVITY_LABELS, SENSITIVITY_SETS
from impact_model.engine import VulnerabilityConfig, VulnerabilityEngine


def sensitivity_r1(df_pool_raw: pd.DataFrame,
                   threshold_sets=SENSITIVITY_SETS,
                   labels=SENSITIVITY_LABELS) -> dict:
    r"""Re-process just the synthetic data with different R1 thresholds to
    analyze the sensitivity of the results to the choice of R1 parameters.
    
    Parameters
    ----------
    df_pool_raw : pd.DataFrame
        Raw DataFrame containing the pool of scenarios with columns including
        `hf_id`, `hf_category`, `scenario_id`, `date`, `is_synthetic`, and other relevant fields.
    threshold_sets : list of dict, optional
        List of dictionaries specifying the R1 thresholds to test, where each dictionary
        should have keys corresponding to the R1 parameters (e.g., `spell_thresholds`) and values
        specifying the thresholds to use. Defaults to `SENSITIVITY_SETS` from the
        configuration.
    labels : list of str, optional
        List of labels corresponding to each set of thresholds for identification in the results.
        Defaults to `SENSITIVITY_LABELS` from the configuration.
    
    Returns
    -------
    results : dict
        A dictionary where keys are the provided labels and values are dictionaries containing
        the median loss for each category under the corresponding R1 threshold set.
    """

    print("\n─── SENSITIVITY R1 ───")
    df_syn = df_pool_raw[df_pool_raw["is_synthetic"] == 1].copy()
    df_syn["_year"] = df_syn["date"].dt.year
    groups = [g for _, g in df_syn.groupby(["hf_id", "scenario_id"], sort=False)]

    results = {}
    for label, tset in zip(labels, threshold_sets):
        cfg = VulnerabilityConfig()
        cfg.spell_thresholds = tset
        cfg.active_rules = {k: (k == "R1_spell") for k in ACTIVE_RULES}
        cfg.rule_weights = {"R1_spell": 1.0, "R2_intensity": 0.0,
                            "R3_proximity": 0.0, "R4_fatigue": 0.0}
        engine = VulnerabilityEngine(cfg)
        cat_loss = {}
        for grp in groups:
            cat = str(grp["hf_category"].iloc[0])
            scen = int(grp["scenario_id"].iloc[0])
            res = engine.apply(grp)
            cat_loss.setdefault((cat, scen), []).append(res["combined_loss"].mean())
        cat_means = {}
        for cat in CAT_ORDER:
            vals = [v for (c, s), vlist in cat_loss.items() if c == cat for v in vlist]
            cat_means[cat] = np.median(vals) if vals else 0.0
        results[label] = cat_means
        print(f"  {label}: {cat_means}")
    return results
