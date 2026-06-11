# -*- coding: utf-8 -*-
__author__ = "Macarena Merida Floriano, mmef"
__maintainer__ = "Macarena Merida Floriano, mmef"
__email__ = "mmef@gmv.com"

from dataclasses import dataclass, field
import numpy as np
import pandas as pd

from impact_model.config import (
    SPELL_THRESHOLDS,
    INTENSITY_THRESHOLDS,
    DISTANCE_CRITICAL_M,
    DISTANCE_MODERATE_M,
    RECOVERY_GAP_DAYS,
    CATEGORY_MULTIPLIERS,
    ACTIVE_RULES,
    RULE_WEIGHTS,
)


# ─────────────────────────────────────────────────────────────────────────────
# VULNERABILITY ENGINE
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class VulnerabilityConfig:
    r"""
    Configuration for the Vulnerability Engine.
    """

    spell_thresholds: list = field(default_factory=lambda: SPELL_THRESHOLDS)
    intensity_thresholds: list = field(default_factory=lambda: INTENSITY_THRESHOLDS)
    distance_critical: float = DISTANCE_CRITICAL_M
    distance_moderate: float = DISTANCE_MODERATE_M
    recovery_gap_days: int = RECOVERY_GAP_DAYS
    category_multipliers: dict = field(default_factory=lambda: CATEGORY_MULTIPLIERS)
    active_rules: dict = field(default_factory=lambda: ACTIVE_RULES)
    rule_weights: dict = field(default_factory=lambda: RULE_WEIGHTS)


def _build_threshold_arrays(thresholds: list) -> tuple[np.ndarray, np.ndarray]:
    r"""Given a list of (threshold, loss) pairs, return sorted arrays of
    thresholds and losses for vectorized application.

    Parameters
    ----------
    thresholds : list of tuples
        Each tuple is (threshold_value, loss_value). The list can be in any order;
        it will be sorted internally by threshold_value in ascending order.

    Returns
    -------
    thrs : np.ndarray
        Array of threshold values sorted in ascending order.
    losses : np.ndarray
        Array of corresponding loss values, aligned with the sorted thresholds.
    """

    sorted_t = sorted(thresholds, key=lambda x: x[0])
    thrs = np.array([t for t, _ in sorted_t], dtype=np.float32)
    losses = np.array([l for _, l in sorted_t], dtype=np.float32)
    return thrs, losses


def _apply_thresholds_vec(values: np.ndarray, thrs: np.ndarray,
                          losses: np.ndarray) -> np.ndarray:
    r"""For each value in the input array, determine the loss based on the
    highest threshold that it meets or exceeds. This is done in a vectorized
    manner for efficiency.

    Parameters
    ----------
    values : np.ndarray
        Array of values to evaluate against the thresholds.
    thrs : np.ndarray
        Array of threshold values sorted in ascending order.
    losses : np.ndarray
        Array of corresponding loss values, aligned with the sorted thresholds.

    Returns
    -------
    np.ndarray
        Array of loss values corresponding to the input values. Each value's loss
        is determined by the highest threshold it meets or exceeds.
    """

    result = np.zeros(len(values), dtype=np.float32)
    for thr, loss in zip(thrs, losses):
        result[values >= thr] = loss
    return result


class VulnerabilityEngine:
    r"""This class implements the vulnerability assessment rules. It can be
    instantiated with a custom configuration, and then applied to a dataframe
    of daily time series data for a single facility to compute the combined
    loss metric.

    Methods
    -------
    - apply(df): takes a dataframe with daily data for a single facility and
        computes the vulnerability metrics and combined loss according to the
        active rules and their weights.
    - _label_spells(occ): helper function to label consecutive wet spells in
        the input array.
    - _merge_spells(spell_id, occ): helper function to merge wet spells that
        are separated by dry spells within the recovery gap.
    - _r1(spell_dur): computes R1 loss based on spell duration using the
        configured thresholds.
    - _r2(pct, spell_id, spell_dur): computes R2 loss based on intensity
        using the configured thresholds.
    - _r3(dist, occ): computes R3 loss based on proximity using the configured
        distance thresholds.
    - _r4(merged_id, spell_id): computes R4 loss based on fatigue using the
        merged spell IDs and the same thresholds as R1.
    - _r5(category): computes R5 multiplier based on facility category.
    """

    def __init__(self, cfg: VulnerabilityConfig = None):
        r"""Initialize the Vulnerability Engine with the given configuration. If no
        configuration is provided, it will use the default values defined in the
        VulnerabilityConfig dataclass.
        """

        self.cfg = cfg or VulnerabilityConfig()
        # Pre-compile threshold arrays once per instance
        self._r1_thrs, self._r1_losses = _build_threshold_arrays(self.cfg.spell_thresholds)
        self._r2_thrs, self._r2_losses = _build_threshold_arrays(self.cfg.intensity_thresholds)

    def _label_spells(self, occ):
        r"""
        This helper function calculates the lengths of consecutive flooding
        spells (occurrence == 1) from a binary array of occurrences. It iterates
        through the array, counting the length of each wet spell and appending it
        to the list of lengths when a dry day (occurrence == 0) is encountered.

        Parameters
        ----------
        occ : numpy.ndarray
            Array of binary values indicating occurrence (1 for flooded, 0
            for not flooded).

        Returns
        -------
        spell_id : numpy.ndarray
            Array of spell IDs for each day, where consecutive wet days
            share the same ID.
        spell_dur : numpy.ndarray
            Array of spell durations for each day, indicating the length
            of the wet spell it belongs to.
        """

        spell_id = np.full(len(occ), -1, dtype=int)
        spell_dur = np.zeros(len(occ), dtype=int)
        sid = 0
        i = 0
        while i < len(occ):
            if occ[i] == 1:
                start = i
                while i < len(occ) and occ[i] == 1:
                    i += 1
                dur = i - start
                spell_id[start:i] = sid
                spell_dur[start:i] = dur
                sid += 1
            else:
                i += 1
        return spell_id, spell_dur

    def _merge_spells(self, spell_id, occ):
        r"""Method to merge wet spells that are separated by dry spells
        within the recovery gap.

        Parameters
        ----------
        spell_id : numpy.ndarray
            Array of spell IDs for each day, where consecutive wet days
            share the same ID.
        occ : numpy.ndarray
            Array of binary values indicating occurrence (1 for flooded, 0
            for not flooded).

        Returns
        -------
        merged : numpy.ndarray
            Array of merged spell IDs, where wet spells separated by dry
            spells within the recovery gap are assigned the same ID.
        """

        gap = self.cfg.recovery_gap_days
        merged = spell_id.copy()
        n = len(occ)
        i = 0

        while i < n:
            if merged[i] < 0:
                i += 1
                continue

            # Find end of current flooded spell
            j = i
            while j < n and occ[j] == 1:
                j += 1
            end_wet = j

            # Search for the next flooded spell within the gap
            dry_count = 0
            next_start = None
            k = end_wet
            while k < n and dry_count <= gap:
                if occ[k] == 0:
                    dry_count += 1
                    k += 1
                else:
                    next_start = k
                    break

            if next_start is not None:
                sid = merged[i]
                # Assign dry days within the gap to the current sid
                for m in range(end_wet, next_start):
                    merged[m] = sid
                # Assign second flooded spell to the current sid
                m = next_start
                while m < n and occ[m] == 1:
                    merged[m] = sid
                    m += 1
                # Advance i to the start of the second flooded spell to continue from there
                i = next_start
            else:
                # No merge: advance to the day after end_wet
                i = end_wet + 1

        return merged

    def _r1(self, spell_dur):
        r"""Method to compute R1 loss based on spell duration using the
        configured thresholds.
        
        Parameters
        ----------
        spell_dur : numpy.ndarray
            Array of spell durations for each day, indicating the length of
            the flooding spell it belongs to.
            
        Returns
        -------
        result : np.ndarray
            Array of R1 loss values corresponding to the input spell durations.
            Each value's loss is determined by the highest threshold it meets or
            exceeds.
        """

        result = np.zeros(len(spell_dur), dtype=np.float32)
        wet = spell_dur > 0
        if wet.any():
            result[wet] = _apply_thresholds_vec(
                spell_dur[wet].astype(np.float32), self._r1_thrs, self._r1_losses
            )
        return result

    def _r2(self, pct, spell_id, spell_dur):
        r"""Method to compute R2 loss based on intensity using the configured
        thresholds. Intensity is defined as mean_pct_flooded * spell_duration
        for flood days.
        
        Parameters
        ----------
        pct : numpy.ndarray
            Array of percentage flooded values for each day (0.0 to 1.0).
        spell_id : numpy.ndarray
            Array of spell IDs for each day, where consecutive flooded days share the
            same ID.
        spell_dur : numpy.ndarray
            Array of spell durations for each day, indicating the length of the
            flooding spell it belongs to.
        
        Returns
        -------
        result : np.ndarray
            Array of R2 loss values corresponding to the input data. Each value's
            loss is determined by the intensity of the flood spell it belongs to,
            using the configured thresholds.
        """

        result = np.zeros(len(pct), dtype=np.float32)
        wet = spell_id >= 0
        if not wet.any():
            return result
        df_tmp = pd.DataFrame({"sid": spell_id[wet], "pct": pct[wet], "dur": spell_dur[wet]})
        grp = df_tmp.groupby("sid").agg(mean_pct=("pct", "mean"), dur=("dur", "first"))
        grp["intensity"] = grp["mean_pct"] * grp["dur"]
        grp["loss"] = _apply_thresholds_vec(
            grp["intensity"].values.astype(np.float32), self._r2_thrs, self._r2_losses
        )
        sid_to_loss = grp["loss"].to_dict()
        wet_idx = np.where(wet)[0]
        result[wet_idx] = np.array(
            [sid_to_loss.get(s, 0.0) for s in spell_id[wet]], dtype=np.float32
        )
        return result

    def _r3(self, dist, occ):
        r"""Method to compute R3 loss based on proximity using the configured distance
        thresholds. If distance < critical, loss=0.5; if distance is between
        critical and moderate, loss=0.25; if distance >= moderate, loss=0.
        
        Parameters
        ----------
        dist : numpy.ndarray
            Array of minimum distance to water for each day (in meters).
        occ : numpy.ndarray
            Array of binary values indicating occurrence (1 for flooded, 0 for not flooded).
        
        Returns
        -------
        result : np.ndarray
            Array of R3 loss values corresponding to the input data. Each value's
            loss is determined by the distance to water on flood days, using the
            configured thresholds.
        """

        loss = np.zeros(len(dist), dtype=np.float32)
        wet = occ == 1
        loss[wet & (dist < self.cfg.distance_critical)] = 0.50
        loss[wet & (dist >= self.cfg.distance_critical) &
             (dist < self.cfg.distance_moderate)] = 0.25
        return loss

    def _r4(self, merged_id, spell_id):
        r"""Method to compute R4 loss based on merged spell IDs using the
        configured thresholds. If a merged spell is active, the loss is
        determined by the effective duration of the merged spell.
        
        Parameters
        ----------
        merged_id : numpy.ndarray
            Array of merged spell IDs for each day, where consecutive flooded days
            that are part of the same merged spell share the same ID.
        spell_id : numpy.ndarray
            Array of spell IDs for each day, where consecutive flooded days share the
            same ID.
        
        Returns
        -------
        result : np.ndarray
            Array of R4 loss values corresponding to the input data. Each value's
            loss is determined by the effective duration of the merged spell it belongs to.
        """

        result = np.zeros(len(merged_id), dtype=np.float32)
        active = merged_id >= 0
        if not active.any():
            return result
        m_ids = merged_id[active]
        counts = np.bincount(m_ids, minlength=int(m_ids.max()) + 1)
        eff_durs = counts[m_ids].astype(np.float32)
        fused = _apply_thresholds_vec(eff_durs, self._r1_thrs, self._r1_losses)
        wet_active = active & (spell_id >= 0)
        result[wet_active] = fused[wet_active[active]]
        return result

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        r"""Method to apply the vulnerability assessment rules to a
        dataframe containing daily time series data for a single facility.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing daily data for a single facility, with columns:
            - 'occurrence': binary (1 for flooded, 0 for not flooded)
            - 'pct_flooded': percentage of area flooded (0.0 to 1
                for each day)
            - 'min_distance': minimum distance to water (in meters)
            - 'hf_category': category of the health facility (for R5)
        
        Returns
        -------
        df : pd.DataFrame
            DataFrame with additional columns for spell durations, spell IDs,
            merged spell IDs, individual rule losses, category multipliers, and
            combined loss.
        """

        df = df.sort_values("date").copy().reset_index(drop=True)
        occ = df["occurrence"].values.astype(int)
        pct = df["pct_flooded"].values.astype(np.float32)
        dist = df["min_distance"].values.astype(np.float32)
        cat = str(df["hf_category"].iloc[0])

        spell_id, spell_dur = self._label_spells(occ)
        merged_id = self._merge_spells(spell_id.copy(), occ)

        ar = self.cfg.active_rules
        w = self.cfg.rule_weights
        z = np.zeros(len(occ), dtype=np.float32)

        r1 = self._r1(spell_dur) if ar.get("R1_spell") else z
        r2 = self._r2(pct, spell_id, spell_dur) if ar.get("R2_intensity") else z
        r3 = self._r3(dist, occ) if ar.get("R3_proximity") else z
        r4 = self._r4(merged_id, spell_id) if ar.get("R4_fatigue") else z
        r5 = self.cfg.category_multipliers.get(cat, 1.0) if ar.get("R5_category") else 1.0

        wt = (w.get("R1_spell", 0.45) + w.get("R2_intensity", 0.25) +
              w.get("R3_proximity", 0.20) + w.get("R4_fatigue", 0.10))
        if wt == 0:
            wt = 1.0

        base_loss = (
            w.get("R1_spell", 0.45) * r1 +
            w.get("R2_intensity", 0.25) * r2 +
            w.get("R3_proximity", 0.20) * r3 +
            w.get("R4_fatigue", 0.10) * r4
        ) / wt

        df["spell_duration"] = spell_dur
        df["spell_id"] = spell_id
        df["merged_spell_id"] = merged_id
        df["r1_loss"] = r1
        df["r2_loss"] = r2
        df["r3_loss"] = r3
        df["r4_loss"] = r4
        df["r5_mult"] = float(r5)
        df["combined_loss"] = np.clip(base_loss * r5, 0.0, 1.0).astype(np.float32)
        return df
