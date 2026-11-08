"""Candidate trading signals for the DA/RT spread.

Position convention (MW held for one hour):
    +1  long the spread  (~ PJM DEC: buy DA, sell RT)  -> earns RT - DA
    -1  short the spread (~ PJM INC: sell DA, buy RT)  -> earns DA - RT
     0  flat
Fractional sizes in [-1, 1] are allowed (risk-sized signal).

Every signal reads only *tradable* inputs (features known at 10:30 EPT on D-1) or model
predictions built from them, and returns 0 wherever an input is missing.
Rationale for the candidate set: ``reports/phase5_signal_rationale.md``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Phase 2 spike-prone conditions, fixed in advance (not tuned): High/Extreme forecast-load
# regimes and the evening peak carried most extreme spreads (lift 1.6x-6.7x and 2.2x).
RULE_FLAT_LOAD_REGIMES = ("High", "Extreme")
RULE_FLAT_HOURS = tuple(range(17, 22))        # HE17-HE21


def _clean(pos: np.ndarray | pd.Series, valid: pd.Series) -> pd.Series:
    pos = pd.Series(np.asarray(pos, dtype=float), index=valid.index)
    return pos.where(valid & pos.notna(), 0.0).clip(-1.0, 1.0)


def flat(df: pd.DataFrame, valid: pd.Series) -> pd.Series:
    """S0: never trade."""
    return _clean(np.zeros(len(df)), valid)


def always_long(df: pd.DataFrame, valid: pd.Series) -> pd.Series:
    """S1: long the spread every hour (plan benchmark)."""
    return _clean(np.ones(len(df)), valid)


def always_short(df: pd.DataFrame, valid: pd.Series) -> pd.Series:
    """S2: short the spread every hour (base-rate benchmark: RT < DA in ~2/3 of hours)."""
    return _clean(-np.ones(len(df)), valid)


def rule_based(df: pd.DataFrame, valid: pd.Series,
               flat_regimes=RULE_FLAT_LOAD_REGIMES, flat_hours=RULE_FLAT_HOURS) -> pd.Series:
    """S3: short, except flat in the spike-prone conditions identified in Phase 2. No model."""
    risky = df["fc_load_regime"].astype(str).isin(flat_regimes) | df["hour_ending"].isin(flat_hours)
    return _clean(np.where(risky, 0.0, -1.0), valid)


def prob_threshold(p_up: pd.Series, valid: pd.Series, upper: float = 0.6, lower: float = 0.4) -> pd.Series:
    """S4 (plan's model signal): long if P(RT > DA) > upper, short if < lower, else flat."""
    if not 0 <= lower <= upper <= 1:
        raise ValueError("need 0 <= lower <= upper <= 1")
    p = np.asarray(p_up, dtype=float)
    return _clean(np.where(p > upper, 1.0, np.where(p < lower, -1.0, 0.0)), valid)


def risk_filtered_short(pred_risk: pd.Series, valid: pd.Series, cutoff: float) -> pd.Series:
    """S5: short when predicted risk is below ``cutoff`` (a value of log(1+|spread|)), else flat."""
    r = np.asarray(pred_risk, dtype=float)
    return _clean(np.where(r < cutoff, -1.0, 0.0), valid)


def risk_sized_short(pred_risk: pd.Series, valid: pd.Series, reference: float) -> pd.Series:
    """S6: short with size min(1, ref / predicted |spread|): full size in calm hours, smaller in risky ones.

    ``reference`` is a predicted |spread| level in $/MWh (e.g. a percentile of training predictions).
    """
    typical_abs = np.expm1(np.asarray(pred_risk, dtype=float)).clip(min=1e-6)
    return _clean(-np.minimum(1.0, reference / typical_abs), valid)


def risk_cutoff(train_pred_risk: pd.Series | np.ndarray, percentile: float) -> float:
    """Convert a percentile of the *training-period* predicted-risk distribution into a cut-off."""
    return float(np.nanquantile(np.asarray(train_pred_risk, dtype=float), percentile / 100.0))
