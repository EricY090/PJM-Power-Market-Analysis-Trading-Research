"""Freeze the trading strategy before the test period is touched, and apply it later.

``freeze`` (run once, at the end of Phase 5):
1. runs the signal selection on the out-of-sample selection set (2024 walk-forward + 2025 H1);
2. applies the pre-declared criterion to pick the primary strategy, and the best eligible
   setting inside every other signal family (so each hypothesis is also judged on test);
3. refits the Phase 4 models on train + validation (2023-01 -> 2025-06);
4. converts percentile rules into numeric cut-offs on the refit models' training predictions;
5. writes everything to ``config/strategy.yaml`` (to be committed before Phase 6).

``positions_for`` turns the frozen file + refit models into positions for any period.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.backtest import selection as sel
from src.backtest import signals as sg
from src.backtest.engine import BASE_COST_PER_MWH
from src.config import PROJECT_ROOT
from src.models.common import feature_spec
from src.models.selected import CHOICES, fit_selected, predict_selected
from src.models.split import usable

STRATEGY_PATH = PROJECT_ROOT / "config" / "strategy.yaml"
FIT_PERIODS = ("train", "validation")


def _family_best(results: pd.DataFrame, family: str) -> dict:
    fam = results[(results["signal"] == family) & (results["coverage"] >= sel.MIN_COVERAGE)]
    return fam.sort_values("sharpe_like", ascending=False).iloc[0]["params"] if len(fam) else {}


def fit_final_models(F: pd.DataFrame):
    spec = feature_spec()
    rows = F[F["period"].isin(FIT_PERIODS) & usable(F, spec.columns)]
    models = fit_selected(rows, spec)
    in_sample_risk = predict_selected(models, rows, spec)["pred_risk"].to_numpy()
    return models, in_sample_risk, rows


def freeze(F: pd.DataFrame, path: Path = STRATEGY_PATH) -> dict[str, Any]:
    if (F["period"] == "holdout").any():
        raise ValueError("the holdout must not be loaded when freezing")
    blocks = sel.build_selection_set(F)
    results = sel.run_grid(blocks)
    best = sel.select(results)
    models, in_sample_risk, rows = fit_final_models(F)

    s5 = _family_best(results, "S5_risk_filtered_short")
    s6 = _family_best(results, "S6_risk_sized_short")
    s4 = _family_best(results, "S4_prob_threshold")
    signals = {
        "S0_flat": {},
        "S1_always_long": {},
        "S2_always_short": {},
        "S3_rule_based": {"flat_load_regimes": list(sg.RULE_FLAT_LOAD_REGIMES), "flat_hours": list(sg.RULE_FLAT_HOURS)},
        "S4_prob_threshold": {k: float(v) for k, v in s4.items()},
        "S5_risk_filtered_short": {"percentile": float(s5["percentile"]),
                                   "cutoff_log1p_abs_spread": sg.risk_cutoff(in_sample_risk, s5["percentile"])},
        "S6_risk_sized_short": {"reference_percentile": float(s6["reference_percentile"]),
                                "reference_abs_spread": float(np.nanquantile(np.expm1(in_sample_risk), s6["reference_percentile"] / 100))},
    }
    # selection results are statistics of PJM data -> kept in git-ignored reports/, not in the committed config
    results_path = PROJECT_ROOT / "reports" / "phase5_selection_results.csv"
    results.assign(params=results["params"].astype(str)).to_csv(results_path, index=False)
    strategy = {
        "frozen_at": dt.date.today().isoformat(),
        "primary_signal": best["signal"],
        "primary_params": {k: float(v) for k, v in best["params"].items()},
        "position_convention": "+1 = long spread (earns RT - DA, ~DEC); -1 = short spread (earns DA - RT, ~INC); 1 MW per hour",
        "decision_time": "10:30 EPT on D-1; tradable features only (see src/data/features.py)",
        "cost_per_mwh": BASE_COST_PER_MWH,
        "selection": {
            "set": "out-of-sample predictions: 2024 Q1-Q4 walk-forward folds + 2025 H1 validation (test never used)",
            "criterion": f"max Sharpe-like of daily net P&L, coverage >= {sel.MIN_COVERAGE:.0%} of hours",
            "grid": {k: v for k, v in sel.GRID.items()},
            "results_file": "reports/phase5_selection_results.csv (git-ignored: derived from PJM data)",
        },
        "models": {"fit_periods": list(FIT_PERIODS), "fit_rows": int(len(rows)),
                   "fit_window": f"{rows['market_date'].min().date()} -> {rows['market_date'].max().date()}",
                   "choices": CHOICES},
        "signals": signals,
    }
    path.write_text(yaml.safe_dump(strategy, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return strategy


def load_strategy(path: Path = STRATEGY_PATH) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def positions_for(strategy: dict, signal: str, rows: pd.DataFrame, models) -> pd.Series:
    """Positions for ``rows`` (any period) from the frozen strategy and refit models."""
    spec = feature_spec()
    ok = usable(rows, spec.columns)
    preds = predict_selected(models, rows[ok], spec).reindex(rows.index)
    p = strategy["signals"][signal]
    if signal == "S0_flat":
        return sg.flat(rows, ok)
    if signal == "S1_always_long":
        return sg.always_long(rows, ok)
    if signal == "S2_always_short":
        return sg.always_short(rows, ok)
    if signal == "S3_rule_based":
        return sg.rule_based(rows, ok, tuple(p["flat_load_regimes"]), tuple(p["flat_hours"]))
    if signal == "S4_prob_threshold":
        return sg.prob_threshold(preds["p_rt_above_da"], ok, p["upper"], p["lower"])
    if signal == "S5_risk_filtered_short":
        return sg.risk_filtered_short(preds["pred_risk"], ok, p["cutoff_log1p_abs_spread"])
    if signal == "S6_risk_sized_short":
        return sg.risk_sized_short(preds["pred_risk"], ok, p["reference_abs_spread"])
    raise ValueError(signal)


if __name__ == "__main__":
    import argparse
    import warnings

    from src.config import load_config

    ap = argparse.ArgumentParser(description="Freeze a strategy: run the selection, refit models, save the result.")
    ap.add_argument("--out", type=Path, default=STRATEGY_PATH, help="strategy file to write")
    args = ap.parse_args()
    warnings.filterwarnings("ignore", category=UserWarning)
    F = pd.read_parquet(load_config()["paths"]["processed"] / "features.parquet")
    s = freeze(F, args.out)
    print(f"saved {args.out}")
    print("primary:", s["primary_signal"], s["primary_params"])
    print(yaml.safe_dump(s["signals"], sort_keys=False))
