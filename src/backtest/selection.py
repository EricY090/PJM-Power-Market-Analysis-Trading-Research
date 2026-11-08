"""Signal selection on out-of-sample predictions (never the test period).

The selection set is 18 contiguous months of out-of-sample predictions:
* 2024 Q1..Q4 - each quarter predicted by models fit on all *earlier* training data;
* 2025 H1 (validation) - predicted by models fit on all of 2023-2024.

Risk thresholds are expressed as percentiles of each model's *own training-period*
predictions, so the same percentile rule transfers to the final refit models.

Pre-declared criterion: maximize the Sharpe-like ratio of daily net P&L (cost $0.50/MWh)
over the pooled 18 months, among parameter settings that trade at least ``MIN_COVERAGE`` of
hours; report per-block results to check the choice is stable rather than driven by one block.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.backtest import signals as sg
from src.backtest.engine import BASE_COST_PER_MWH, daily_pnl, hourly_pnl, max_drawdown, sharpe_like
from src.models.common import FeatureSpec, feature_spec
from src.models.selected import fit_selected, predict_selected
from src.models.split import usable, walk_forward_folds

MIN_COVERAGE = 0.25
GRID = {
    "S4_prob_threshold": [{"upper": u, "lower": round(1 - u, 2)} for u in (0.55, 0.60, 0.65, 0.70)],
    "S5_risk_filtered_short": [{"percentile": p} for p in (30, 40, 50, 60, 70, 80, 90)],
    "S6_risk_sized_short": [{"reference_percentile": p} for p in (10, 25, 50)],
}


@dataclass
class Block:
    name: str
    rows: pd.DataFrame            # out-of-sample rows (all hours, incl. unusable)
    preds: pd.DataFrame           # predictions for usable rows
    train_risk: np.ndarray        # in-sample predicted risk on the block's training rows


def build_selection_set(F: pd.DataFrame, spec: FeatureSpec | None = None) -> list[Block]:
    spec = spec or feature_spec()
    ok = usable(F, spec.columns)
    D = F[F["period"].isin(["train", "validation"])]
    plan = [(D.index[(D["period"] == "train") & D.index.isin(a)], D.index[D.index.isin(b)], f"{name}")
            for a, b, name in walk_forward_folds(D)]
    plan.append((D.index[D["period"] == "train"], D.index[D["period"] == "validation"], "2025-H1 (validation)"))
    blocks = []
    for fit_idx, eval_idx, name in plan:
        fit_rows = F.loc[fit_idx][ok.loc[fit_idx]]
        models = fit_selected(fit_rows, spec)
        rows = F.loc[eval_idx]
        use = rows[ok.loc[eval_idx]]
        blocks.append(Block(name, rows, predict_selected(models, use, spec),
                            predict_selected(models, fit_rows, spec)["pred_risk"].to_numpy()))
    return blocks


def positions(block: Block, signal: str, params: dict | None = None) -> pd.Series:
    """Position for every hour of the block (0 where no prediction is available)."""
    params = params or {}
    rows = block.rows
    valid = pd.Series(rows.index.isin(block.preds.index), index=rows.index)
    p = block.preds.reindex(rows.index)
    if signal == "S0_flat":
        return sg.flat(rows, valid)
    if signal == "S1_always_long":
        return sg.always_long(rows, valid)
    if signal == "S2_always_short":
        return sg.always_short(rows, valid)
    if signal == "S3_rule_based":
        return sg.rule_based(rows, valid)
    if signal == "S4_prob_threshold":
        return sg.prob_threshold(p["p_rt_above_da"], valid, params["upper"], params["lower"])
    if signal == "S5_risk_filtered_short":
        return sg.risk_filtered_short(p["pred_risk"], valid, sg.risk_cutoff(block.train_risk, params["percentile"]))
    if signal == "S6_risk_sized_short":
        ref = float(np.nanquantile(np.expm1(block.train_risk), params["reference_percentile"] / 100))
        return sg.risk_sized_short(p["pred_risk"], valid, ref)
    raise ValueError(signal)


def evaluate(blocks: list[Block], signal: str, params: dict | None = None,
             cost: float = BASE_COST_PER_MWH) -> tuple[pd.Series, pd.DataFrame]:
    """Pooled metrics and per-block metrics for one signal / parameter setting."""
    per_block, dailies, hourly = [], [], []
    for b in blocks:
        pos = positions(b, signal, params)
        pnl = hourly_pnl(pos, b.rows["da_rt_spread"], cost)
        d = daily_pnl(pnl, b.rows["market_date"])
        dailies.append(d); hourly.append(pd.DataFrame({"pos": pos, "pnl": pnl}))
        per_block.append({"block": b.name, "net_pnl": pnl.sum(), "sharpe_like": sharpe_like(d),
                          "coverage": (pos != 0).mean(), "mwh": pos.abs().sum()})
    daily = pd.concat(dailies).sort_index()
    h = pd.concat(hourly)
    traded = h["pos"] != 0
    pooled = pd.Series({
        "sharpe_like": sharpe_like(daily),
        "net_pnl": h["pnl"].sum(),
        "mwh_traded": h["pos"].abs().sum(),
        "pnl_per_mwh": h["pnl"].sum() / max(h["pos"].abs().sum(), 1e-9),
        "coverage": traded.mean(),
        "hit_rate": (h.loc[traded, "pnl"] > 0).mean() if traded.any() else np.nan,
        "max_drawdown": max_drawdown(daily),
        "worst_day": daily.min(),
        "blocks_positive": int(sum(r["net_pnl"] > 0 for r in per_block)),
    })
    return pooled, pd.DataFrame(per_block)


def run_grid(blocks: list[Block], cost: float = BASE_COST_PER_MWH) -> pd.DataFrame:
    candidates = [("S0_flat", {}), ("S1_always_long", {}), ("S2_always_short", {}), ("S3_rule_based", {})]
    candidates += [(s, p) for s, grid in GRID.items() for p in grid]
    rows = []
    for signal, params in candidates:
        pooled, _ = evaluate(blocks, signal, params, cost)
        rows.append({"signal": signal, "params": params, **pooled.to_dict()})
    return pd.DataFrame(rows)


def select(results: pd.DataFrame, min_coverage: float = MIN_COVERAGE) -> pd.Series:
    """Apply the pre-declared criterion: best pooled Sharpe-like with coverage >= min_coverage."""
    eligible = results[(results["coverage"] >= min_coverage) & results["sharpe_like"].notna()]
    return eligible.sort_values("sharpe_like", ascending=False).iloc[0]
