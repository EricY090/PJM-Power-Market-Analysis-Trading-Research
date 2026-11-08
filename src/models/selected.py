"""The model choices made in Phase 4, in one place, so later phases reuse exactly these.

Hyperparameters were chosen on the 2024 quarterly walk-forward folds (training data only):
* direction:  logistic regression, C = 0.01 (best mean fold AUC; alternatives within ±0.005)
* magnitude:  ridge, alpha = 1000, fit on the spread clipped to ±$100 (no skill vs constant; kept as reference)
* risk:       ridge on log(1 + |spread|), alpha = 1000 (best mean fold Spearman)
"""

from __future__ import annotations

import pandas as pd
from sklearn.pipeline import Pipeline

from src.models import baseline as bl
from src.models import classifier as clf
from src.models.common import FeatureSpec, feature_spec, to_model_frame

CHOICES = {
    "direction": {"model": "logistic", "C": 0.01},
    "magnitude": {"model": "ridge", "alpha": 1000.0, "target_clip": bl.TARGET_CLIP},
    "risk": {"model": "ridge", "alpha": 1000.0, "target": "log1p(|spread|)"},
}


def fit_selected(train: pd.DataFrame, spec: FeatureSpec | None = None) -> dict[str, Pipeline]:
    """Fit the three selected models on ``train`` (rows must be usable: no missing features)."""
    spec = spec or feature_spec()
    X = to_model_frame(train, spec)
    s = train["da_rt_spread"].to_numpy()
    return {
        "direction": clf.make_logit(spec, C=CHOICES["direction"]["C"]).fit(X, train["target_rt_above_da"]),
        "magnitude": bl.make_ridge(spec, alpha=CHOICES["magnitude"]["alpha"]).fit(X, bl.clip_target(s)),
        "risk": bl.make_ridge(spec, alpha=CHOICES["risk"]["alpha"]).fit(X, bl.risk_target(s)),
    }


def predict_selected(models: dict[str, Pipeline], df: pd.DataFrame, spec: FeatureSpec | None = None) -> pd.DataFrame:
    """Predictions for ``df``: P(RT > DA), expected spread, and predicted log(1 + |spread|)."""
    spec = spec or feature_spec()
    X = to_model_frame(df, spec)
    return pd.DataFrame({
        "p_rt_above_da": models["direction"].predict_proba(X)[:, 1],
        "pred_spread": models["magnitude"].predict(X),
        "pred_risk": models["risk"].predict(X),
    }, index=df.index)
