"""Learned pre-solve conflict triage. Predictions never authorize or skip a solve.

Serving uses plain JSON coefficients and NumPy, never pickle or executable model
artifacts. Only pre-solve account data is available to feature extraction.
"""

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from lotwise.models import Account
from lotwise.optimizer import arrays

FEATURES = [
    "restricted_weight",
    "mandatory_net_gain_weight",
    "gain_budget_weight",
    "optional_loss_capacity_weight",
    "mandatory_gain_gap_weight",
    "over_cap_weight",
    "cash_reserve_gap_weight",
    "overweight_weight",
    "cash_weight",
    "minimum_cash_weight",
    "position_cap",
    "transaction_cost_rate",
]


def features(account: Account) -> np.ndarray:
    symbols, _prices, incidence, values, gains, nav = arrays(account)
    current = incidence @ values
    target = np.array([s.target_weight for s in account.securities])
    restricted = np.array([s in account.restricted_symbols for s in symbols])
    caps = np.where(restricted, 0, account.max_position_weight)
    sellable = current > np.minimum(target, caps) + 1e-10
    forced_lots = incidence.T @ restricted.astype(float)
    optional_lots = incidence.T @ (sellable & ~restricted).astype(float)
    forced_gain = float((values * gains * forced_lots).sum())
    loss_capacity = float(-(values * np.minimum(gains, 0) * optional_lots).sum())
    return np.array(
        [
            current[restricted].sum(),
            forced_gain,
            account.realized_gain_budget / nav,
            loss_capacity,
            forced_gain - loss_capacity - account.realized_gain_budget / nav,
            np.maximum(current - caps, 0).sum(),
            account.min_cash_weight - account.cash / nav,
            np.maximum(current - target, 0).sum(),
            account.cash / nav,
            account.min_cash_weight,
            account.max_position_weight,
            account.transaction_cost_bps / 10000,
        ],
        dtype=float,
    )


@lru_cache(maxsize=1)
def load_model() -> dict:
    return json.loads((Path(__file__).parent / "artifacts" / "triage_model.json").read_text())


def predict(account: Account) -> dict:
    model = load_model()
    if model["feature_names"] != FEATURES:
        raise ValueError("triage artifact feature schema mismatch")
    x = features(account)
    standardized = (x - np.array(model["mean"])) / np.array(model["scale"])
    contributions = standardized * np.array(model["coefficients"])
    logit = float(model["intercept"] + contributions.sum())
    score = float(1 / (1 + np.exp(-np.clip(logit, -40, 40))))
    ranked = np.argsort(np.abs(contributions))[::-1][:3]
    return {
        "model_version": model["version"],
        "conflict_score": score,
        "predicted_conflict": score >= 0.5,
        "outside_training_shape": len(account.securities) != 20 or len(account.lots) != 60,
        "label": "Potential constraint conflict"
        if score >= 0.5
        else "Lower predicted conflict risk",
        "drivers": [
            {
                "feature": FEATURES[i],
                "value": float(x[i]),
                "log_odds_contribution": float(contributions[i]),
                "direction": "raises" if contributions[i] > 0 else "lowers",
            }
            for i in ranked
        ],
        "evaluation": model["evaluation"],
        "note": "Learned on synthetic accounts. Score is not a guarantee or investment risk estimate; the optimizer and ledger checks decide feasibility.",
    }
