"""Per-transaction explanations via grouped feature occlusion.

For each feature *group* we replace the group's values with a typical (reference)
value and measure how much the model's log-odds drop. A large drop means "this
group is what pushed the score up". Grouping correlated features (e.g. the four
velocity counters) avoids the credit-splitting problem of single-feature
occlusion and gives analyst-friendly reasons.

This is intentionally simple and dependency-free. It is *not* Shapley values -
it ignores feature interactions across groups - and the README says so. In a
stack with SHAP available, ``TreeExplainer`` is the drop-in upgrade.
"""
from __future__ import annotations

import math

import numpy as np

from .features import CATEGORICAL_FEATURES, FEATURE_GROUPS, FEATURES
from .config import CHANNELS, MCC


def reference_values(X: np.ndarray, features: list[str] = FEATURES) -> dict[str, float]:
    """Typical value per feature: median for numeric, mode for categorical."""
    ref = {}
    for i, f in enumerate(features):
        col = X[:, i]
        col = col[~np.isnan(col)]
        if f in CATEGORICAL_FEATURES:
            vals, counts = np.unique(col, return_counts=True)
            ref[f] = float(vals[np.argmax(counts)])
        else:
            ref[f] = float(np.median(col)) if len(col) else 0.0
    return ref


class Explainer:
    def __init__(self, model, reference: dict[str, float], features: list[str] = FEATURES, groups=FEATURE_GROUPS):
        self.model = model
        idx = {f: i for i, f in enumerate(features)}
        self.groups = list(groups)
        self.cols = {g: [idx[f] for f in fs] for g, fs in groups.items()}
        self.ref = np.array([reference[f] for f in features], dtype=float)

    def contributions(self, X: np.ndarray) -> np.ndarray:
        """Log-odds contribution per group, shape (n_rows, n_groups). Positive = raises risk."""
        X = np.atleast_2d(X)
        base = self.model.decision_function(X)
        out = np.zeros((len(X), len(self.groups)))
        for j, g in enumerate(self.groups):
            Xg = X.copy()
            Xg[:, self.cols[g]] = self.ref[self.cols[g]]
            out[:, j] = base - self.model.decision_function(Xg)
        return out

    def reasons(self, feats: dict[str, float], contribs: np.ndarray, top_k: int = 3, min_contrib: float = 0.3) -> list[dict]:
        order = np.argsort(-contribs)[:top_k]
        result = []
        for j in order:
            if contribs[j] < min_contrib:
                break
            g = self.groups[j]
            result.append({"factor": g, "impact": round(float(contribs[j]), 2), "detail": describe(g, feats)})
        return result


def describe(group: str, f: dict[str, float]) -> str:
    """Human-readable sentence for a factor group, built from the raw feature values."""
    if group == "velocity":
        return f"{int(f['cnt_10m'])} transactions in the last 10 min ({int(f['cnt_1h'])} in the last hour, {int(f['distinct_merchants_1h'])} merchants)"
    if group == "amount":
        return f"amount {f['amount']:.2f} is {f['amount_z']:+.1f} std devs from this account's norm"
    if group == "device":
        parts = []
        if f["new_device"]:
            parts.append("device never seen on this account")
        if f["device_n_accounts"] >= 1:
            parts.append(f"device also used on {int(f['device_n_accounts'])} other accounts")
        return "; ".join(parts) or "device profile unusual"
    if group == "geo":
        parts = []
        if f["new_country"]:
            parts.append("first transaction from this country")
        sp = f["speed_kmh_last"]
        if not math.isnan(sp) and sp > 500:
            parts.append(f"implied travel speed {sp:,.0f} km/h since last transaction")
        return "; ".join(parts) or "location unusual for this account"
    if group == "time":
        return f"transaction at {int(f['hour']):02d}:00 UTC"
    if group == "context":
        return f"{MCC[int(f['mcc_code'])]} purchase via {CHANNELS[int(f['channel_code'])]}"
    return group
