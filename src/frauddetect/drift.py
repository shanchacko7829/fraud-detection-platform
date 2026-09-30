"""Population Stability Index (PSI) drift monitoring.

PSI compares the distribution a feature had at training time with what the live
service is seeing now. Rule of thumb used across banking model-risk teams:
< 0.10 stable, 0.10-0.25 moderate shift (watch), > 0.25 significant shift (investigate).
"""
from __future__ import annotations

from collections import deque

import numpy as np

_LOW_CARDINALITY = 12


def _cuts(x: np.ndarray, n_bins: int) -> np.ndarray:
    """Inner cut points: midpoints for low-cardinality features, quantiles otherwise."""
    x = x[~np.isnan(x)]
    uniq = np.unique(x)
    if len(uniq) <= 1:
        return np.array([])
    if len(uniq) <= _LOW_CARDINALITY:
        return (uniq[:-1] + uniq[1:]) / 2.0
    return np.unique(np.quantile(x, np.linspace(0, 1, n_bins + 1)[1:-1]))


def _proportions(x: np.ndarray, cuts: np.ndarray) -> np.ndarray:
    nan = np.isnan(x)
    idx = np.digitize(x[~nan], cuts, right=True)
    counts = np.bincount(idx, minlength=len(cuts) + 1).astype(float)
    counts = np.append(counts, float(nan.sum()))          # dedicated bin for missing values
    return counts / max(counts.sum(), 1.0)


def psi_from_props(ref: np.ndarray, cur: np.ndarray, eps: float = 1e-4) -> float:
    r, c = np.clip(ref, eps, None), np.clip(cur, eps, None)
    return float(np.sum((c - r) * np.log(c / r)))


def build_reference(columns: dict[str, np.ndarray], n_bins: int = 10) -> dict[str, dict]:
    """Freeze bin edges + reference proportions for each named column."""
    ref = {}
    for name, x in columns.items():
        x = np.asarray(x, dtype=float)
        cuts = _cuts(x, n_bins)
        ref[name] = {"cuts": cuts, "props": _proportions(x, cuts)}
    return ref


def psi(reference: dict, name: str, values: np.ndarray) -> float:
    r = reference[name]
    return psi_from_props(r["props"], _proportions(np.asarray(values, dtype=float), r["cuts"]))


def status(value: float) -> str:
    return "stable" if value < 0.10 else "watch" if value < 0.25 else "drift"


def drift_report(reference: dict, columns: dict[str, np.ndarray]) -> dict:
    per = {n: psi(reference, n, columns[n]) for n in reference if n in columns}
    worst = max(per, key=per.get) if per else None
    return {
        "psi": {n: round(v, 4) for n, v in sorted(per.items(), key=lambda kv: -kv[1])},
        "max_psi": round(per[worst], 4) if worst else 0.0,
        "max_feature": worst,
        "status": status(per[worst]) if worst else "stable",
    }


class DriftMonitor:
    """Rolling window over live traffic, compared against the training reference."""

    def __init__(self, reference: dict, feature_names: list[str], window: int = 5_000, min_samples: int = 500):
        self.reference = reference
        self.feature_names = feature_names
        self.min_samples = min_samples
        self._rows: deque = deque(maxlen=window)   # each row: feature values + score

    def observe(self, feature_values: list[float], score: float) -> None:
        self._rows.append([*feature_values, score])

    def report(self) -> dict:
        if len(self._rows) < self.min_samples:
            return {"status": "insufficient_data", "samples": len(self._rows), "min_samples": self.min_samples}
        arr = np.array(self._rows, dtype=float)
        cols = {n: arr[:, i] for i, n in enumerate(self.feature_names)}
        cols["score"] = arr[:, -1]
        out = drift_report(self.reference, cols)
        out["samples"] = len(self._rows)
        return out
