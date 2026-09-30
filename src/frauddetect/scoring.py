"""Online scorer: feature store + model + explanations + monitoring behind one call."""
from __future__ import annotations

import threading
import time
from collections import Counter, OrderedDict, deque
from typing import Mapping

import numpy as np

from .drift import DriftMonitor
from .explain import Explainer
from .features import FeatureStore, compute_features


def _percentiles(values) -> dict:
    if not values:
        return {}
    a = np.asarray(values)
    return {"p50": round(float(np.percentile(a, 50)), 3), "p95": round(float(np.percentile(a, 95)), 3),
            "p99": round(float(np.percentile(a, 99)), 3), "max": round(float(a.max()), 3)}


class Scorer:
    """Thread-safe scorer.

    * State (feature store) is mutated on every call, so calls are serialised with a lock.
      Scaling out means moving that state to Redis so workers become stateless.
    * Scoring is **idempotent per txn_id**: a message redelivered by the broker (at-least-once
      delivery) returns the cached decision and does *not* update state twice.
    """

    def __init__(self, artifact: Mapping, cache_size: int = 100_000, drift_window: int = 5_000):
        self.model = artifact["model"]
        self.features: list[str] = artifact["features"]
        self.thr_review = artifact["thresholds"]["review"]
        self.thr_block = artifact["thresholds"]["block"]
        self.version = artifact["meta"]["model_version"]
        self.explainer = Explainer(self.model, artifact["reference_values"], self.features)
        self.store = FeatureStore()
        self.monitor = DriftMonitor(artifact["drift_reference"], self.features, window=drift_window)
        self._lock = threading.Lock()
        self._seen: OrderedDict[str, dict] = OrderedDict()
        self._cache_size = cache_size
        self._latency: deque = deque(maxlen=20_000)
        self._decisions: Counter = Counter()
        self.started_at = time.time()

    def warm_up(self, df) -> int:
        """Replay historical transactions into the feature store without scoring them."""
        with self._lock:
            compute_features(df, self.store)
        return len(df)

    def decide(self, score: float) -> str:
        return "decline" if score >= self.thr_block else "review" if score >= self.thr_review else "approve"

    def score(self, txn: Mapping) -> dict:
        t0 = time.perf_counter()
        with self._lock:
            txn_id = txn.get("txn_id")
            if txn_id is not None and txn_id in self._seen:
                return {**self._seen[txn_id], "duplicate": True}

            feats = self.store.features_and_update(txn)
            x = np.array([[feats[f] for f in self.features]], dtype=float)
            raw = float(self.model.decision_function(x)[0])
            risk = 1.0 / (1.0 + np.exp(-raw))
            decision = self.decide(risk)
            reasons = []
            if decision != "approve":                     # explain only what a human will look at
                reasons = self.explainer.reasons(feats, self.explainer.contributions(x)[0])
            self.monitor.observe([feats[f] for f in self.features], risk)

            result = {"txn_id": txn_id, "risk_score": round(float(risk), 5), "decision": decision,
                      "reasons": reasons, "model_version": self.version, "duplicate": False}
            if txn_id is not None:
                self._seen[txn_id] = result
                if len(self._seen) > self._cache_size:
                    self._seen.popitem(last=False)
            self._decisions[decision] += 1

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self._latency.append(elapsed_ms)
        return {**result, "latency_ms": round(elapsed_ms, 3)}

    def metrics(self) -> dict:
        return {
            "model_version": self.version,
            "uptime_s": round(time.time() - self.started_at, 1),
            "scored": int(sum(self._decisions.values())),
            "decisions": dict(self._decisions),
            "latency_ms": _percentiles(list(self._latency)),
            "accounts_tracked": len(self.store),
            "drift": self.monitor.report(),
        }
