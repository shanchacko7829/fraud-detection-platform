"""Point-in-time feature computation shared by training, streaming and the API.

The single most important design rule of this project: **the exact same code path
computes features offline (training) and online (serving)**. That removes
train/serve skew, one of the most common causes of ML systems that look great in a
notebook and fail in production.

``FeatureStore.compute`` only *reads* state (features describe the account's past),
and ``FeatureStore.update`` folds the transaction in afterwards, so a transaction can
never see itself or the future (no target/temporal leakage).
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
import pandas as pd

from .config import CHANNELS, MCC

FEATURES = [
    "amount", "log_amount", "amount_z", "hour", "is_night",
    "cnt_1m", "cnt_10m", "cnt_1h", "cnt_24h", "sum_1h", "distinct_merchants_1h",
    "secs_since_last", "new_device", "device_n_accounts", "new_country",
    "dist_km_last", "speed_kmh_last", "mcc_code", "channel_code",
]
CATEGORICAL_FEATURES = ["mcc_code", "channel_code"]

# Correlated features are explained together, so reasons read like an analyst wrote them.
FEATURE_GROUPS = {
    "amount": ["amount", "log_amount", "amount_z"],
    "velocity": ["cnt_1m", "cnt_10m", "cnt_1h", "cnt_24h", "sum_1h",
                 "distinct_merchants_1h", "secs_since_last"],
    "device": ["new_device", "device_n_accounts"],
    "geo": ["new_country", "dist_km_last", "speed_kmh_last"],
    "time": ["hour", "is_night"],
    "context": ["mcc_code", "channel_code"],
}

_MCC_INDEX = {m: i for i, m in enumerate(MCC)}
_CH_INDEX = {c: i for i, c in enumerate(CHANNELS)}
_DAY = 86_400.0
_MAX_EVENTS = 2_000  # per-account cap so a bot flood cannot exhaust memory
_NAN = float("nan")

TXN_FIELDS = ["timestamp", "account_id", "amount", "merchant_id", "mcc", "channel",
              "device_id", "country", "lat", "lon"]


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi, dlmb = p2 - p1, math.radians(lon2 - lon1)
    h = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * 6371.0 * math.asin(min(1.0, math.sqrt(h)))


@dataclass
class _AccountState:
    events: deque = field(default_factory=lambda: deque(maxlen=_MAX_EVENTS))  # (ts, amount, merchant)
    n: int = 0
    mean: float = 0.0   # Welford running mean / M2 of log1p(amount)
    m2: float = 0.0
    devices: set = field(default_factory=set)
    countries: set = field(default_factory=set)
    last_ts: float | None = None
    last_lat: float = 0.0
    last_lon: float = 0.0


class FeatureStore:
    """In-memory online feature store (a Redis hash per account in production)."""

    def __init__(self) -> None:
        self._accounts: dict[str, _AccountState] = {}
        self._device_accounts: dict[str, set[str]] = {}

    def __len__(self) -> int:
        return len(self._accounts)

    # -- read path -----------------------------------------------------------------
    def compute(self, txn: Mapping) -> dict[str, float]:
        ts = float(txn["timestamp"])
        amount = float(txn["amount"])
        acc = txn["account_id"]
        st = self._accounts.get(acc)
        log_amt = math.log1p(amount)

        # Deliberately no "account age" feature: in a simulation that starts cold it only encodes the
        # calendar and shows up as fake drift. (In production, account age is worth adding.)
        if st is not None and st.n >= 3:
            std = math.sqrt(st.m2 / (st.n - 1))
            z = (log_amt - st.mean) / max(std, 0.25)   # floor avoids exploding z on flat histories
        else:
            z = 0.0

        c1 = c10 = c60 = c24 = 0
        s60 = 0.0
        merchants: set[str] = set()
        if st is not None:
            for t, a, m in st.events:
                dt = max(ts - t, 0.0)          # tolerate slightly out-of-order events
                if dt <= _DAY:
                    c24 += 1
                    if dt <= 3600:
                        c60 += 1
                        s60 += a
                        merchants.add(m)
                        if dt <= 600:
                            c10 += 1
                            if dt <= 60:
                                c1 += 1

        if st is not None and st.last_ts is not None:
            secs = max(ts - st.last_ts, 0.0)
            dist = haversine_km(st.last_lat, st.last_lon, float(txn["lat"]), float(txn["lon"]))
            speed = min(dist / (max(secs, 1.0) / 3600.0), 20_000.0)
        else:
            secs = dist = speed = _NAN

        device = txn["device_id"]
        accs = self._device_accounts.get(device)
        dev_n = (len(accs) - (1 if acc in accs else 0)) if accs else 0
        hour = int((ts % _DAY) // 3600)

        return {
            "amount": amount, "log_amount": log_amt, "amount_z": z,
            "hour": float(hour), "is_night": 1.0 if hour < 6 else 0.0,
            "cnt_1m": float(c1), "cnt_10m": float(c10), "cnt_1h": float(c60), "cnt_24h": float(c24),
            "sum_1h": s60, "distinct_merchants_1h": float(len(merchants)),
            "secs_since_last": secs,
            "new_device": 1.0 if (st is None or device not in st.devices) else 0.0,
            "device_n_accounts": float(dev_n),
            "new_country": 1.0 if (st is None or txn["country"] not in st.countries) else 0.0,
            "dist_km_last": dist, "speed_kmh_last": speed,
            "mcc_code": float(_MCC_INDEX[txn["mcc"]]), "channel_code": float(_CH_INDEX[txn["channel"]]),
        }

    # -- write path ----------------------------------------------------------------
    def update(self, txn: Mapping) -> None:
        ts = float(txn["timestamp"])
        acc = txn["account_id"]
        st = self._accounts.setdefault(acc, _AccountState())
        amount = float(txn["amount"])
        st.events.append((ts, amount, txn["merchant_id"]))
        newest = max(ts, st.last_ts or ts)
        while st.events and newest - st.events[0][0] > _DAY:
            st.events.popleft()
        x = math.log1p(amount)
        st.n += 1
        delta = x - st.mean
        st.mean += delta / st.n
        st.m2 += delta * (x - st.mean)
        st.devices.add(txn["device_id"])
        st.countries.add(txn["country"])
        if st.last_ts is None or ts >= st.last_ts:   # a late event must not rewind "last seen"
            st.last_ts, st.last_lat, st.last_lon = ts, float(txn["lat"]), float(txn["lon"])
        self._device_accounts.setdefault(txn["device_id"], set()).add(acc)

    def features_and_update(self, txn: Mapping) -> dict[str, float]:
        feats = self.compute(txn)
        self.update(txn)
        return feats


def compute_features(df: pd.DataFrame, store: FeatureStore | None = None) -> tuple[pd.DataFrame, FeatureStore]:
    """Replay ``df`` (must be time-ordered) through the store; returns features + final store."""
    # NB: `store or ...` would be a bug: an empty FeatureStore is falsy (len 0) and would be replaced.
    store = store if store is not None else FeatureStore()
    records = df[TXN_FIELDS].to_dict("records")
    rows = [store.features_and_update(r) for r in records]
    return pd.DataFrame(rows, columns=FEATURES, index=df.index), store
