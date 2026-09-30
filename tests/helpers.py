"""Shared, cached fixtures: a small dataset and a model trained on it (a few seconds, built once)."""
from __future__ import annotations

from functools import lru_cache

from frauddetect.data import GenConfig, generate_transactions
from frauddetect.model import train

SMALL = GenConfig(n_accounts=600, n_legit=15_000, days=40, fraud_events=70, legit_bursts=60, seed=3)


@lru_cache(maxsize=1)
def small_df():
    return generate_transactions(SMALL)


@lru_cache(maxsize=1)
def small_model():
    """(artifact, report) trained on the small dataset."""
    return train(small_df(), verbose=False)


def make_txn(**over):
    base = {"txn_id": "T1", "timestamp": 1_770_000_000, "account_id": "A00001", "amount": 25.0,
            "merchant_id": "M00-001", "mcc": "grocery", "channel": "pos", "device_id": "D00001-0",
            "country": "IN", "lat": 13.08, "lon": 80.27}
    base.update(over)
    return base
