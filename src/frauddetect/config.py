"""Shared constants: vocabularies, geography and defaults used across the package."""
from __future__ import annotations

SEED = 42

# Merchant category codes (simplified). Order matters: the index is the categorical code.
MCC = [
    "grocery", "restaurant", "fuel", "utilities", "healthcare", "travel",
    "electronics", "luxury", "gambling", "crypto", "digital_goods", "atm_cash",
]
CHANNELS = ["pos", "ecom", "atm"]

# Rough country centroids (lat, lon) used to place transactions on the globe.
COUNTRIES = {
    "IN": (20.6, 78.9),
    "US": (39.8, -98.6),
    "GB": (54.0, -2.0),
    "DE": (51.2, 10.4),
    "SG": (1.35, 103.8),
    "AE": (24.0, 54.0),
    "BR": (-10.8, -52.9),
    "NG": (9.1, 8.7),
    "RU": (61.5, 105.3),
    "CN": (35.9, 104.2),
}

# Decision policy: FPR targets used to pick thresholds on the validation window.
REVIEW_FPR = 0.005   # send to manual review / step-up auth
BLOCK_FPR = 0.0005   # decline outright

# Fraud labels arrive late (chargebacks take days), so the training window stops
# this many days before the validation window starts.
LABEL_DELAY_DAYS = 2.0

# Velocity/history features are meaningless until accounts have some history, so the first
# days of the timeline only build feature-store state and are excluded from training.
BURN_IN_DAYS = 7.0

DEFAULT_DATA_PATH = "data/transactions.csv.gz"
DEFAULT_MODEL_PATH = "models/fraud_model.joblib"
