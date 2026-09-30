"""Synthetic card-transaction generator with realistic fraud typologies.

Real card data is confidential, so this module simulates it. The goal is not to
be "easy to classify" but to include the awkward cases real systems face:

* legitimate travellers (new country + occasionally a new device),
* legitimate bursts of quick purchases (false-positive bait for velocity rules),
* fraud that looks completely normal (``low_and_slow`` on a compromised device),
* attacker devices reused across many victim accounts (a ring signal).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import CHANNELS, COUNTRIES, MCC, SEED

BASE_TS = 1_767_225_600  # 2026-01-01T00:00:00Z

_CODES = list(COUNTRIES)
_LAT = np.array([COUNTRIES[c][0] for c in _CODES])
_LON = np.array([COUNTRIES[c][1] for c in _CODES])
_HOME_P = np.array([0.30, 0.20, 0.10, 0.08, 0.07, 0.07, 0.06, 0.04, 0.04, 0.04])

_M = {m: i for i, m in enumerate(MCC)}
_ATM = _M["atm_cash"]
# Popularity, log-amount offset and e-commerce probability, aligned with MCC order.
_POP = np.array([.22, .18, .10, .08, .05, .05, .07, .02, .02, .01, .10, .10])
_AMT_OFFSET = np.array([0.0, 0.0, 0.2, 0.3, 0.4, 0.9, 1.0, 1.3, 0.2, 0.6, -0.7, 0.6])
_ECOM_P = np.array([.05, .05, 0.0, .6, .2, .7, .5, .3, .95, 1.0, 1.0, 0.0])

# Shared clock (UTC) for everyone: quiet at night, busy in the evening.
_HOUR_W = np.array([1, 1, 1, 1, 1, 2, 4, 6, 8, 9, 9, 10,
                    11, 10, 9, 9, 10, 11, 12, 12, 10, 7, 4, 2], dtype=float)
_HOUR_P = _HOUR_W / _HOUR_W.sum()

COLUMNS = ["timestamp", "account_id", "amount", "merchant_id", "mcc", "channel",
           "device_id", "country", "lat", "lon", "is_fraud", "fraud_type"]


@dataclass(frozen=True)
class GenConfig:
    n_accounts: int = 5_000
    n_legit: int = 200_000
    days: int = 60
    fraud_events: int = 450
    legit_bursts: int = 500
    seed: int = SEED
    # Knobs used by the drift demo to shift the population.
    amount_scale: float = 1.0
    ecom_shift: float = 0.0


def _ids(prefix: str, values: np.ndarray, width: int) -> np.ndarray:
    return np.char.add(prefix, np.char.zfill(values.astype(str), width))


def _make_accounts(rng: np.random.Generator, cfg: GenConfig) -> dict[str, np.ndarray]:
    n = cfg.n_accounts
    home = rng.choice(len(_CODES), size=n, p=_HOME_P)
    has_trip = rng.random(n) < 0.25
    dest = rng.integers(0, len(_CODES), n)
    dest = np.where(dest == home, (dest + 1) % len(_CODES), dest)
    return {
        "home": home,
        "lat": _LAT[home] + rng.normal(0, 1.0, n),
        "lon": _LON[home] + rng.normal(0, 1.0, n),
        "mu": rng.normal(3.7, 0.55, n),            # mean log-amount (~ 40 currency units)
        "sigma": rng.uniform(0.45, 0.9, n),
        "activity": rng.gamma(2.0, 1.0, n),
        "n_dev": rng.integers(1, 4, n),
        "pref": rng.dirichlet(_POP * 30, size=n),  # per-account category preferences
        "trip_start": np.where(has_trip, rng.integers(0, max(cfg.days - 9, 1), n), -1),
        "trip_len": rng.integers(3, 10, n),
        "trip_dest": dest,
    }


def _legit_transactions(rng: np.random.Generator, cfg: GenConfig, acc) -> pd.DataFrame:
    n = cfg.n_legit
    p = acc["activity"] / acc["activity"].sum()
    a = rng.choice(cfg.n_accounts, size=n, p=p)
    day = rng.integers(0, cfg.days, n)
    hour = rng.choice(24, size=n, p=_HOUR_P)
    ts = BASE_TS + day * 86_400 + hour * 3_600 + rng.integers(0, 3_600, n)

    cum = acc["pref"].cumsum(axis=1)
    cat = np.minimum((rng.random(n)[:, None] > cum[a]).sum(axis=1), len(MCC) - 1)
    ecom_p = np.where(cat == _ATM, 0.0, np.clip(_ECOM_P[cat] + cfg.ecom_shift, 0, 1))
    channel = np.where(cat == _ATM, 2, np.where(rng.random(n) < ecom_p, 1, 0))
    amount = np.exp(acc["mu"][a] + _AMT_OFFSET[cat] + acc["sigma"][a] * rng.standard_normal(n))
    amount = np.round(np.clip(amount * cfg.amount_scale, 0.5, None), 2)

    # Trips: while travelling the account transacts from the destination country.
    ts0 = acc["trip_start"][a]
    away = (ts0 >= 0) & (day >= ts0) & (day < ts0 + acc["trip_len"][a])
    ctry = np.where(away, acc["trip_dest"][a], acc["home"][a])
    lat = np.where(away, _LAT[ctry], acc["lat"][a]) + rng.normal(0, 0.25, n)
    lon = np.where(away, _LON[ctry], acc["lon"][a]) + rng.normal(0, 0.25, n)

    slot = np.minimum(rng.choice(3, size=n, p=[0.75, 0.18, 0.07]), acc["n_dev"][a] - 1)
    acc_s = np.char.zfill(a.astype(str), 5)
    known = np.char.add(np.char.add("D", acc_s), np.char.add("-", slot.astype(str)))
    odd_dev = np.char.add(np.char.add("N", acc_s), np.char.add("-", rng.integers(0, 2, n).astype(str)))
    device = np.where(rng.random(n) < np.where(away, 0.12, 0.004), odd_dev, known)

    merchant = np.char.add(np.char.add("M", np.char.zfill(cat.astype(str), 2)),
                           np.char.add("-", np.char.zfill(rng.integers(0, 150, n).astype(str), 3)))
    return pd.DataFrame({
        "timestamp": ts, "account_id": _ids("A", a, 5), "amount": amount,
        "merchant_id": merchant, "mcc": np.array(MCC)[cat], "channel": np.array(CHANNELS)[channel],
        "device_id": device, "country": np.array(_CODES)[ctry], "lat": lat, "lon": lon,
        "is_fraud": 0, "fraud_type": "none",
    })


def _row(ts, a, amount, cat, channel, device, ctry, lat, lon, fraud, kind, rng):
    return {
        "timestamp": int(ts), "account_id": f"A{a:05d}", "amount": round(max(float(amount), 0.5), 2),
        "merchant_id": f"M{cat:02d}-{int(rng.integers(0, 150)):03d}", "mcc": MCC[cat],
        "channel": channel, "device_id": device, "country": _CODES[ctry],
        "lat": float(lat), "lon": float(lon), "is_fraud": fraud, "fraud_type": kind,
    }


def _place(acc, a, ctry, rng):
    """Location for a transaction in country index ``ctry`` (home city if it is the home country)."""
    if ctry == int(acc["home"][a]):
        return acc["lat"][a] + rng.normal(0, 0.25), acc["lon"][a] + rng.normal(0, 0.25)
    return _LAT[ctry] + rng.normal(0, 0.4), _LON[ctry] + rng.normal(0, 0.4)


def _legit_bursts(rng: np.random.Generator, cfg: GenConfig, acc) -> pd.DataFrame:
    """Legitimate quick sequences of purchases (shopping trips) - velocity false-positive bait."""
    rows = []
    for _ in range(cfg.legit_bursts):
        a = int(rng.integers(0, cfg.n_accounts))
        t0 = BASE_TS + int(rng.integers(0, cfg.days)) * 86_400 + int(rng.choice(24, p=_HOUR_P)) * 3_600 + int(rng.integers(0, 3_600))
        k = int(rng.integers(2, 7))
        gaps = rng.uniform(20, 300, k).cumsum()
        cats = rng.choice(len(MCC), size=k, p=acc["pref"][a])
        home = int(acc["home"][a])
        for g, c in zip(gaps, cats):
            c = int(c)
            amt = np.exp(acc["mu"][a] + _AMT_OFFSET[c] + acc["sigma"][a] * rng.standard_normal()) * cfg.amount_scale
            ch = "atm" if c == _ATM else "pos"
            lat, lon = _place(acc, a, home, rng)
            rows.append(_row(t0 + g, a, amt, c, ch, f"D{a:05d}-0", home, lat, lon, 0, "none", rng))
    return pd.DataFrame(rows, columns=COLUMNS)


def _fraud_transactions(rng: np.random.Generator, cfg: GenConfig, acc) -> pd.DataFrame:
    kinds = rng.choice(["card_testing", "account_takeover", "atm_cashout", "low_and_slow"],
                       size=cfg.fraud_events, p=[0.15, 0.40, 0.10, 0.35])
    attacker_devices = [f"X{i:03d}" for i in range(40)]
    risky = [_M[m] for m in ("electronics", "luxury", "crypto", "gambling", "digital_goods", "travel")]
    rows = []
    for event_no, kind in enumerate(kinds):
        a = int(rng.integers(0, cfg.n_accounts))
        home = int(acc["home"][a])
        typical = float(np.exp(acc["mu"][a]))
        night = rng.random() < 0.6
        hour = int(rng.integers(0, 6)) if night else int(rng.choice(24, p=_HOUR_P))
        t0 = BASE_TS + int(rng.integers(5, cfg.days)) * 86_400 + hour * 3_600 + int(rng.integers(0, 3_600))
        foreign = int(rng.choice([i for i in range(len(_CODES)) if i != home]))
        # Half of the attacks come from a few devices reused across many victims (Zipf-like, a
        # "ring" signal); the other half rotate to a brand-new device, as real attackers do.
        if rng.random() < 0.5:
            att_dev = attacker_devices[min(int(rng.zipf(1.6)) - 1, len(attacker_devices) - 1)]
        else:
            att_dev = f"X{1_000 + event_no:04d}"

        if kind == "card_testing":
            n = int(rng.integers(8, 26))
            gaps = rng.uniform(3, 45, n).cumsum()
            amounts = rng.uniform(0.5, 6.0, n)
            amounts[-2:] = typical * rng.uniform(2, 8, 2)
            cats = rng.choice([_M["digital_goods"], _M["gambling"]], size=n, p=[0.8, 0.2])
            channel, device, ctry = "ecom", att_dev, foreign
        elif kind == "account_takeover":
            n = int(rng.integers(1, 5))
            gaps = rng.uniform(0, 1_500, n).cumsum()
            amounts = typical * rng.uniform(3, 14, n)
            cats = rng.choice(risky, size=n)
            channel, device = "ecom", att_dev
            ctry = foreign if rng.random() < 0.7 else home
        elif kind == "atm_cashout":
            n = int(rng.integers(3, 7))
            gaps = rng.uniform(120, 900, n).cumsum()
            amounts = np.maximum(typical * rng.uniform(3, 8, n), 100.0)
            cats = np.full(n, _ATM)
            channel, device, ctry = "atm", att_dev, foreign
        else:  # low_and_slow: compromised card on the victim's own device, home location
            n = int(rng.integers(1, 4))
            gaps = rng.uniform(600, 7_200, n).cumsum()
            amounts = typical * rng.uniform(1.0, 3.0, n)
            cats = rng.choice(risky[:4], size=n)
            channel, device, ctry = "ecom", f"D{a:05d}-0", home

        for g, amt, c in zip(gaps, amounts, cats):
            lat, lon = _place(acc, a, ctry, rng)
            rows.append(_row(t0 + g, a, amt, int(c), channel, device, ctry, lat, lon, 1, kind, rng))
    return pd.DataFrame(rows, columns=COLUMNS)


def generate_transactions(cfg: GenConfig | None = None) -> pd.DataFrame:
    """Return a time-ordered transaction table (deterministic for a given config)."""
    cfg = cfg or GenConfig()
    rng = np.random.default_rng(cfg.seed)
    acc = _make_accounts(rng, cfg)
    parts = [_legit_transactions(rng, cfg, acc), _legit_bursts(rng, cfg, acc), _fraud_transactions(rng, cfg, acc)]
    df = pd.concat(parts, ignore_index=True)
    df = df.sort_values("timestamp", kind="stable").reset_index(drop=True)
    df.insert(0, "txn_id", "T" + df.index.astype(str).str.zfill(7))
    df["timestamp"] = df["timestamp"].astype("int64")
    df["is_fraud"] = df["is_fraud"].astype("int8")
    return df
