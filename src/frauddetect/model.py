"""Training, imbalance handling, baselines and evaluation.

Key ideas (all discussed in the README):

* **Time-based split** (never random): train on the past, validate and test on later windows.
* **Label delay**: the train window ends ``LABEL_DELAY_DAYS`` before validation starts.
* **Class imbalance**: compare no weighting vs sqrt vs full balanced weights, pick on validation PR-AUC.
* **Operating points chosen by false-positive rate**, not by 0.5: the business decides how
  many good customers it will bother, and the threshold follows.
* **Honest baselines**: hand-written rules and logistic regression, on the same split.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer

from . import drift
from .config import BLOCK_FPR, BURN_IN_DAYS, LABEL_DELAY_DAYS, MCC, REVIEW_FPR, SEED
from .explain import Explainer, reference_values
from .features import CATEGORICAL_FEATURES, FEATURES, compute_features

_RISKY_MCC = [MCC.index(m) for m in ("electronics", "luxury", "crypto", "gambling", "digital_goods")]


@dataclass
class Splits:
    train: np.ndarray
    valid: np.ndarray
    test: np.ndarray


def time_splits(ts: pd.Series, train_frac: float = 0.70, valid_frac: float = 0.15,
                label_delay_days: float = LABEL_DELAY_DAYS, burn_in_days: float = BURN_IN_DAYS) -> Splits:
    t_train = ts.quantile(train_frac)
    t_valid = ts.quantile(train_frac + valid_frac)
    return Splits(
        train=((ts <= t_train - label_delay_days * 86_400) & (ts >= ts.min() + burn_in_days * 86_400)).to_numpy(),
        valid=((ts > t_train) & (ts <= t_valid)).to_numpy(),
        test=(ts > t_valid).to_numpy(),
    )


def sample_weights(y: np.ndarray, mode: str) -> np.ndarray:
    ratio = (y == 0).sum() / max((y == 1).sum(), 1)
    pos_w = {"none": 1.0, "sqrt": float(np.sqrt(ratio)), "balanced": float(ratio)}[mode]
    return np.where(y == 1, pos_w, 1.0)


def fit_hgb(X: np.ndarray, y: np.ndarray, weight_mode: str = "sqrt", seed: int = SEED) -> HistGradientBoostingClassifier:
    cat_idx = [FEATURES.index(c) for c in CATEGORICAL_FEATURES]
    model = HistGradientBoostingClassifier(
        learning_rate=0.08, max_iter=300, max_leaf_nodes=31, min_samples_leaf=30,
        l2_regularization=1.0, categorical_features=cat_idx,
        early_stopping=True, validation_fraction=0.12, n_iter_no_change=25, random_state=seed,
    )
    return model.fit(X, y, sample_weight=sample_weights(y, weight_mode))


def rule_score(F: pd.DataFrame) -> np.ndarray:
    """What a hand-written rules engine typically looks like (the thing ML has to beat)."""
    s = np.zeros(len(F))
    s += 1.0 * (F["amount_z"].to_numpy() > 3)
    s += 1.0 * (F["new_device"].to_numpy() == 1)
    s += 1.0 * (F["device_n_accounts"].to_numpy() >= 2)
    s += 1.5 * (F["cnt_10m"].to_numpy() >= 5)
    s += 2.0 * (np.nan_to_num(F["speed_kmh_last"].to_numpy()) > 900)
    s += 1.0 * (F["new_country"].to_numpy() == 1)
    s += 0.5 * (F["is_night"].to_numpy() == 1)
    s += 0.5 * np.isin(F["mcc_code"].to_numpy(), _RISKY_MCC)
    return s + 1e-3 * np.tanh(F["amount_z"].to_numpy())      # tiny tie-break


def fit_logreg(F: pd.DataFrame, y: np.ndarray, seed: int = SEED) -> Pipeline:
    num = [f for f in FEATURES if f not in CATEGORICAL_FEATURES]
    pre = ColumnTransformer([
        ("num", Pipeline([
            ("imp", SimpleImputer(strategy="median", add_indicator=True)),
            ("q", QuantileTransformer(n_quantiles=200, output_distribution="normal", random_state=seed)),
        ]), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])
    clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000)
    return Pipeline([("pre", pre), ("clf", clf)]).fit(F, y)


# ---------------------------------------------------------------- metrics
def recall_at_fpr(y: np.ndarray, s: np.ndarray, fpr_target: float) -> float:
    fpr, tpr, _ = roc_curve(y, s)
    return float(tpr[fpr <= fpr_target].max())


def threshold_for_fpr(y: np.ndarray, s: np.ndarray, fpr: float) -> float:
    """Score threshold that flags roughly ``fpr`` of the legitimate transactions."""
    return float(np.quantile(s[y == 0], 1.0 - fpr))


def operating_point(y: np.ndarray, s: np.ndarray, amount: np.ndarray, thr: float) -> dict:
    pred = s >= thr
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    tn = int((~pred & (y == 0)).sum())
    fraud_amt = amount[y == 1].sum()
    return {
        "threshold": round(thr, 6), "alerts": int(pred.sum()), "tp": tp, "fp": fp, "fn": fn,
        "precision": round(tp / max(tp + fp, 1), 4), "recall": round(tp / max(tp + fn, 1), 4),
        "fpr": round(fp / max(fp + tn, 1), 5),
        "amount_recall": round(float(amount[pred & (y == 1)].sum() / max(fraud_amt, 1e-9)), 4),
    }


def score_metrics(y: np.ndarray, s: np.ndarray) -> dict:
    return {
        "pr_auc": round(float(average_precision_score(y, s)), 4),
        "roc_auc": round(float(roc_auc_score(y, s)), 4),
        "recall@0.1%fpr": round(recall_at_fpr(y, s, 0.001), 4),
        "recall@0.5%fpr": round(recall_at_fpr(y, s, 0.005), 4),
        "recall@1%fpr": round(recall_at_fpr(y, s, 0.01), 4),
    }


def pr_curve_points(y: np.ndarray, s: np.ndarray, n: int = 250) -> dict:
    """Down-sampled precision/recall curve (small enough to store as JSON for plotting)."""
    p, r, _ = precision_recall_curve(y, s)
    idx = np.unique(np.linspace(0, len(p) - 1, n).astype(int))
    return {"precision": p[idx].round(4).tolist(), "recall": r[idx].round(4).tolist()}


def recall_by_typology(fraud_type: np.ndarray, y: np.ndarray, s: np.ndarray, thr: float) -> dict:
    out = {}
    for t in sorted(set(fraud_type[y == 1])):
        m = (fraud_type == t) & (y == 1)
        out[str(t)] = {"n": int(m.sum()), "recall": round(float((s[m] >= thr).mean()), 4)}
    return out


# ---------------------------------------------------------------- pipeline
def train(df: pd.DataFrame, seed: int = SEED, verbose: bool = True) -> tuple[dict, dict]:
    """Full offline pipeline. Returns (artifact for serving, report with all metrics)."""
    log = print if verbose else (lambda *a, **k: None)
    t0 = time.time()
    F, _ = compute_features(df)
    X = F.to_numpy(dtype=float)
    y = df["is_fraud"].to_numpy()
    amt = df["amount"].to_numpy()
    ftype = df["fraud_type"].to_numpy()
    sp = time_splits(df["timestamp"])
    log(f"features: {X.shape} in {time.time() - t0:.1f}s | train={sp.train.sum():,} valid={sp.valid.sum():,} test={sp.test.sum():,}")

    # 1) imbalance strategy chosen on the validation window
    weight_search, best = {}, None
    for mode in ("none", "sqrt", "balanced"):
        m = fit_hgb(X[sp.train], y[sp.train], mode, seed)
        ap = float(average_precision_score(y[sp.valid], m.predict_proba(X[sp.valid])[:, 1]))
        weight_search[mode] = {"valid_pr_auc": round(ap, 4), "trees": int(m.n_iter_)}
        log(f"  weight={mode:<8} valid PR-AUC={ap:.4f} trees={m.n_iter_}")
        if best is None or ap > best[0]:
            best = (ap, mode, m)
    _, best_mode, model = best

    # 2) operating points from validation scores
    s_valid = model.predict_proba(X[sp.valid])[:, 1]
    thr_review = threshold_for_fpr(y[sp.valid], s_valid, REVIEW_FPR)
    thr_block = threshold_for_fpr(y[sp.valid], s_valid, BLOCK_FPR)

    # 3) held-out test comparison against baselines
    s_test = model.predict_proba(X[sp.test])[:, 1]
    lr = fit_logreg(F[sp.train], y[sp.train], seed)
    s_lr = lr.predict_proba(F[sp.test])[:, 1]
    s_rule = rule_score(F[sp.test])
    yt, at = y[sp.test], amt[sp.test]
    comparison = {
        "rules": score_metrics(yt, s_rule),
        "logistic_regression": score_metrics(yt, s_lr),
        "gradient_boosting": score_metrics(yt, s_test),
    }
    for name, s in (("rules", s_rule), ("logistic_regression", s_lr)):
        thr = threshold_for_fpr(y[sp.valid], s_rule_or_lr(name, F, lr, sp), REVIEW_FPR)
        comparison[name]["at_review_threshold"] = operating_point(yt, s, at, thr)

    # 4) explanations sanity + global group importance on test frauds
    ref_vals = reference_values(X[sp.train])
    explainer = Explainer(model, ref_vals)
    fraud_idx = np.where(sp.test)[0][yt == 1]
    contrib = explainer.contributions(X[fraud_idx])
    group_importance = {g: round(float(np.clip(contrib[:, j], 0, None).mean()), 3) for j, g in enumerate(explainer.groups)}

    report = {
        "dataset": {"rows": int(len(df)), "fraud_rate": round(float(y.mean()), 5),
                    "train_rows": int(sp.train.sum()), "valid_rows": int(sp.valid.sum()), "test_rows": int(sp.test.sum()),
                    "test_fraud_rows": int(yt.sum())},
        "imbalance_strategy": {"chosen": best_mode, "search": weight_search},
        "comparison_test": comparison,
        "operating_points_test": {
            f"review (validation FPR<={REVIEW_FPR:.2%})": operating_point(yt, s_test, at, thr_review),
            f"block (validation FPR<={BLOCK_FPR:.3%})": operating_point(yt, s_test, at, thr_block),
        },
        "recall_by_typology_at_review": recall_by_typology(ftype[sp.test], yt, s_test, thr_review),
        "group_importance_mean_logodds_on_test_fraud": dict(sorted(group_importance.items(), key=lambda kv: -kv[1])),
        "seconds": round(time.time() - t0, 1),
        "pr_curves": {"rules": pr_curve_points(yt, s_rule), "logistic_regression": pr_curve_points(yt, s_lr),
                      "gradient_boosting": pr_curve_points(yt, s_test)},
    }

    score_ref = drift.build_reference({**{f: X[sp.train][:, i] for i, f in enumerate(FEATURES)},
                                       "score": model.predict_proba(X[sp.train])[:, 1]})
    artifact = {
        "model": model, "features": FEATURES,
        "thresholds": {"review": thr_review, "block": thr_block},
        "reference_values": ref_vals, "drift_reference": score_ref,
        "meta": {"trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "seed": seed,
                 "weight_mode": best_mode, "train_rows": int(sp.train.sum()),
                 "model_version": f"hgb-{best_mode}-{int(sp.train.sum())}-s{seed}"},
    }
    if verbose:
        for k, v in report["operating_points_test"].items():
            log(f"  {k}: {v}")
    return artifact, report


def s_rule_or_lr(name, F, lr, sp):
    """Validation-window scores for a baseline (used to set its threshold on validation, not test)."""
    return rule_score(F[sp.valid]) if name == "rules" else lr.predict_proba(F[sp.valid])[:, 1]


def save_artifact(artifact: dict, path: str) -> None:
    joblib.dump(artifact, path, compress=3)


def load_artifact(path: str) -> dict:
    """Only load artifacts you trained yourself: joblib/pickle can execute code on load."""
    return joblib.load(path)
