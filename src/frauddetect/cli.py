"""Command line entry point:  python -m frauddetect <command>"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

from .config import DEFAULT_DATA_PATH, DEFAULT_MODEL_PATH, SEED


def _save_json(obj, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, default=float))


def _load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def cmd_generate(a) -> None:
    from .data import GenConfig, generate_transactions

    df = generate_transactions(GenConfig(n_legit=a.n_legit, n_accounts=a.n_accounts, seed=a.seed))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"wrote {len(df):,} transactions ({df.is_fraud.mean():.2%} fraud) -> {a.out}")
    print(df.fraud_type.value_counts().to_string())


def cmd_train(a) -> None:
    from .model import save_artifact, train

    df = _load_data(a.data)
    artifact, report = train(df, seed=a.seed)
    _save_json(report.pop("pr_curves"), "results/pr_curves.json")
    Path(a.model).parent.mkdir(parents=True, exist_ok=True)
    save_artifact(artifact, a.model)
    _save_json(report, a.report)
    print(f"model -> {a.model}\nreport -> {a.report}")


def _test_window(df: pd.DataFrame) -> pd.DataFrame:
    from .model import time_splits

    return df[time_splits(df["timestamp"]).test]


def _warm_scorer(model_path: str, df: pd.DataFrame):
    from .model import load_artifact, time_splits
    from .scoring import Scorer

    scorer = Scorer(load_artifact(model_path))
    sp = time_splits(df["timestamp"])
    scorer.warm_up(df[~sp.test])          # everything before the test window builds state
    return scorer, df[sp.test]


def cmd_stream(a) -> None:
    from .stream import replay

    scorer, test = _warm_scorer(a.model, _load_data(a.data))
    stats = replay(scorer, test, max_rows=a.max_rows)
    _save_json(stats, a.out)
    print(json.dumps({k: v for k, v in stats.items() if k != "sample_alerts"}, indent=2))
    print(f"{len(stats['sample_alerts'])} sample alerts saved to {a.out}")


def cmd_benchmark(a) -> None:
    from .benchmark import bench_http, bench_inprocess

    df = _load_data(a.data)
    scorer, test = _warm_scorer(a.model, df)
    out = {"in_process": bench_inprocess(scorer, test, a.n)}
    print("in-process:", json.dumps(out["in_process"]))
    if not a.no_http:
        out["http_loopback"] = bench_http(a.model, test, min(a.n, 2_000))
        print("http:", json.dumps(out["http_loopback"]))
    _save_json(out, a.out)


def cmd_drift(a) -> None:
    from .data import GenConfig, generate_transactions
    from .drift import drift_report
    from .features import compute_features
    from .model import load_artifact

    ref = load_artifact(a.model)["drift_reference"]
    df = _load_data(a.data)
    base_test = _test_window(df)
    F_base, _ = compute_features(df)
    shifted = generate_transactions(GenConfig(amount_scale=1.6, ecom_shift=0.25))
    F_shift, _ = compute_features(shifted)
    idx_b, idx_s = base_test.index, _test_window(shifted).index
    res = {
        "same_distribution": drift_report(ref, {c: F_base.loc[idx_b, c].to_numpy() for c in F_base}),
        "shifted_population": drift_report(ref, {c: F_shift.loc[idx_s, c].to_numpy() for c in F_shift}),
    }
    _save_json(res, a.out)
    for name, r in res.items():
        top = list(r["psi"].items())[:4]
        print(f"{name}: status={r['status']} max_psi={r['max_psi']} top={top}")


def cmd_plots(a) -> None:
    from .plots import make_plots

    for path in make_plots(a.results, a.out):
        print("wrote", path)


def cmd_serve(a) -> None:
    import uvicorn

    os.environ["FRAUD_MODEL_PATH"] = a.model
    uvicorn.run("frauddetect.service:create_app", factory=True, host=a.host, port=a.port)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="frauddetect", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="create the synthetic dataset")
    g.add_argument("--n-legit", type=int, default=200_000)
    g.add_argument("--n-accounts", type=int, default=5_000)
    g.add_argument("--seed", type=int, default=SEED)
    g.add_argument("--out", default=DEFAULT_DATA_PATH)
    g.set_defaults(fn=cmd_generate)

    t = sub.add_parser("train", help="build features, train, evaluate, save artifact")
    t.add_argument("--data", default=DEFAULT_DATA_PATH)
    t.add_argument("--model", default=DEFAULT_MODEL_PATH)
    t.add_argument("--report", default="results/train_report.json")
    t.add_argument("--seed", type=int, default=SEED)
    t.set_defaults(fn=cmd_train)

    s = sub.add_parser("stream", help="replay the test window through the scorer")
    s.add_argument("--data", default=DEFAULT_DATA_PATH)
    s.add_argument("--model", default=DEFAULT_MODEL_PATH)
    s.add_argument("--max-rows", type=int, default=None)
    s.add_argument("--out", default="results/stream_report.json")
    s.set_defaults(fn=cmd_stream)

    b = sub.add_parser("benchmark", help="latency benchmark (in-process + HTTP)")
    b.add_argument("--data", default=DEFAULT_DATA_PATH)
    b.add_argument("--model", default=DEFAULT_MODEL_PATH)
    b.add_argument("--n", type=int, default=5_000)
    b.add_argument("--no-http", action="store_true")
    b.add_argument("--out", default="results/benchmark.json")
    b.set_defaults(fn=cmd_benchmark)

    d = sub.add_parser("drift-demo", help="show PSI on a same-distribution vs shifted population")
    d.add_argument("--data", default=DEFAULT_DATA_PATH)
    d.add_argument("--model", default=DEFAULT_MODEL_PATH)
    d.add_argument("--out", default="results/drift_demo.json")
    d.set_defaults(fn=cmd_drift)

    pl = sub.add_parser("plots", help="render README charts from results/*.json")
    pl.add_argument("--results", default="results")
    pl.add_argument("--out", default="docs")
    pl.set_defaults(fn=cmd_plots)

    v = sub.add_parser("serve", help="run the HTTP API")
    v.add_argument("--model", default=DEFAULT_MODEL_PATH)
    v.add_argument("--host", default="127.0.0.1")
    v.add_argument("--port", type=int, default=8000)
    v.set_defaults(fn=cmd_serve)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
