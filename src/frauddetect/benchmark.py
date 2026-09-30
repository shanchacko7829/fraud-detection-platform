"""Latency benchmarks: in-process scoring path and a real HTTP round trip."""
from __future__ import annotations

import os
import subprocess
import sys
import time

import numpy as np

from .features import TXN_FIELDS
from .scoring import Scorer, _percentiles


def bench_inprocess(scorer: Scorer, df, n: int = 5_000) -> dict:
    """Wall-clock time of ``Scorer.score`` per transaction (features + model [+ explanation])."""
    rows = df[["txn_id", *TXN_FIELDS]].head(n).to_dict("records")
    lat = []
    t0 = time.perf_counter()
    for r in rows:
        t = time.perf_counter()
        scorer.score(r)
        lat.append((time.perf_counter() - t) * 1000.0)
    wall = time.perf_counter() - t0
    return {"n": len(rows), "throughput_tps": round(len(rows) / wall, 1), "latency_ms": _percentiles(lat)}


def bench_http(model_path: str, df, n: int = 2_000, port: int = 8099) -> dict:
    """Start uvicorn as a subprocess and measure client-side round-trip latency over loopback HTTP."""
    import requests

    env = {**os.environ, "FRAUD_MODEL_PATH": model_path,
           "PYTHONPATH": os.pathsep.join(filter(None, ["src", os.environ.get("PYTHONPATH", "")]))}
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "frauddetect.service:create_app", "--factory",
         "--port", str(port), "--log-level", "warning"], env=env)
    base = f"http://127.0.0.1:{port}"
    try:
        for _ in range(120):
            try:
                if requests.get(f"{base}/health", timeout=1).status_code == 200:
                    break
            except requests.RequestException:
                time.sleep(0.5)
        else:
            raise RuntimeError("service did not start")
        rows = df[["txn_id", *TXN_FIELDS]].head(n).to_dict("records")
        sess = requests.Session()
        lat = []
        t0 = time.perf_counter()
        for r in rows:
            t = time.perf_counter()
            resp = sess.post(f"{base}/v1/score", json=r, timeout=5)
            resp.raise_for_status()
            lat.append((time.perf_counter() - t) * 1000.0)
        wall = time.perf_counter() - t0
        return {"n": len(rows), "throughput_rps": round(len(rows) / wall, 1), "latency_ms": _percentiles(lat)}
    finally:
        proc.terminate()
        proc.wait(timeout=10)
