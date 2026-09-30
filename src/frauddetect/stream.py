"""Stream simulator: a producer publishes transactions to a bounded queue, a consumer scores them.

The queue stands in for a Kafka topic (bounded => back-pressure, ordered => per-key ordering
like a partition keyed by account_id). The consumer is exactly the code a Kafka consumer loop
would call, so swapping in ``confluent-kafka`` only changes ``producer`` and the ``get`` call.
"""
from __future__ import annotations

import queue
import threading
import time

import numpy as np

from .features import TXN_FIELDS
from .scoring import Scorer, _percentiles


def replay(scorer: Scorer, df, max_rows: int | None = None, queue_size: int = 1_000, keep_alerts: int = 10) -> dict:
    cols = ["txn_id", *TXN_FIELDS, "is_fraud", "fraud_type"]
    rows = df[cols].to_dict("records")
    if max_rows:
        rows = rows[:max_rows]

    q: queue.Queue = queue.Queue(maxsize=queue_size)

    def producer() -> None:
        for r in rows:
            q.put(r)
        q.put(None)

    threading.Thread(target=producer, daemon=True).start()

    kept: dict[str, int] = {}   # at most 2 example alerts per fraud type
    lat, tp, fp, fn, decisions, samples = [], 0, 0, 0, {"approve": 0, "review": 0, "decline": 0}, []
    start = time.perf_counter()
    n = 0
    while (r := q.get()) is not None:
        res = scorer.score({k: r[k] for k in ["txn_id", *TXN_FIELDS]})
        n += 1
        lat.append(res["latency_ms"])
        decisions[res["decision"]] += 1
        flagged, fraud = res["decision"] != "approve", r["is_fraud"] == 1
        tp += flagged and fraud
        fp += flagged and not fraud
        fn += (not flagged) and fraud
        if flagged and fraud and res["reasons"] and len(samples) < keep_alerts and kept.get(r["fraud_type"], 0) < 2:
            kept[r["fraud_type"]] = kept.get(r["fraud_type"], 0) + 1
            samples.append({"txn_id": r["txn_id"], "amount": r["amount"], "true_type": r["fraud_type"], **{
                k: res[k] for k in ("risk_score", "decision", "reasons")}})
    wall = time.perf_counter() - start
    return {
        "transactions": n, "wall_seconds": round(wall, 2), "throughput_tps": round(n / wall, 1),
        "score_latency_ms": _percentiles(lat), "decisions": decisions,
        "alert_precision": round(tp / max(tp + fp, 1), 4), "fraud_recall": round(tp / max(tp + fn, 1), 4),
        "sample_alerts": samples,
    }
