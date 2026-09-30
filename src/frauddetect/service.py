"""HTTP scoring API (Starlette + pydantic).

Run:  uvicorn frauddetect.service:create_app --factory --port 8000
Env:  FRAUD_MODEL_PATH (default models/fraud_model.joblib)

Starlette is the ASGI toolkit FastAPI is built on; using it directly keeps the
dependency footprint small. Porting the two handlers to FastAPI is mechanical.
"""
from __future__ import annotations

import json
import os

from pydantic import BaseModel, Field, ValidationError, field_validator
from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from .config import CHANNELS, DEFAULT_MODEL_PATH, MCC
from .model import load_artifact
from .scoring import Scorer


class Transaction(BaseModel):
    txn_id: str = Field(min_length=1, max_length=64)
    timestamp: float = Field(gt=0, description="Unix seconds (UTC)")
    account_id: str = Field(min_length=1, max_length=64)
    amount: float = Field(gt=0, lt=10_000_000)
    merchant_id: str = Field(min_length=1, max_length=64)
    mcc: str
    channel: str
    device_id: str = Field(min_length=1, max_length=64)
    country: str = Field(min_length=2, max_length=3)
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)

    @field_validator("mcc")
    @classmethod
    def _mcc(cls, v: str) -> str:
        if v not in MCC:
            raise ValueError(f"mcc must be one of {MCC}")
        return v

    @field_validator("channel")
    @classmethod
    def _channel(cls, v: str) -> str:
        if v not in CHANNELS:
            raise ValueError(f"channel must be one of {CHANNELS}")
        return v


def create_app(scorer: Scorer | None = None) -> Starlette:
    if scorer is None:
        scorer = Scorer(load_artifact(os.environ.get("FRAUD_MODEL_PATH", DEFAULT_MODEL_PATH)))

    async def score(request: Request) -> JSONResponse:
        try:
            payload = await request.json()
        except ValueError:
            return JSONResponse({"error": "invalid_json"}, status_code=400)
        try:
            txn = Transaction.model_validate(payload)
        except ValidationError as exc:
            return JSONResponse({"error": "validation_error", "details": json.loads(exc.json(include_url=False))},
                                status_code=422)
        # Scoring is CPU-bound; keep it off the event loop.
        return JSONResponse(await run_in_threadpool(scorer.score, txn.model_dump()))

    async def health(request: Request) -> JSONResponse:
        return JSONResponse({"status": "ok", "model_version": scorer.version})

    async def metrics(request: Request) -> JSONResponse:
        return JSONResponse(scorer.metrics())

    app = Starlette(routes=[
        Route("/v1/score", score, methods=["POST"]),
        Route("/health", health, methods=["GET"]),
        Route("/metrics", metrics, methods=["GET"]),
    ])
    app.state.scorer = scorer
    return app
