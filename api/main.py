"""Prediction service.

Run with::

    uvicorn api.main:app --host 0.0.0.0 --port 8000

The model directory is read from the environment variable ``FRAUD_MODEL_DIR``
(default: ``models/fraud_model``).
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Annotated

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from api.schemas import Health, Prediction, Transaction
from fraud_detection import __version__
from fraud_detection.serving import AMOUNT_FIELD, RAW_TIME_FIELD, FraudModel

DEFAULT_MODEL_DIR = "models/fraud_model"

logger = logging.getLogger("fraud_detection.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    directory = os.environ.get("FRAUD_MODEL_DIR", DEFAULT_MODEL_DIR)
    try:
        app.state.model = FraudModel.load(directory)
        logger.info("Model loaded from %s", directory)
    except (OSError, ValueError, KeyError) as error:
        # The service starts without a model and reports it through /health.
        app.state.model = None
        logger.error("No model could be loaded from %s: %s", directory, error)
    yield


app = FastAPI(
    title="Fraud Detection API",
    version=__version__,
    summary="Calibrated fraud probability, cost-based decision and explanation.",
    lifespan=lifespan,
)


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, error: RequestValidationError) -> JSONResponse:
    """Report invalid requests without echoing the submitted values.

    The default handler returns the offending input, which cannot be encoded as JSON
    when it is a non-finite number.
    """
    detail = [
        {"loc": list(item["loc"]), "msg": item["msg"], "type": item["type"]}
        for item in error.errors()
    ]
    return JSONResponse(status_code=422, content={"detail": detail})


def _model(request: Request) -> FraudModel:
    model = request.app.state.model
    if model is None:
        raise HTTPException(status_code=503, detail="No model is loaded.")
    return model


@app.get("/health", response_model=Health)
def health(request: Request) -> Health:
    """Liveness of the service and availability of the model."""
    loaded = request.app.state.model is not None
    return Health(status="ok" if loaded else "degraded", model_loaded=loaded)


@app.get("/version")
def version(request: Request) -> dict:
    """Provenance of the model in service."""
    model = _model(request)
    return {
        "api_version": __version__,
        "decision": {"rule": "expected_cost", "alert_cost": model.alert_cost},
        "calibration": {
            "method": "sigmoid",
            "slope": model.calibration_slope,
            "intercept": model.calibration_intercept,
        },
        "input_fields": [
            "time_seconds",
            "amount",
            *[f for f in model.input_fields if f not in (RAW_TIME_FIELD, AMOUNT_FIELD)],
        ],
        **model.metadata,
    }


@app.post("/predict", response_model=Prediction, response_model_exclude_none=True)
def predict(
    transaction: Transaction,
    request: Request,
    explain: Annotated[
        bool, Query(description="Include the contribution of each feature.")
    ] = False,
    top_k: Annotated[
        int,
        Query(ge=1, description="Number of contributions reported; capped at the feature count."),
    ] = 5,
) -> Prediction:
    """Score one transaction.

    Returns the calibrated probability of fraud and the decision of the expected-cost
    rule: an alert is raised when the probability multiplied by the amount exceeds the
    handling cost of an alert.
    """
    model = _model(request)
    fields = transaction.model_dump()
    fields[RAW_TIME_FIELD] = fields.pop("time_seconds")
    fields[AMOUNT_FIELD] = fields.pop("amount")
    frame = pd.DataFrame([fields])

    probability = float(model.predict_proba(frame)[0])
    decision = model.decide(probability, fields[AMOUNT_FIELD])
    return Prediction(
        fraud_probability=probability,
        decision={
            "rule": "expected_cost",
            "alert_cost": model.alert_cost,
            "expected_loss": float(decision["expected_loss"]),
            "alert": bool(decision["alert"]),
        },
        model_version=str(model.metadata.get("model_version", "unknown")),
        explanation=(
            model.explain(frame, top_k=min(top_k, len(model.feature_names))) if explain else None
        ),
    )
