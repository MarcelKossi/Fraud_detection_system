"""Prediction service.

Run with::

    uvicorn api.main:app --host 0.0.0.0 --port 8000

The model directory is read from the environment variable ``FRAUD_MODEL_DIR``
(default: ``models/fraud_model``) when the application starts.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Annotated

import pandas as pd
from fastapi import APIRouter, FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from api.schemas import Health, Prediction, Transaction
from fraud_detection import __version__
from fraud_detection.serving import AMOUNT_FIELD, RAW_TIME_FIELD, FraudModel, ModelLoadError

DEFAULT_MODEL_DIR = "models/fraud_model"

logger = logging.getLogger("fraud_detection.api")
router = APIRouter()


@asynccontextmanager
async def lifespan(app: FastAPI):
    directory = os.environ.get("FRAUD_MODEL_DIR", DEFAULT_MODEL_DIR)
    try:
        app.state.model = FraudModel.load(directory)
        logger.info("Model loaded from %s", directory)
    except ModelLoadError as error:
        # The service starts without a model and reports it through /health.
        app.state.model = None
        logger.error("No model in service: %s", error)
    yield


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


NO_MODEL = {503: {"description": "No model is loaded."}}


@router.get("/health", response_model=Health, responses={503: {"model": Health}})
def health(request: Request) -> JSONResponse:
    """Readiness of the service: 200 when a model is loaded, 503 otherwise."""
    loaded = request.app.state.model is not None
    report = Health(status="ok" if loaded else "degraded", model_loaded=loaded)
    return JSONResponse(status_code=200 if loaded else 503, content=report.model_dump())


@router.get("/version", responses=NO_MODEL)
def version(request: Request) -> dict:
    """Provenance of the model in service."""
    model = _model(request)
    # The entries set here take precedence over those of the stored metadata.
    return {
        **model.metadata,
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
    }


@router.post(
    "/predict", response_model=Prediction, response_model_exclude_none=True, responses=NO_MODEL
)
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


def create_app() -> FastAPI:
    """Build an application; each call returns an independent instance."""
    application = FastAPI(
        title="Fraud Detection API",
        version=__version__,
        summary="Calibrated fraud probability, cost-based decision and explanation.",
        lifespan=lifespan,
    )
    application.add_exception_handler(RequestValidationError, validation_error)
    application.include_router(router)
    return application


app = create_app()
