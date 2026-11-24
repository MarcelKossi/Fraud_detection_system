"""Request and response models of the prediction service."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, create_model

N_COMPONENTS = 28

_component_fields = {
    f"V{i}": (
        float,
        Field(..., allow_inf_nan=False, description=f"Principal component V{i} of the dataset."),
    )
    for i in range(1, N_COMPONENTS + 1)
}

Transaction = create_model(
    "Transaction",
    __config__=ConfigDict(extra="forbid"),
    __doc__=(
        "One card transaction, described by the fields of the Credit Card Fraud Detection dataset."
    ),
    time_seconds=(
        float,
        Field(
            ...,
            ge=0,
            allow_inf_nan=False,
            description=(
                "Seconds elapsed since the reference origin of the dataset (its `Time` "
                "column). Only its value modulo 24 hours is used."
            ),
        ),
    ),
    amount=(
        float,
        Field(..., ge=0, allow_inf_nan=False, description="Transaction amount."),
    ),
    **_component_fields,
)


class Decision(BaseModel):
    rule: str = Field(description="Decision rule applied.")
    alert_cost: float = Field(description="Handling cost of an alert assumed by the rule.")
    expected_loss: float = Field(description="Fraud probability multiplied by the amount.")
    alert: bool = Field(description="True when the expected loss exceeds the alert cost.")


class Contribution(BaseModel):
    feature: str
    value: float
    contribution: float = Field(description="Contribution to the log-odds of fraud.")


class Explanation(BaseModel):
    scale: str
    base_value: float = Field(description="Log-odds predicted without feature information.")
    prediction: float = Field(description="Predicted log-odds; sum of all the terms.")
    contributions: list[Contribution]
    other_contributions: float = Field(
        description="Sum of the contributions of the features not listed."
    )


class Prediction(BaseModel):
    fraud_probability: float = Field(description="Calibrated probability of fraud.")
    decision: Decision
    model_version: str
    explanation: Explanation | None = None


class Health(BaseModel):
    status: str
    model_loaded: bool
