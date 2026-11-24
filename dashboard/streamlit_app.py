"""Dashboard of the prediction service.

Run with::

    streamlit run dashboard/streamlit_app.py

The address of the API is read from the environment variable ``FRAUD_API_URL``
(default: ``http://127.0.0.1:8000``).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

API_URL = os.environ.get("FRAUD_API_URL", "http://127.0.0.1:8000").rstrip("/")
EXAMPLES_PATH = Path(__file__).resolve().parents[1] / "models" / "fraud_model" / "examples.json"
COMPONENTS = [f"V{i}" for i in range(1, 29)]
FIELDS = ["time_seconds", "amount", *COMPONENTS]
TIMEOUT_SECONDS = 10
CUSTOM = "Custom transaction"


def call_api(method: str, path: str, **kwargs) -> tuple[dict | None, str | None]:
    """Call the API; return the decoded response or an error message."""
    try:
        response = requests.request(method, f"{API_URL}{path}", timeout=TIMEOUT_SECONDS, **kwargs)
    except requests.RequestException as error:
        return None, f"The API at {API_URL} cannot be reached ({type(error).__name__})."
    if response.status_code != 200:
        return None, f"The API answered with status {response.status_code}: {response.text[:300]}"
    return response.json(), None


def load_examples() -> list[dict]:
    if not EXAMPLES_PATH.exists():
        return []
    return json.loads(EXAMPLES_PATH.read_text(encoding="utf-8"))


def apply_example() -> None:
    """Copy the selected example into the input fields."""
    selected = st.session_state["example"]
    for example in load_examples():
        if example["label"] == selected:
            for field in FIELDS:
                st.session_state[field] = float(example["transaction"][field])


st.set_page_config(page_title="Fraud detection", layout="wide")
st.title("Credit card fraud detection")
st.caption(
    "Calibrated probability of fraud, cost-based decision and explanation for one "
    "transaction, as returned by the prediction service."
)

for field in FIELDS:
    st.session_state.setdefault(field, 0.0)

with st.sidebar:
    st.header("Service")
    st.code(API_URL)
    version, error = call_api("GET", "/version")
    if error:
        st.error(error)
    else:
        st.success("Model in service")
        st.write(f"Model version: `{version.get('model_version', 'unknown')}`")
        st.write(f"Trained: `{version.get('trained_at_utc', 'unknown')}`")
        st.write(f"Decision rule: `{version['decision']['rule']}`")
        st.write(f"Alert cost: `{version['decision']['alert_cost']}`")

examples = load_examples()
if examples:
    st.selectbox(
        "Example transaction",
        [CUSTOM, *[example["label"] for example in examples]],
        key="example",
        on_change=apply_example,
        help="Transactions of the development period, stored with the model.",
    )

left, right = st.columns([3, 2])

with left:
    st.subheader("Transaction")
    first, second = st.columns(2)
    first.number_input("Amount", min_value=0.0, key="amount", format="%.2f")
    second.number_input(
        "Time (seconds since the origin of the dataset)",
        min_value=0.0,
        key="time_seconds",
        format="%.0f",
        help="Only the time of day, i.e. this value modulo 24 hours, is used.",
    )
    with st.expander("Principal components V1 to V28", expanded=not examples):
        columns = st.columns(4)
        for index, name in enumerate(COMPONENTS):
            columns[index % 4].number_input(name, key=name, format="%.4f")
    submitted = st.button("Score the transaction", type="primary")

with right:
    st.subheader("Result")
    if submitted:
        payload = {field: float(st.session_state[field]) for field in FIELDS}
        result, error = call_api(
            "POST", "/predict", params={"explain": "true", "top_k": 8}, json=payload
        )
        if error:
            st.error(error)
        else:
            decision = result["decision"]
            if decision["alert"]:
                st.error("Alert: the expected loss exceeds the cost of an alert.")
            else:
                st.success("No alert: the expected loss is below the cost of an alert.")
            a, b, c = st.columns(3)
            a.metric("Fraud probability", f"{result['fraud_probability']:.2%}")
            b.metric("Expected loss", f"{decision['expected_loss']:.2f}")
            c.metric("Alert cost", f"{decision['alert_cost']:.2f}")

            explanation = result["explanation"]
            st.markdown("**Largest contributions to the log-odds of fraud**")
            table = pd.DataFrame(explanation["contributions"])
            st.dataframe(
                table.style.format({"value": "{:.4f}", "contribution": "{:+.3f}"}),
                hide_index=True,
                width="stretch",
            )
            st.caption(
                f"Base value {explanation['base_value']:+.3f}; other features "
                f"{explanation['other_contributions']:+.3f}; total "
                f"{explanation['prediction']:+.3f}. A positive contribution raises the "
                "predicted probability. Contributions describe the model, not the causes "
                "of fraud."
            )
    else:
        st.info("Select an example or enter a transaction, then score it.")
