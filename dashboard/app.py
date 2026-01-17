import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]


# -------------------------------------------------
# Helpers
# -------------------------------------------------
def load_eval_results():
    path = ROOT / "runs" / "results" / "eval_results_v1.json"
    if not path.exists():
        st.error("Evaluation results not found. Run evaluation first.")
        st.stop()
    return json.loads(path.read_text())


def load_regression_config():
    path = ROOT / "evals" / "regression_config.json"
    return json.loads(path.read_text())


def flatten_results(results):
    rows = []
    for r in results:
        row = {"sample_id": r["id"]}
        row.update(r["scores"])
        rows.append(row)
    return pd.DataFrame(rows)


# -------------------------------------------------
# Streamlit UI
# -------------------------------------------------
st.set_page_config(page_title="LLM Evaluation Dashboard", layout="wide")

st.title("🧠 LLM Evaluation & Regression Dashboard")
st.caption("Offline, deterministic evaluation of LLM quality")

# ---------------- Load data ----------------
results = load_eval_results()
config = load_regression_config()
df = flatten_results(results)

# ---------------- Aggregate metrics ----------------
st.subheader("📊 Aggregate Metrics")

agg = df.drop(columns=["sample_id"]).mean().round(3)
st.metric("Faithfulness", agg.get("faithfulness", 0.0))
st.metric("Relevance", agg.get("relevance", 0.0))
st.metric("Format Accuracy", agg.get("format_accuracy", 0.0))

# ---------------- Regression Policy ----------------
st.subheader("🚨 Regression Policy")

st.json(config)

# ---------------- Per-sample inspection ----------------
st.subheader("🔍 Per-Sample Scores")

st.dataframe(df, use_container_width=True)

# ---------------- Failures ----------------
st.subheader("❌ Potential Failures")

failures = []

for metric, rules in config.items():
    if "min_score" in rules:
        failures.append(df[df[metric] < rules["min_score"]])

if failures:
    st.warning("Some samples violate regression thresholds")
    st.dataframe(pd.concat(failures).drop_duplicates(), use_container_width=True)
else:
    st.success("No regression violations detected")

# ---------------- Footer ----------------
st.caption(
    "This dashboard is read-only and reflects the latest offline evaluation run."
)
