# app/streamlit_app.py
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src.config import THRESH_APPROVE, THRESH_REVIEW
from src.models.scorecard import prob_to_score, decision_from_prob

st.set_page_config(page_title="Underwriting Decisioning Demo", layout="wide")
st.title("Underwriting Decisioning System — Demo UI")
st.caption("Scores applicants with P(creditworthy) → scorecard score → decision. Includes calibration + artifacts.")

MODEL_DIR = Path("models")

@st.cache_resource
def load_artifacts():
    if not MODEL_DIR.exists():
        return None, None
    calibrator_path = MODEL_DIR / "calibrator.pkl"
    meta_path = MODEL_DIR / "metadata.json"
    if not calibrator_path.exists() or not meta_path.exists():
        return None, None
    calibrated = joblib.load(calibrator_path)
    meta = json.loads(meta_path.read_text())
    return calibrated, meta

calibrated, meta = load_artifacts()

with st.sidebar:
    st.header("Controls")
    approve = st.slider("Approve threshold", 0.50, 0.95, float(THRESH_APPROVE), 0.01)
    review = st.slider("Review threshold", 0.10, 0.90, float(THRESH_REVIEW), 0.01)
    st.markdown("---")
    st.write("Artifacts status:")
    st.write("✅ Loaded" if calibrated else "❌ Train first")

st.markdown("### 1) Upload CSV (or use sample)")
uploaded = st.file_uploader("Upload a CSV with feature columns", type=["csv"])

if uploaded is None:
    sample_path = Path("data/sample/credit_sample.csv")
    if sample_path.exists():
        df = pd.read_csv(sample_path)
        st.info("Using sample dataset: data/sample/credit_sample.csv")
    else:
        st.warning("No sample found. Generate it with: python -m src.data.make_dataset")
        st.stop()
else:
    df = pd.read_csv(uploaded)

st.write("Preview:", df.head())

if not calibrated:
    st.warning("Model artifacts not found. Train the model first:\n\n`python -m src.models.train --data data/sample/credit_sample.csv`")
    st.stop()

features = meta["features"]
missing = [c for c in features if c not in df.columns]
if missing:
    st.error(f"Missing required feature columns: {missing}")
    st.stop()

st.markdown("### 2) Score applicants")
row_idx = st.number_input("Choose a row index to score", min_value=0, max_value=len(df)-1, value=0, step=1)

if st.button("Score selected row"):
    x = df.loc[[row_idx], features]
    p_good = float(calibrated.predict_proba(x)[:, 1][0])
    score = int(prob_to_score([p_good])[0])
    decision = decision_from_prob(p_good, approve, review)

    c1, c2, c3 = st.columns(3)
    c1.metric("P(creditworthy)", f"{p_good:.3f}")
    c2.metric("Scorecard score", score)
    c3.metric("Decision", decision)

    st.markdown("### 3) (Optional) Export scored CSV")
    p_all = calibrated.predict_proba(df[features])[:, 1]
    out = df.copy()
    out["p_creditworthy"] = p_all
    out["credit_score"] = prob_to_score(p_all)
    out["decision"] = [decision_from_prob(p, approve, review) for p in p_all]

    st.download_button(
        "Download scored.csv",
        data=out.to_csv(index=False).encode("utf-8"),
        file_name="scored.csv",
        mime="text/csv",
    )
