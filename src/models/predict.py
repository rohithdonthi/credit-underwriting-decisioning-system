# src/models/predict.py
import argparse
from pathlib import Path

import joblib
import pandas as pd

from src.config import THRESH_APPROVE, THRESH_REVIEW
from src.models.scorecard import prob_to_score, decision_from_prob

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/sample/credit_sample.csv")
    ap.add_argument("--model_dir", type=str, default="models")
    ap.add_argument("--out", type=str, default="outputs/scored.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.data)

    calibrated = joblib.load(Path(args.model_dir) / "calibrator.pkl")
    metadata = pd.read_json(Path(args.model_dir) / "metadata.json")
    # metadata is dict-like but pd.read_json may parse oddly; safer:
    import json
    meta = json.loads(Path(args.model_dir, "metadata.json").read_text())
    feats = meta["features"]

    X = df[feats]
    p_good = calibrated.predict_proba(X)[:, 1]

    df_out = df.copy()
    df_out["p_creditworthy"] = p_good
    df_out["credit_score"] = prob_to_score(p_good)
    df_out["decision"] = [decision_from_prob(p, THRESH_APPROVE, THRESH_REVIEW) for p in p_good]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out, index=False)
    print(f"Saved scored file: {out}")

if __name__ == "__main__":
    main()
