# src/models/train.py
import argparse, json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

from src.config import TIME_COL, TARGET_COL, RANDOM_SEED
from src.models.evaluate import compute_metrics

NUM_COLS = ["income","utilization","dti","delinq_12m","inquiries_6m","tradelines","thin_file","high_util"]

def time_split(df: pd.DataFrame):
    # sort by time
    d = df.copy()
    d[TIME_COL] = d[TIME_COL].astype(str)
    months = sorted(d[TIME_COL].unique())

    # 70/15/15 by months
    n = len(months)
    train_end = int(0.70 * n)
    calib_end = int(0.85 * n)

    train_months = set(months[:train_end])
    calib_months = set(months[train_end:calib_end])
    test_months  = set(months[calib_end:])

    train = d[d[TIME_COL].isin(train_months)]
    calib = d[d[TIME_COL].isin(calib_months)]
    test  = d[d[TIME_COL].isin(test_months)]
    return train, calib, test, {"train_months": sorted(train_months), "calib_months": sorted(calib_months), "test_months": sorted(test_months)}

def build_base_model():
    pre = ColumnTransformer(
        transformers=[("num", StandardScaler(), NUM_COLS)],
        remainder="drop"
    )
    clf = LogisticRegression(max_iter=2000, random_state=RANDOM_SEED)
    return Pipeline([("pre", pre), ("clf", clf)])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/sample/credit_sample.csv")
    ap.add_argument("--model_dir", type=str, default="models")
    ap.add_argument("--report_dir", type=str, default="reports")
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    train_df, calib_df, test_df, split_info = time_split(df)

    X_train, y_train = train_df[NUM_COLS], train_df[TARGET_COL].astype(int)
    X_calib, y_calib = calib_df[NUM_COLS], calib_df[TARGET_COL].astype(int)
    X_test,  y_test  = test_df[NUM_COLS],  test_df[TARGET_COL].astype(int)

    base = build_base_model()
    base.fit(X_train, y_train)

    # calibrate using a held-out calibration slice
    calibrated = CalibratedClassifierCV(base, method="sigmoid", cv="prefit")
    calibrated.fit(X_calib, y_calib)

    # evaluate on test
    p_test = calibrated.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test, p_test)

    model_dir = Path(args.model_dir); model_dir.mkdir(parents=True, exist_ok=True)
    report_dir = Path(args.report_dir); report_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(base, model_dir / "model.pkl")
    joblib.dump(calibrated, model_dir / "calibrator.pkl")

    metadata = {
        "features": NUM_COLS,
        "target": TARGET_COL,
        "time_col": TIME_COL,
        "split_info": split_info,
        "metrics_test": metrics
    }
    (model_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    (report_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print("Saved model artifacts to:", model_dir)
    print("Test metrics:", json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
