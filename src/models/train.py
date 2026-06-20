import argparse
import json
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

# Real credit risk features from your bureau dataset
NUM_COLS = [
    "fico8",                    # Primary credit score
    "revolving_utilization",    # Credit utilization (key risk signal)
    "delinq_24mo_count",        # Delinquency count
    "worst_delinq_24mo",        # Severity of worst delinquency
    "inquiries_12mo_hard",      # Hard inquiries
    "has_bankruptcy",           # Bankruptcy flag
    "has_collection",           # Collections flag
    "monthly_salary",           # Income
    "balance_to_income",        # Debt-to-income proxy
    "thin_file_flag",           # Thin file risk
    "months_since_oldest_account",  # Credit age
    "tradelines_total",         # Breadth of credit
]

def time_split(df: pd.DataFrame):
    d = df.copy()
    d[TIME_COL] = d[TIME_COL].astype(str)
    months = sorted(d[TIME_COL].unique())
    n = len(months)
    train_end = int(0.70 * n)
    calib_end = int(0.85 * n)
    train_months = set(months[:train_end])
    calib_months = set(months[train_end:calib_end])
    test_months  = set(months[calib_end:])
    return (
        d[d[TIME_COL].isin(train_months)],
        d[d[TIME_COL].isin(calib_months)],
        d[d[TIME_COL].isin(test_months)],
        {"train": sorted(train_months),
         "calib": sorted(calib_months),
         "test":  sorted(test_months)},
    )

def build_base_model():
    pre = ColumnTransformer([
        ("num", StandardScaler(), NUM_COLS)
    ])
    return Pipeline([
        ("pre", pre),
        ("clf", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_SEED
        ))
    ])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data",       type=str, default="data/sample/credit_sample.csv")
    ap.add_argument("--model_dir",  type=str, default="models")
    ap.add_argument("--report_dir", type=str, default="reports")
    args = ap.parse_args()

    df = pd.read_csv(args.data)

    # Validate required columns exist
    missing = [c for c in NUM_COLS + [TARGET_COL, TIME_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")

    print(f"Dataset: {df.shape[0]} rows, {df.shape[1]} cols")
    print(f"Features: {NUM_COLS}")
    print(f"Target distribution:\n{df[TARGET_COL].value_counts()}")

    train_df, calib_df, test_df, split_info = time_split(df)

    X_train, y_train = train_df[NUM_COLS], train_df[TARGET_COL].astype(int)
    X_calib, y_calib = calib_df[NUM_COLS], calib_df[TARGET_COL].astype(int)
    X_test,  y_test  = test_df[NUM_COLS],  test_df[TARGET_COL].astype(int)

    print(f"\nTrain: {len(X_train)} | Calib: {len(X_calib)} | Test: {len(X_test)}")

    base = build_base_model()
    base.fit(X_train, y_train)

    calibrated = CalibratedClassifierCV(base, method="sigmoid", cv=5)
    calibrated.fit(X_calib, y_calib)

    p_test  = calibrated.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test, p_test)
    print("\nTest metrics:", json.dumps(metrics, indent=2))

    model_dir  = Path(args.model_dir);  model_dir.mkdir(parents=True, exist_ok=True)
    report_dir = Path(args.report_dir); report_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(base,       model_dir / "model.pkl")
    joblib.dump(calibrated, model_dir / "calibrator.pkl")

    metadata = {
        "features":   NUM_COLS,
        "target":     TARGET_COL,
        "time_col":   TIME_COL,
        "split_info": split_info,
        "metrics":    metrics,
    }
    (model_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"\nSaved model artifacts to: {args.model_dir}")

if __name__ == "__main__":
    main()