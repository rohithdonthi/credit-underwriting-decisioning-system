# src/data/make_dataset.py
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from src.config import RANDOM_SEED

def generate(n=5000, start="2023-01", months=12) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)

    # month labels like 2023-01, 2023-02...
    ym = pd.period_range(start=start, periods=months, freq="M").astype(str)
    ym_col = rng.choice(ym, size=n, replace=True)

    income = rng.lognormal(mean=10.5, sigma=0.45, size=n)  # synthetic
    util = np.clip(rng.normal(loc=0.35, scale=0.20, size=n), 0, 1)
    dti = np.clip(rng.normal(loc=0.28, scale=0.12, size=n), 0, 1)
    delinq_12m = rng.poisson(lam=0.3, size=n)
    inquiries_6m = rng.poisson(lam=1.0, size=n)
    tradelines = np.clip(rng.normal(loc=8, scale=4, size=n), 0, None).astype(int)

    thin_file = (tradelines < 3).astype(int)
    high_util = (util > 0.8).astype(int)

    # generate probability of being "good" with a simple ground-truth function
    logit = (
        2.2
        + 0.00002 * income
        - 2.4 * util
        - 1.8 * dti
        - 0.7 * delinq_12m
        - 0.25 * inquiries_6m
        + 0.12 * tradelines
        - 0.8 * thin_file
        - 0.9 * high_util
    )
    p_good = 1 / (1 + np.exp(-logit))
    good = (rng.random(n) < p_good).astype(int)

    df = pd.DataFrame({
        "ym": ym_col,
        "income": income,
        "utilization": util,
        "dti": dti,
        "delinq_12m": delinq_12m,
        "inquiries_6m": inquiries_6m,
        "tradelines": tradelines,
        "thin_file": thin_file,
        "high_util": high_util,
        "good": good,
    })
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="data/sample/credit_sample.csv")
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--months", type=int, default=12)
    args = ap.parse_args()

    df = generate(n=args.n, months=args.months)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Saved: {out} ({len(df)} rows)")

if __name__ == "__main__":
    main()
