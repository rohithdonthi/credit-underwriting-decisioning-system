# src/monitoring/drift.py
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

def psi(expected, actual, bins=10):
    expected = np.asarray(expected)
    actual = np.asarray(actual)

    # bin edges on expected distribution
    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.quantile(expected, quantiles)
    edges[0], edges[-1] = -np.inf, np.inf

    e_counts = np.histogram(expected, bins=edges)[0].astype(float)
    a_counts = np.histogram(actual, bins=edges)[0].astype(float)

    e_perc = np.clip(e_counts / max(e_counts.sum(), 1), 1e-6, 1)
    a_perc = np.clip(a_counts / max(a_counts.sum(), 1), 1e-6, 1)
    return float(np.sum((a_perc - e_perc) * np.log(a_perc / e_perc)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", type=str, required=True, help="baseline scored csv")
    ap.add_argument("--current", type=str, required=True, help="current scored csv")
    ap.add_argument("--out", type=str, default="reports/drift_psi.json")
    args = ap.parse_args()

    base = pd.read_csv(args.baseline)
    curr = pd.read_csv(args.current)

    report = {
        "psi_p_creditworthy": psi(base["p_creditworthy"], curr["p_creditworthy"]),
        "psi_credit_score": psi(base["credit_score"], curr["credit_score"]),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print("Saved drift report:", out)
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
