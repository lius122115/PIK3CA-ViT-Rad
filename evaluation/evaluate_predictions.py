"""Summarize frozen-model predictions and bootstrap AUC confidence intervals."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def bootstrap_auc(y, p, seed=42, n_resamples=1000):
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    values = []
    for _ in range(n_resamples):
        idx = rng.integers(0, len(y), len(y))
        if np.unique(y[idx]).size < 2:
            continue
        values.append(roc_auc_score(y[idx], p[idx]))
    if not values:
        return np.nan, np.nan, np.nan
    return float(roc_auc_score(y, p)), float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions-dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--label", default="pik3ca_status")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-resamples", type=int, default=1000)
    args = ap.parse_args()
    rows = []
    for path in sorted(Path(args.predictions_dir).glob("*_predictions.csv")):
        df = pd.read_csv(path)
        if df.empty or df[args.label].nunique() < 2:
            continue
        auc, lo, hi = bootstrap_auc(df[args.label], df["probability"], args.seed, args.n_resamples)
        rows.append({"file": path.name, "auc": auc, "auc_ci_lower": lo, "auc_ci_upper": hi})
    if not rows:
        raise RuntimeError("No prediction files with both outcome classes were found.")
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)


if __name__ == "__main__":
    main()
