"""
Track D extension: validates whether the entanglement-based diagnostic
(Objective 2 -- "a cheap, edit-native, pre-erasure diagnostic that could flag
which concepts will be hard to erase cleanly") actually predicts on concepts
it wasn't fit on, not just correlates in-sample. correlation_analysis.py's
Spearman rho is an in-sample statistic; a real diagnostic claim needs
leave-one-out cross-validation on top of that.

Small n (10 concepts, or 6 for the metrics that only have data for concepts
whose NEAR_DOMAIN_PAIRS partner is also in the completed set) means this is a
modest validation, not a strong one -- reported plainly, not oversold. Reads
artifacts/results.parquet (built by correlation_analysis.py -- run that
first if this file doesn't exist or is stale).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
RESULTS_PATH = ROOT / "artifacts" / "results.parquet"

# The pairs worth validating: metric/outcome combinations that showed real
# in-sample signal in correlation_analysis.py's output, not every possible
# combination -- a LOO pass on a pair with no in-sample correlation at all
# wouldn't tell us anything correlation_analysis.py didn't already.
PAIRS_TO_VALIDATE = [
    ("entanglement_cosine", "specificity_mmlu"),  # the significant one: rho=0.68, p=0.03, n=10
    ("entanglement_cosine", "specificity_simdomain"),  # borderline: rho=0.62, p=0.056, n=10
    ("entanglement_cosine", "efficacy"),
    ("entanglement_cosine_paired", "efficacy"),  # borderline pairwise: rho=-0.77, p=0.072, n=6
]


def loo_validate(df: pd.DataFrame, predictor_col: str, outcome_col: str) -> pd.DataFrame:
    """For each concept with both predictor_col and outcome_col non-null,
    fits a simple linear regression on every OTHER such concept, predicts
    this concept's outcome from its own predictor value alone, and records
    actual vs predicted. One row per held-out concept in the return value."""
    sub = df[["concept", predictor_col, outcome_col]].dropna().reset_index(drop=True)
    n = len(sub)
    rows = []
    for i in range(n):
        train = sub.drop(index=i)
        if len(train) < 2:
            continue
        slope, intercept = np.polyfit(train[predictor_col], train[outcome_col], deg=1)
        held_out = sub.iloc[i]
        predicted = slope * held_out[predictor_col] + intercept
        rows.append({
            "concept": held_out["concept"],
            predictor_col: held_out[predictor_col],
            f"actual_{outcome_col}": held_out[outcome_col],
            f"predicted_{outcome_col}": predicted,
        })
    return pd.DataFrame(rows)


def summarize(loo_df: pd.DataFrame, outcome_col: str) -> dict:
    actual = loo_df[f"actual_{outcome_col}"]
    predicted = loo_df[f"predicted_{outcome_col}"]
    if len(loo_df) < 3:
        rho, p = float("nan"), float("nan")
    else:
        rho, p = spearmanr(actual, predicted)
    mae = float(np.mean(np.abs(actual - predicted))) if len(loo_df) else float("nan")
    return {"n": len(loo_df), "loo_spearman_rho": rho, "loo_p_value": p, "mae": mae}


def main():
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(f"{RESULTS_PATH} not found -- run track_d_analysis/correlation_analysis.py first.")
    df = pd.read_parquet(RESULTS_PATH)

    summary_rows = []
    for predictor_col, outcome_col in PAIRS_TO_VALIDATE:
        loo_df = loo_validate(df, predictor_col, outcome_col)
        stats = summarize(loo_df, outcome_col)
        print(f"=== LOO: {predictor_col} -> {outcome_col} ===")
        print(loo_df.to_string(index=False))
        print(f"n={stats['n']}  LOO Spearman rho={stats['loo_spearman_rho']:.3f}  "
              f"p={stats['loo_p_value']:.3f}  MAE={stats['mae']:.3f}")
        print()
        summary_rows.append({"predictor": predictor_col, "outcome": outcome_col, **stats})

    summary_df = pd.DataFrame(summary_rows)
    summary_path = ROOT / "artifacts" / "diagnostic_validation.csv"
    summary_df.to_csv(summary_path, index=False)
    print("=== Summary (out-of-sample) ===")
    print(summary_df.to_string(index=False))
    print(f"Wrote LOO validation summary to {summary_path}")


if __name__ == "__main__":
    main()
