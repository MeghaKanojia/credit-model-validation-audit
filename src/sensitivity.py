"""
Sensitivity analysis: perturbs a shortlist of economically meaningful inputs
across a sample of applicants and measures how much the model's predicted
score moves. A model whose output swings disproportionately to a small,
plausible change in one input is fragile and should be flagged for review.
"""
import numpy as np
import pandas as pd

SENSITIVITY_FEATURES = {
    # feature: perturbation deltas to apply (absolute for bounded [0,1]
    # external scores and age in years, multiplicative for amounts/ratios)
    "EXT_SOURCE_2": [-0.10, -0.05, 0.05, 0.10],
    "EXT_SOURCE_3": [-0.10, -0.05, 0.05, 0.10],
    "AMT_INCOME_TOTAL": [-0.20, -0.10, 0.10, 0.20],
    "CREDIT_INCOME_RATIO": [-0.20, -0.10, 0.10, 0.20],
    "AGE_YEARS": [-5, -2, 2, 5],
}


def run(scorecard, score_fn, X_sample: pd.DataFrame) -> pd.DataFrame:
    base_scores = score_fn(scorecard, X_sample)
    base_std = float(np.std(base_scores))
    rows = []
    for feat, deltas in SENSITIVITY_FEATURES.items():
        if feat not in X_sample.columns:
            continue
        for delta in deltas:
            X_pert = X_sample.copy()
            if feat in ("EXT_SOURCE_2", "EXT_SOURCE_3"):
                X_pert[feat] = (X_pert[feat] + delta).clip(0, 1)
                delta_label = f"{delta:+.2f} (absolute)"
            elif feat == "AGE_YEARS":
                X_pert[feat] = (X_pert[feat] + delta).clip(lower=18)
                delta_label = f"{delta:+d} years"
            else:
                X_pert[feat] = X_pert[feat] * (1 + delta)
                delta_label = f"{delta:+.0%}"

            pert_scores = score_fn(scorecard, X_pert)
            shift = pert_scores - base_scores
            rows.append({
                "feature": feat,
                "perturbation": delta_label,
                "mean_score_shift": round(float(np.mean(shift)), 2),
                "mean_abs_score_shift": round(float(np.mean(np.abs(shift))), 2),
                "sensitivity_ratio": round(float(np.mean(np.abs(shift)) / base_std), 4),
            })
    return pd.DataFrame(rows)
