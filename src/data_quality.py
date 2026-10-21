"""
Data quality and missingness assessment on the model's feature set, plus a
check for distributional outliers (values beyond a robust z-score bound,
using the median absolute deviation so a handful of extreme values can't
distort the estimate the way a mean/std-based check would).
"""
import numpy as np
import pandas as pd


def assess(X: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in X.columns:
        s = X[col]
        missing_pct = s.isna().mean()
        outlier_pct = np.nan
        if pd.api.types.is_numeric_dtype(s):
            med = s.median()
            mad = (s - med).abs().median()
            if mad and mad > 0:
                robust_z = (s - med).abs() / (1.4826 * mad)
                outlier_pct = float((robust_z > 5).mean())
        rows.append({
            "feature": col,
            "dtype": str(s.dtype),
            "missing_pct": round(float(missing_pct), 4),
            "outlier_pct_robust_z5": None if pd.isna(outlier_pct) else round(outlier_pct, 4),
            "n_unique": int(s.nunique(dropna=True)),
        })
    return (
        pd.DataFrame(rows)
        .sort_values("missing_pct", ascending=False)
        .reset_index(drop=True)
    )
