"""
Population Stability Index (PSI): the standard model-validation measure of
whether a feature's (or a score's) distribution has shifted between two
samples. Used here to compare train vs test as a proxy for an out-of-time
stability check (the dataset has no application-date field for a genuine
OOT split).

Industry rule of thumb: PSI < 0.10 no significant shift, 0.10-0.25 moderate
shift (monitor), > 0.25 significant shift (model may need recalibration).
"""
import numpy as np
import pandas as pd


def psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    expected = pd.Series(expected).dropna().values
    actual = pd.Series(actual).dropna().values
    breakpoints = np.unique(np.quantile(expected, np.linspace(0, 1, buckets + 1)))
    if len(breakpoints) < 3:
        return 0.0  # not enough distinct values to bucket meaningfully

    def _dist(a):
        counts, _ = np.histogram(a, bins=breakpoints)
        counts = counts.astype(float)
        counts[counts == 0] = 0.5  # avoid log(0) / divide-by-zero on empty buckets
        return counts / counts.sum()

    e_dist, a_dist = _dist(expected), _dist(actual)
    return float(np.sum((a_dist - e_dist) * np.log(a_dist / e_dist)))


def psi_table(X_tr: pd.DataFrame, X_te: pd.DataFrame, score_tr, score_te) -> pd.DataFrame:
    rows = []
    for col in X_tr.columns:
        if pd.api.types.is_numeric_dtype(X_tr[col]):
            rows.append({"feature": col, "psi": round(psi(X_tr[col].values, X_te[col].values), 4)})
    rows.append({"feature": "SCORE (model output)", "psi": round(psi(score_tr, score_te), 4)})
    out = pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
    out["flag"] = pd.cut(
        out["psi"], bins=[-1, 0.10, 0.25, np.inf],
        labels=["stable", "moderate_shift", "significant_shift"],
    )
    return out
