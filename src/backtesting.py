"""
Backtesting: (1) discrimination/calibration on held-out test data by PD
decile, with a Hosmer-Lemeshow-style chi-square goodness-of-fit test for
whether predicted PD matches observed default rate; (2) an overfitting check
comparing train vs test performance.
"""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve


def _ks(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(tpr - fpr))


def performance(y_true, y_pd) -> dict:
    auc = roc_auc_score(y_true, y_pd)
    return {
        "auc": round(float(auc), 4),
        "gini": round(float(2 * auc - 1), 4),
        "ks": round(_ks(y_true, y_pd), 4),
    }


def overfitting_gap(y_tr, pd_tr, y_te, pd_te, model_name: str) -> dict:
    """A large train-vs-test performance gap suggests the model has
    memorised training-set noise rather than learned a generalisable
    relationship."""
    tr, te = performance(y_tr, pd_tr), performance(y_te, pd_te)
    return {
        "model": model_name,
        "train_auc": tr["auc"], "test_auc": te["auc"], "auc_gap": round(tr["auc"] - te["auc"], 4),
        "train_gini": tr["gini"], "test_gini": te["gini"], "gini_gap": round(tr["gini"] - te["gini"], 4),
    }


def hosmer_lemeshow(y_true, y_pd, n_bands: int = 10):
    """Groups applicants into PD deciles and tests observed vs. expected
    default counts with a chi-square goodness-of-fit statistic. A low
    p-value (< 0.05) means predicted probabilities are NOT well calibrated
    against actual outcomes under this band structure."""
    df = pd.DataFrame({"y": y_true, "pd": y_pd})
    df["band"] = pd.qcut(df["pd"], q=n_bands, duplicates="drop")
    g = (
        df.groupby("band", observed=True)
        .agg(n=("y", "size"), observed=("y", "sum"), mean_pd=("pd", "mean"))
        .reset_index()
    )
    g["expected"] = g["n"] * g["mean_pd"]
    g["observed_rate"] = (g["observed"] / g["n"]).round(4)
    e = g["expected"].clip(lower=1e-6)
    n_minus_e = (g["n"] - g["expected"]).clip(lower=1e-6)
    g["chi_term"] = (g["observed"] - g["expected"]) ** 2 * (1 / e + 1 / n_minus_e)

    chi2 = float(g["chi_term"].sum())
    dof = len(g) - 2
    p_value = float(1 - stats.chi2.cdf(chi2, dof)) if dof > 0 else float("nan")
    g["band_rank"] = range(1, len(g) + 1)
    g["mean_pd"] = g["mean_pd"].round(4)
    return g.drop(columns="chi_term"), chi2, dof, p_value
