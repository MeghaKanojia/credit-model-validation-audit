"""
Multicollinearity check via Variance Inflation Factor (VIF) on the model's
numeric inputs. VIF > 5 is commonly flagged as moderate concern, > 10 as
severe: redundant or collinear features can destabilise coefficient
estimates in the logistic-regression champion, even though the WoE binning
step tempers this somewhat.
"""
import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor


def vif_table(X: pd.DataFrame) -> pd.DataFrame:
    num = X.select_dtypes(include=[np.number]).copy()
    num = num.fillna(num.median())
    num = num.loc[:, num.std() > 0]  # zero-variance columns make VIF undefined

    rows = []
    arr = num.values
    for i, col in enumerate(num.columns):
        try:
            vif = variance_inflation_factor(arr, i)
        except Exception:
            vif = np.nan
        rows.append({"feature": col, "vif": round(float(vif), 2) if np.isfinite(vif) else None})

    out = pd.DataFrame(rows).sort_values("vif", ascending=False, na_position="last").reset_index(drop=True)
    out["flag"] = pd.cut(out["vif"], bins=[-1, 5, 10, np.inf], labels=["low", "moderate", "high"])
    return out
