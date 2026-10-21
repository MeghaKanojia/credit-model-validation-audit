"""
Stress testing: applies an adverse macroeconomic scenario to the whole test
portfolio simultaneously (unlike sensitivity analysis, which perturbs one
feature at a time) and measures the resulting shift in portfolio-level mean
PD and Expected Loss -- the kind of exercise Pillar 2 / ICAAP stress testing
requires.
"""
import numpy as np
import pandas as pd

SCENARIOS = {
    "mild_downturn": {
        "AMT_INCOME_TOTAL": -0.10, "EXT_SOURCE_2": -0.05, "EXT_SOURCE_3": -0.05,
        "CREDIT_INCOME_RATIO": 0.10,
    },
    "severe_recession": {
        "AMT_INCOME_TOTAL": -0.25, "EXT_SOURCE_2": -0.15, "EXT_SOURCE_3": -0.15,
        "CREDIT_INCOME_RATIO": 0.30, "CC_UTILISATION_MEAN": 0.20,
    },
}


def apply_scenario(X: pd.DataFrame, scenario: dict) -> pd.DataFrame:
    X_s = X.copy()
    for feat, shock in scenario.items():
        if feat not in X_s.columns:
            continue
        if feat in ("EXT_SOURCE_2", "EXT_SOURCE_3"):
            X_s[feat] = (X_s[feat] + shock).clip(0, 1)
        else:
            X_s[feat] = X_s[feat] * (1 + shock)
    return X_s


def run(scorecard, predict_fn, X_test, y_test, ead_test, lgd: float = 0.45) -> pd.DataFrame:
    rows = []
    base_pd = predict_fn(scorecard, X_test)
    base_el = float(np.sum(base_pd * ead_test * lgd))
    rows.append({
        "scenario": "baseline", "mean_pd": round(float(np.mean(base_pd)), 4),
        "portfolio_expected_loss": round(base_el, 0), "el_change_pct": 0.0,
    })
    for name, scenario in SCENARIOS.items():
        X_s = apply_scenario(X_test, scenario)
        stressed_pd = predict_fn(scorecard, X_s)
        stressed_el = float(np.sum(stressed_pd * ead_test * lgd))
        rows.append({
            "scenario": name, "mean_pd": round(float(np.mean(stressed_pd)), 4),
            "portfolio_expected_loss": round(stressed_el, 0),
            "el_change_pct": round((stressed_el - base_el) / base_el * 100, 2),
        })
    return pd.DataFrame(rows)
