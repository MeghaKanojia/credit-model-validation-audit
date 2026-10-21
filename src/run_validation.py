"""
Validation orchestrator.

Independently reproduces the champion WoE scorecard and LightGBM challenger
from the credit-risk-scorecard codebase, then runs a full suite of model
validation checks against them: data quality, conceptual soundness,
discrimination/calibration backtesting, overfitting, population stability
(PSI), multicollinearity (VIF), sensitivity analysis, and stress testing.
Writes all artefacts and a markdown Validation Report to outputs/.

Usage:
    python src/run_validation.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from load_model import load_model_under_review, SOURCE_REPO
from data_quality import assess as data_quality_assess
from stability import psi_table
from multicollinearity import vif_table
from backtesting import overfitting_gap, hosmer_lemeshow
from sensitivity import run as run_sensitivity
from stress_testing import run as run_stress_test

sys.path.insert(0, str(SOURCE_REPO / "src"))
from woe_scorecard import score_points  # noqa: E402


def main():
    root = Path(__file__).resolve().parent.parent
    out_dir = root / "outputs"
    fig_dir = out_dir / "figures"
    out_dir.mkdir(exist_ok=True)
    fig_dir.mkdir(exist_ok=True)

    print("Reproducing model under review from the source repository...")
    m = load_model_under_review()
    X_tr, X_te, y_tr, y_te = m["X_tr"], m["X_te"], m["y_tr"], m["y_te"]

    print("Running data quality assessment...")
    dq = data_quality_assess(pd.concat([X_tr, X_te]))
    dq.to_csv(out_dir / "data_quality.csv", index=False)

    print("Computing Population Stability Index (train vs test)...")
    psi_df = psi_table(X_tr, X_te, m["scores_tr"], m["scores_te"])
    psi_df.to_csv(out_dir / "psi_features.csv", index=False)

    print("Computing multicollinearity (VIF)...")
    vif_df = vif_table(X_tr)
    vif_df.to_csv(out_dir / "vif.csv", index=False)

    print("Backtesting discrimination, calibration, and overfitting...")
    sc_gap = overfitting_gap(y_tr, m["sc_pd_tr"], y_te, m["sc_pd_te"], "WoE scorecard")
    gbm_gap = overfitting_gap(y_tr, m["gbm_pd_tr"], y_te, m["gbm_pd_te"], "LightGBM challenger")
    overfit_df = pd.DataFrame([sc_gap, gbm_gap])
    overfit_df.to_csv(out_dir / "overfitting_gap.csv", index=False)

    hl_bands, hl_chi2, hl_dof, hl_p = hosmer_lemeshow(y_te.values, m["sc_pd_te"])
    hl_bands.to_csv(out_dir / "backtest_bands.csv", index=False)

    print("Running sensitivity analysis...")
    sample = X_te.sample(min(20000, len(X_te)), random_state=1)
    sens_df = run_sensitivity(m["scorecard"], score_points, sample)
    sens_df.to_csv(out_dir / "sensitivity.csv", index=False)

    print("Running stress tests...")
    predict_fn = lambda sc, X: sc.predict_proba(X)[:, 1]
    stress_df = run_stress_test(m["scorecard"], predict_fn, X_te, y_te.values, m["ead_te"])
    stress_df.to_csv(out_dir / "stress_test.csv", index=False)

    suspicious = m["iv_table"][m["iv_table"]["predictive_strength"] == "suspicious"]

    print("Generating figures...")
    _plot_calibration(hl_bands, fig_dir)
    _plot_psi(psi_df, fig_dir)
    _plot_stress(stress_df, fig_dir)

    summary = {
        "n_train": int(len(X_tr)), "n_test": int(len(X_te)),
        "overfitting_gap": overfit_df.to_dict(orient="records"),
        "hosmer_lemeshow": {
            "chi2": round(hl_chi2, 2), "dof": hl_dof,
            "p_value": round(hl_p, 4) if hl_p == hl_p else None,
        },
        "psi_max": float(psi_df["psi"].max()),
        "psi_significant_shift_count": int((psi_df["flag"] == "significant_shift").sum()),
        "vif_high_count": int((vif_df["flag"] == "high").sum()),
        "suspicious_iv_features": suspicious["name"].tolist() if len(suspicious) else [],
        "data_quality_high_missing": dq[dq["missing_pct"] > 0.3]["feature"].tolist(),
    }
    with open(out_dir / "validation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("Writing validation report...")
    _write_report(out_dir, summary, dq, psi_df, vif_df, overfit_df, hl_bands,
                   hl_chi2, hl_dof, hl_p, sens_df, stress_df, suspicious)

    print(f"\nDone. See {out_dir / 'validation_report.md'}")


def _plot_calibration(bands, fig_dir):
    plt.figure(figsize=(8, 5))
    plt.plot(bands["band_rank"], bands["observed_rate"], "o-", label="Observed default rate")
    plt.plot(bands["band_rank"], bands["mean_pd"], "s--", label="Predicted PD")
    plt.xlabel("PD decile (1 = lowest risk)"); plt.ylabel("Rate")
    plt.title("Backtest: observed vs predicted default rate by PD decile")
    plt.legend(); plt.tight_layout()
    plt.savefig(fig_dir / "calibration_backtest.png", dpi=120); plt.close()


def _plot_psi(psi_df, fig_dir):
    top = psi_df[psi_df["feature"] != "SCORE (model output)"].head(15).iloc[::-1]
    color_map = {"stable": "steelblue", "moderate_shift": "orange", "significant_shift": "crimson"}
    colors = top["flag"].astype(str).map(color_map)
    plt.figure(figsize=(8, 6))
    plt.barh(top["feature"], top["psi"], color=colors)
    plt.axvline(0.10, color="orange", linestyle="--", linewidth=1)
    plt.axvline(0.25, color="crimson", linestyle="--", linewidth=1)
    plt.xlabel("PSI (train vs test)"); plt.title("Population Stability Index by feature")
    plt.tight_layout(); plt.savefig(fig_dir / "psi_by_feature.png", dpi=120); plt.close()


def _plot_stress(stress_df, fig_dir):
    plt.figure(figsize=(7, 5))
    plt.bar(stress_df["scenario"], stress_df["portfolio_expected_loss"], color="steelblue")
    plt.ylabel("Portfolio Expected Loss"); plt.title("Expected Loss under stress scenarios")
    plt.xticks(rotation=15); plt.tight_layout()
    plt.savefig(fig_dir / "stress_test_el.png", dpi=120); plt.close()


def _write_report(out_dir, summary, dq, psi_df, vif_df, overfit_df, hl_bands,
                   hl_chi2, hl_dof, hl_p, sens_df, stress_df, suspicious):
    findings = []
    if summary["psi_significant_shift_count"] > 0:
        findings.append(
            f"- {summary['psi_significant_shift_count']} feature(s) show a significant "
            f"train-vs-test population shift (PSI > 0.25). See Section 4."
        )
    if summary["vif_high_count"] > 0:
        findings.append(
            f"- {summary['vif_high_count']} feature(s) show high multicollinearity (VIF > 10). "
            f"See Section 5."
        )
    if summary["suspicious_iv_features"]:
        findings.append(
            f"- {len(summary['suspicious_iv_features'])} feature(s) have a suspiciously high "
            f"Information Value (> 0.5): {', '.join(summary['suspicious_iv_features'])}. "
            f"This can indicate target leakage and should be investigated before "
            f"the model is relied upon. See Section 3."
        )
    if summary["data_quality_high_missing"]:
        findings.append(
            f"- {len(summary['data_quality_high_missing'])} feature(s) are missing in over 30% "
            f"of records, which the champion model's binning silently absorbs into a "
            f"'missing' bin. Business sign-off is needed on whether that is an acceptable "
            f"risk treatment. See Section 2."
        )
    hl_p_ok = hl_p is not None and hl_p == hl_p
    if hl_p_ok and hl_p < 0.05:
        findings.append(
            f"- The Hosmer-Lemeshow test rejects calibration (chi2={hl_chi2:.1f}, "
            f"dof={hl_dof}, p={hl_p:.4f} < 0.05): predicted PD does not statistically "
            f"match observed default rates across deciles. See Section 3."
        )
    gap_row = next(r for r in summary["overfitting_gap"] if r["model"] == "LightGBM challenger")
    if gap_row["gini_gap"] > 0.05:
        findings.append(
            f"- The LightGBM challenger shows a {gap_row['gini_gap']:.3f} Gini gap between "
            f"train ({gap_row['train_gini']}) and test ({gap_row['test_gini']}), consistent "
            f"with some overfitting. It is used only as a benchmark, not for decisioning, "
            f"which limits the materiality of this finding. See Section 6."
        )
    if not findings:
        findings.append("- No material issues identified across the checks performed.")

    verdict = (
        "Conceptually sound, with findings listed below that should be tracked "
        "as model limitations rather than blocking issues."
        if len(findings) <= 2 else
        "Conceptually reasonable but with multiple findings that warrant follow-up "
        "before the model is relied upon without additional controls."
    )

    lines = [
        "# Independent Model Validation Report",
        "",
        "**Model under review:** Home Credit Default Risk WoE scorecard (champion) and "
        "LightGBM challenger, from the `credit-risk-scorecard` repository.",
        "",
        "**Validation scope:** data quality, conceptual soundness, discrimination and "
        "calibration backtesting, overfitting, population stability (PSI), "
        "multicollinearity (VIF), sensitivity analysis, and stress testing. All checks "
        "were performed by independently reproducing the model from the development "
        "team's codebase on the same train/test split, then assessing it from first "
        "principles rather than reusing the development team's own reported metrics.",
        "",
        "---",
        "",
        "## 1. Executive Summary",
        "",
        f"Sample: {summary['n_train']:,} training / {summary['n_test']:,} test applicants.",
        "",
        f"**Overall assessment: {verdict}**",
        "",
        "**Key findings:**",
        "",
        *findings,
        "",
        "---",
        "",
        "## 2. Data Quality Assessment",
        "",
        "Missingness, type consistency, and robust outlier rate (values beyond 5 "
        "median-absolute-deviations) for every model input:",
        "",
        dq.to_markdown(index=False),
        "",
        "---",
        "",
        "## 3. Conceptual Soundness",
        "",
        "Information Value (IV) ranks each feature's standalone predictive strength. "
        "IV in the 'strong' or 'medium' range is expected for a well-built scorecard; "
        "IV above 0.5 ('suspicious') more often signals target leakage than genuine "
        "signal and should be investigated.",
        "",
        (
            f"Suspicious-IV features: {', '.join(summary['suspicious_iv_features'])}."
            if summary["suspicious_iv_features"] else
            "No features fall in the suspicious IV range (> 0.5)."
        ),
        "",
        "**Discrimination and calibration backtest** (held-out test set, PD deciles, "
        "1 = lowest predicted risk):",
        "",
        hl_bands.to_markdown(index=False),
        "",
        f"Hosmer-Lemeshow goodness-of-fit: chi2 = {hl_chi2:.2f}, dof = {hl_dof}, "
        f"p-value = {('%.4f' % hl_p) if hl_p_ok else 'n/a'} "
        f"({'calibration not rejected at the 5% level' if (hl_p_ok and hl_p >= 0.05) else 'calibration rejected at the 5% level' if hl_p_ok else 'not computable'}).",
        "",
        "![Calibration backtest](figures/calibration_backtest.png)",
        "",
        "---",
        "",
        "## 4. Population Stability (PSI, train vs test)",
        "",
        "PSI < 0.10 stable, 0.10-0.25 moderate shift (monitor), > 0.25 significant "
        "shift (recalibration may be needed). This dataset has no application-date "
        "field, so train-vs-test stands in for a genuine out-of-time comparison.",
        "",
        psi_df.head(15).to_markdown(index=False),
        "",
        "![PSI by feature](figures/psi_by_feature.png)",
        "",
        "---",
        "",
        "## 5. Multicollinearity (VIF)",
        "",
        "VIF > 5 moderate concern, > 10 severe. The WoE binning step tempers the "
        "practical impact of collinearity on the fitted scorecard, but raw-feature "
        "VIF is still reported here as the independent-input-space view a validator "
        "needs, separate from how the champion model internally transforms them.",
        "",
        vif_df.to_markdown(index=False),
        "",
        "---",
        "",
        "## 6. Overfitting Check (train vs test performance)",
        "",
        overfit_df.to_markdown(index=False),
        "",
        "---",
        "",
        "## 7. Sensitivity Analysis",
        "",
        "Score shift (in scorecard points) from perturbing one input at a time on a "
        "20,000-applicant sample, holding all other inputs fixed. A high "
        "`sensitivity_ratio` (mean absolute shift relative to the natural spread of "
        "scores) flags a feature the model may be disproportionately reactive to.",
        "",
        sens_df.to_markdown(index=False),
        "",
        "---",
        "",
        "## 8. Stress Testing",
        "",
        "Portfolio-level Expected Loss (PD x LGD x EAD, LGD = 0.45) under two combined "
        "adverse scenarios applied to the full test portfolio simultaneously, modelled "
        "on the kind of income/collateral shock a Pillar 2 stress test would apply:",
        "",
        stress_df.to_markdown(index=False),
        "",
        "![Expected Loss under stress](figures/stress_test_el.png)",
        "",
        "---",
        "",
        "## 9. Limitations of This Validation",
        "",
        "- Train/test split is a random hold-out, not a true out-of-time sample; PSI "
        "and backtesting results should be revisited once vintage/time data is "
        "available.",
        "- Stress scenarios are illustrative shock magnitudes, not calibrated to a "
        "specific regulator-mandated macroeconomic scenario.",
        "- LGD is a fixed 0.45 Basel proxy and EAD uses credit amount as a proxy, "
        "both simplifying assumptions inherited from the model under review, not "
        "independently re-derived here.",
        "",
        "---",
        "",
        "## 10. Recommendations for Model Governance",
        "",
        "1. Track this report's findings (Section 1) in the model's limitations "
        "log and revisit at the next scheduled review.",
        "2. Investigate any suspicious-IV features for leakage before relying on "
        "the model for a go-live decision.",
        "3. Re-run PSI once genuine out-of-time data is available; a random "
        "train/test split understates how much real-world drift could occur.",
        "4. Extend stress testing to a wider range of severities and, where "
        "possible, align scenario design with the Bank's official stress-testing "
        "methodology rather than illustrative shocks.",
        "",
    ]

    with open(out_dir / "validation_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
