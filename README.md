# Independent Credit Model Validation & Backtesting Audit

An independent Model Validation exercise on a credit risk model: this repo
takes the WoE scorecard (champion) and LightGBM challenger built in
[`credit-risk-scorecard`](https://github.com/MeghaKanojia/credit-risk-scorecard)
as the "model under review" and audits it the way a bank's Model Validation
function would, rather than reusing the development team's own reported
numbers.

## What this demonstrates

- Independent reproduction of a model from its specification (code +
  train/test seed), rather than trusting pre-computed development-team
  outputs
- Data quality and missingness assessment
- Conceptual soundness review (Information Value screening for suspiciously
  strong / leaky features)
- Discrimination and calibration backtesting, including a
  Hosmer-Lemeshow goodness-of-fit test on observed-vs-predicted default
  rate by PD decile
- Overfitting diagnostics (train vs test performance gap)
- Population Stability Index (PSI) to flag feature and score drift
- Multicollinearity screening via Variance Inflation Factor (VIF)
- Sensitivity analysis (single-input perturbation) and stress testing
  (combined adverse-scenario shocks) on portfolio Expected Loss
- A structured Validation Report with an executive summary, findings, and
  model governance recommendations

## How this maps to real bank model validation practice

| Check performed here | Real-world equivalent |
|---|---|
| Independent pipeline reproduction | Confirming a model can be rebuilt from its spec before anything else is assessed |
| Missingness / outlier assessment | Data quality sign-off ahead of model use |
| Suspicious-IV screening | Target-leakage review, conceptual soundness |
| Hosmer-Lemeshow backtest | IFRS 9 / IRB-style discrimination and calibration testing |
| PSI | Population stability monitoring between development and application samples |
| VIF | Coefficient stability / redundant-feature review |
| Sensitivity analysis | Robustness challenge on individual risk drivers |
| Stress testing | Pillar 2 / ICAAP-style adverse scenario testing |

## Running it

This repo does not redistribute the dataset or the model's own code; it
imports the sibling repo directly, so both must be cloned side by side:

    DevProjects/
      credit-risk-scorecard/        (model under review, data + src/)
      credit-model-validation-audit/  (this repo)

With the Kaggle Home Credit Default Risk CSVs already placed in
`credit-risk-scorecard/data/` (see that repo's README):

    pip install -r requirements.txt
    python src/run_validation.py

All findings, tables, and figures are written to `outputs/`, including the
full `validation_report.md`.

## Repository structure

    src/
      load_model.py          reproduces the champion + challenger from the source repo
      data_quality.py        missingness, dtype, outlier checks
      stability.py           Population Stability Index (PSI)
      multicollinearity.py   Variance Inflation Factor (VIF)
      backtesting.py         discrimination/calibration backtest, overfitting gap
      sensitivity.py         single-input perturbation analysis
      stress_testing.py      combined adverse-scenario stress testing
      run_validation.py      orchestrates all checks, writes outputs/ and the report
    outputs/
      validation_report.md       full written validation report
      validation_summary.json    machine-readable findings summary
      data_quality.csv, psi_features.csv, vif.csv, overfitting_gap.csv,
      backtest_bands.csv, sensitivity.csv, stress_test.csv
      figures/                   calibration, PSI, and stress-test charts

## Important

This is a portfolio project simulating a Model Validation workflow, not an
actual regulatory submission. Stress scenarios are illustrative shock
magnitudes, not calibrated to a specific regulator-mandated macroeconomic
scenario, and the train/test split is a random hold-out rather than a true
out-of-time sample (the dataset has no application-date field). Both
limitations are called out explicitly in the generated report.
