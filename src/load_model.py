"""
Loads the champion (WoE scorecard) and challenger (LightGBM) models under
review by independently reproducing them from the model development team's
codebase (the sibling `credit-risk-scorecard` repository), using the exact
same feature set, train/test split, and random seed as the original run.

Reproducing a model from its specification, rather than trusting pre-computed
outputs, is itself a validation step: an independent party must be able to
rebuild the model and get matching results before anything else is assessed.
"""
import sys
from pathlib import Path

from sklearn.model_selection import train_test_split

SOURCE_REPO = Path(__file__).resolve().parent.parent.parent / "credit-risk-scorecard"
sys.path.insert(0, str(SOURCE_REPO / "src"))

from feature_engineering import build_features  # noqa: E402
from woe_scorecard import fit_scorecard, score_points  # noqa: E402
from challenger_model import fit_challenger  # noqa: E402

MODEL_FEATURES = [
    "EXT_SOURCE_2", "EXT_SOURCE_3", "EXT_SOURCE_MEAN",
    "CREDIT_INCOME_RATIO", "ANNUITY_INCOME_RATIO", "CREDIT_ANNUITY_RATIO",
    "AGE_YEARS", "EMPLOYED_YEARS", "AMT_INCOME_TOTAL", "AMT_CREDIT",
    "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS", "CODE_GENDER", "CNT_CHILDREN",
    "BUREAU_COUNT", "BUREAU_ACTIVE_COUNT", "BUREAU_DEBT_CREDIT_RATIO",
    "BUREAU_OVERDUE_MAX", "PREV_COUNT", "PREV_REFUSED_RATIO",
    "CC_UTILISATION_MEAN", "CC_UTILISATION_MAX", "CC_DPD_MAX",
    "INST_PAYMENT_RATIO_MEAN", "INST_DAYS_LATE_MEAN", "INST_DAYS_LATE_MAX",
]


def load_model_under_review():
    """Rebuild the champion scorecard and challenger exactly as the model
    development team specified them. Returns a dict with everything the
    validation checks need."""
    df = build_features(str(SOURCE_REPO / "data"))
    feats = [f for f in MODEL_FEATURES if f in df.columns]
    X = df[feats]
    y = df["TARGET"]
    exposure = df["AMT_CREDIT"].values

    X_tr, X_te, y_tr, y_te, ead_tr, ead_te = train_test_split(
        X, y, exposure, test_size=0.25, stratify=y, random_state=42
    )

    print("Fitting WoE scorecard (champion)...")
    scorecard, iv_table, sc_pd_tr, sc_pd_te = fit_scorecard(X_tr, y_tr, X_te)
    scores_te = score_points(scorecard, X_te)
    scores_tr = score_points(scorecard, X_tr)

    print("Fitting LightGBM (challenger)...")
    gbm, gbm_pd_tr, gbm_pd_te = fit_challenger(X_tr, y_tr, X_te)

    return {
        "X_tr": X_tr, "X_te": X_te, "y_tr": y_tr, "y_te": y_te,
        "ead_tr": ead_tr, "ead_te": ead_te,
        "scorecard": scorecard, "iv_table": iv_table,
        "sc_pd_tr": sc_pd_tr, "sc_pd_te": sc_pd_te,
        "scores_tr": scores_tr, "scores_te": scores_te,
        "gbm": gbm, "gbm_pd_tr": gbm_pd_tr, "gbm_pd_te": gbm_pd_te,
        "feats": feats,
    }
