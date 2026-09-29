"""
Phase 7 - Machine learning model comparison for CreditWise AI.

Compares FOUR fixed-baseline classifiers on the APPLICATION-ONLY feature set
(excludes the lender-assigned loan_grade / loan_int_rate) using the exact
same 5-fold StratifiedKFold cross-validation as Phase 6:

    1. Logistic Regression
    2. Random Forest
    3. XGBoost
    4. CatBoost

The FULL-configuration Logistic Regression CV metrics from Phase 6 are kept
as a single reference benchmark row ONLY (no re-training on the FULL set).

Run with:
    python ml/compare_models.py

Split rules
-----------
    - ONE stratified holdout split exists (test_size=0.20, random_state=42,
      stratify=y), identical logic to Phases 5/6.
    - Cross-validation uses ONLY X_train / y_train. The holdout
      X_test / y_test is never seen by any model.

Pipeline (per fold)
-------------------
    Feature engineering (deterministic, done once up front)
        -> Preprocessor (median imputation + one-hot + scaling)
        -> Estimator (fixed baseline, random seed 42)

The preprocessor is refitted FROM SCRATCH on each fold's training portion
only (it stays INSIDE the Pipeline), so imputation/scaling/encoding can
never leak validation information.

Metric priority (default is the minority/risk class)
----------------------------------------------------
    PR-AUC -> ROC-AUC -> Recall -> F1  (accuracy is NOT used for selection)

Design rules (deliberately enforced)
------------------------------------
    - Default probability threshold 0.50
    - No hyperparameter tuning (GridSearch/Randomized/Optuna)
    - No SMOTE / ADASYN / undersampling / class_weight / threshold tuning
    - No SHAP
    - No production models are saved - results are printed only
    - The test holdout set is never evaluated
"""

import sys
from pathlib import Path

# Make the project root importable so `ml.*` modules work no matter from
# which folder this script is executed (e.g. python ml/compare_models.py).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from ml.cross_validate_baseline import (
    N_SPLITS,
    print_fold_scores,
    run_cv_scores,
)
from ml.data_cleaning import clean_credit_data
from ml.evaluate import SCALAR_METRIC_LABELS
from ml.feature_engineering import engineer_features
from ml.preprocessing import TARGET
from ml.train_baseline import RANDOM_STATE, build_model, get_split, load_data

# ---------------------------------------------------------------------------
# Global configuration
# ---------------------------------------------------------------------------
CV_CONFIG = "application_only"

# Metrics used for model selection (accuracy intentionally excluded).
SELECTION_METRICS = ["pr_auc", "roc_auc", "recall", "f1"]

# The FULL Logistic Regression CV metrics from Phase 6 (printed-only there).
# Used as a reference benchmark row; Phase 7 never retrains the FULL config.
PHASE6_FULL_LR_REFERENCE = {
    "roc_auc": {"mean": 0.8723, "std": 0.0053},
    "pr_auc": {"mean": 0.7174, "std": 0.0123},
    "precision": {"mean": 0.7493, "std": 0.0189},
    "recall": {"mean": 0.5317, "std": 0.0107},
    "f1": {"mean": 0.6220, "std": 0.0116},
}

# Fixed baseline estimators - same seed everywhere, no tuning.
MODEL_REGISTRY = {
    "logistic_regression": {
        "label": "Logistic Regression",
        "feature_set": "Application-only",
        "estimator": lambda: LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    },
    "random_forest": {
        "label": "Random Forest",
        "feature_set": "Application-only",
        "estimator": lambda: RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1
        ),
    },
    "xgboost": {
        "label": "XGBoost",
        "feature_set": "Application-only",
        "estimator": lambda: XGBClassifier(
            n_estimators=100, random_state=RANDOM_STATE, eval_metric="logloss"
        ),
    },
    "catboost": {
        "label": "CatBoost",
        "feature_set": "Application-only",
        "estimator": lambda: CatBoostClassifier(
            iterations=100, random_seed=RANDOM_STATE, verbose=False, allow_writing_files=False
        ),
    },
}

CANDIDATE_NAMES = list(MODEL_REGISTRY.keys())
TREE_NAMES = ["random_forest", "xgboost", "catboost"]


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------
def build_model_pipeline(model_name):
    """
    Build (unfitted) the full Pipeline for a registered model.

    Preprocessor = the SAME application-only ColumnTransformer used in Phases
    5/6; only the estimator changes. Keeping one shared feature space keeps
    the comparison fair and the code minimal.
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_name}'.")
    estimator = MODEL_REGISTRY[model_name]["estimator"]()
    model, _ = build_model(CV_CONFIG, estimator=estimator)
    return model


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------
def prepare_data():
    """
    Load, clean, engineer, and apply the Phase 5 holdout split.

    Returns (X_train, X_test, y_train, y_test). Only X_train/y_train may be
    used for cross-validation.
    """
    df = load_data()
    cleaned, _ = clean_credit_data(df)
    engineered, _ = engineer_features(cleaned)

    X = engineered.drop(columns=[TARGET])
    y = engineered[TARGET]
    return get_split(X, y)


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------
def run_comparison(X_train, y_train):
    """
    5-fold CV for every candidate model on X_train/y_train only.

    Returns {model_name: per-fold DataFrame} including per-fold fit times.
    """
    results = {}
    for name in CANDIDATE_NAMES:
        print(f"\nRunning 5-fold StratifiedKFold CV for model: {name}")
        results[name] = run_cv_scores(
            X_train,
            y_train,
            lambda X_fold_train, y_fold_train, n=name: build_model_pipeline(n),
            n_splits=N_SPLITS,
            time_fits=True,
        )
    return results


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------
def make_comparison_table(results):
    """
    Final comparison table sorted primarily by PR-AUC (descending).

    Columns:
        Model | Feature Set | ROC-AUC Mean | ROC-AUC Std | PR-AUC Mean |
        PR-AUC Std | Precision Mean | Recall Mean | F1 Mean | Runtime
    """
    rows = []
    for name in CANDIDATE_NAMES:
        fold_df = results[name]
        rows.append(
            {
                "Model": MODEL_REGISTRY[name]["label"],
                "Feature Set": MODEL_REGISTRY[name]["feature_set"],
                "ROC-AUC Mean": fold_df["roc_auc"].mean(),
                "ROC-AUC Std": fold_df["roc_auc"].std(),
                "PR-AUC Mean": fold_df["pr_auc"].mean(),
                "PR-AUC Std": fold_df["pr_auc"].std(),
                "Precision Mean": fold_df["precision"].mean(),
                "Recall Mean": fold_df["recall"].mean(),
                "F1 Mean": fold_df["f1"].mean(),
                "Runtime": f"{fold_df['fit_time_s'].mean():.2f}s",
            }
        )

    # FULL Logistic Regression - Phase 6 CV reference row (no retraining).
    ref = PHASE6_FULL_LR_REFERENCE
    rows.append(
        {
            "Model": "Logistic Regression (FULL ref)",
            "Feature Set": "Full (bank-assigned)",
            "ROC-AUC Mean": ref["roc_auc"]["mean"],
            "ROC-AUC Std": ref["roc_auc"]["std"],
            "PR-AUC Mean": ref["pr_auc"]["mean"],
            "PR-AUC Std": ref["pr_auc"]["std"],
            "Precision Mean": ref["precision"]["mean"],
            "Recall Mean": ref["recall"]["mean"],
            "F1 Mean": ref["f1"]["mean"],
            "Runtime": "n/a (Phase 6)",
        }
    )

    table = pd.DataFrame(rows)
    # Sort primarily by PR-AUC (higher is better), drops the string Runtime
    # column out of numeric sorting via na_position handling.
    return table.sort_values("PR-AUC Mean", ascending=False).reset_index(drop=True)


def format_table(table):
    """Human-readable copy of the comparison table (4-decimal metrics)."""
    display = table.copy()
    for col in [
        "ROC-AUC Mean",
        "ROC-AUC Std",
        "PR-AUC Mean",
        "PR-AUC Std",
        "Precision Mean",
        "Recall Mean",
        "F1 Mean",
    ]:
        display[col] = display[col].map(lambda v: f"{v:.4f}")
    return display


# ---------------------------------------------------------------------------
# Selection insights
# ---------------------------------------------------------------------------
def print_best_models(results):
    """Best PR-AUC / ROC-AUC / Recall / F1 + most stable model (candidates)."""
    separator = "=" * 60
    print("\n" + separator)
    print("MODEL SELECTION INSIGHTS (candidates only)")
    print(separator)
    for metric in SELECTION_METRICS:
        label = SCALAR_METRIC_LABELS[metric]
        best_name = max(
            (n for n in CANDIDATE_NAMES),
            key=lambda n: results[n][metric].mean(),
        )
        value = results[best_name][metric].mean()
        print(f"  Best {label:10s}: {MODEL_REGISTRY[best_name]['label']} "
              f"(mean {value:.4f})")

    # Most stable = smallest average fold variability across PR-AUC and ROC-AUC.
    def stability_score(name):
        fold_df = results[name]
        return (fold_df["roc_auc"].std() + fold_df["pr_auc"].std()) / 2

    stable_name = min(CANDIDATE_NAMES, key=stability_score)
    print(f"  Most stable     : {MODEL_REGISTRY[stable_name]['label']} "
          f"(fold std avg {stability_score(stable_name):.4f})")


def print_tree_improvements_over_lr(results):
    """RF / XGB / CatBoost improvement vs the APPLICATION-ONLY LR baseline."""
    separator = "=" * 60
    print("\n" + separator)
    print("IMPROVEMENT OVER APPLICATION-ONLY LOGISTIC REGRESSION BASELINE")
    print(separator)
    baseline = results["logistic_regression"]
    print("  (positive values = tree model beats the LR baseline)")
    for tree in TREE_NAMES:
        print(f"\n  {MODEL_REGISTRY[tree]['label']}:")
        for metric in SELECTION_METRICS:
            delta = results[tree][metric].mean() - baseline[metric].mean()
            label = SCALAR_METRIC_LABELS[metric]
            print(f"    {label:10s}: {delta:+.4f}")


def print_recommendation(results):
    """Recommend at most TWO candidates for Phase 8 (PR-AUC first, ROC-AUC second)."""
    separator = "=" * 60
    print("\n" + separator)
    print("PHASE 8 RECOMMENDATION")
    print(separator)

    ranked = sorted(
        CANDIDATE_NAMES,
        key=lambda n: (results[n]["pr_auc"].mean(), results[n]["roc_auc"].mean()),
        reverse=True,
    )
    top_2 = ranked[:2]
    for i, name in enumerate(top_2, start=1):
        pr = results[name]["pr_auc"].mean()
        roc = results[name]["roc_auc"].mean()
        print(f"  #{i}: {MODEL_REGISTRY[name]['label']} "
              f"(PR-AUC {pr:.4f}, ROC-AUC {roc:.4f})")
    print("\n  No production model is declared in this phase - final selection")
    print("  happens after Phase 8 (decision logic + explainability).")
    print(separator)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main():
    print("Loading raw dataset...")
    X_train, X_test, y_train, y_test = prepare_data()
    print("  Data prepared. Holdout split (unused by this phase):")
    print(f"    X_train / y_train : {len(X_train):,} rows")
    print(f"    X_test  / y_test  : {len(X_test):,} rows (held out)")

    results = run_comparison(X_train, y_train)

    # Per-fold + mean/std reporting for every model.
    print("\n" + "=" * 60)
    print("FOLD-BY-FOLD SCORES")
    print("=" * 60)
    for name in CANDIDATE_NAMES:
        title = f"MODEL ({name}) - PER-FOLD SCORES"
        print_fold_scores(results[name], title)
        fold_df = results[name]
        print(f"\n  Fit time per fold : "
              f"{fold_df['fit_time_s'].mean():.2f}s +/- {fold_df['fit_time_s'].std():.2f}s "
              f"(mean +/- std)")

    # Final comparison table (sorted by PR-AUC).
    print("\n" + "=" * 60)
    print("FINAL COMPARISON TABLE (sorted by PR-AUC)")
    print("=" * 60)
    table = make_comparison_table(results)
    print(format_table(table).to_string(index=False))

    print_best_models(results)
    print_tree_improvements_over_lr(results)
    print_recommendation(results)

    print("\nPhase 7 complete. No production model was saved - results are printed only.")


if __name__ == "__main__":
    main()