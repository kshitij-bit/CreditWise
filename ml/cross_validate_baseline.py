"""
Phase 6 - Baseline validation via 5-fold StratifiedKFold cross-validation.

Validates the TWO Phase 5 Logistic Regression configurations on the SAME
cleaned+engineered dataset to check whether the performance difference
(FULL vs APPLICATION-ONLY) is stable across folds:

    Model A - FULL            : includes loan_grade and loan_int_rate
    Model B - APPLICATION-ONLY: excludes loan_grade and loan_int_rate

Run with:
    python ml/cross_validate_baseline.py

Split rules
-----------
    - ONE stratified holdout split exists (test_size=0.20, random_state=42,
      stratify=y), identical logic to Phase 5.
    - Cross-validation uses ONLY X_train / y_train. The holdout
      X_test / y_test is never seen by the CV loop or the models.

Pipeline (per fold)
-------------------
    Feature engineering (deterministic, done once up front)
        -> Preprocessor (imputation + scaling + encoding)
        -> LogisticRegression(max_iter=1000, random_state=42)

The preprocessor is refitted FROM SCRATCH on each fold's training portion
only, so imputation/scaling/encoding can never leak validation information.

Design rules (deliberately enforced)
------------------------------------
    - Default probability threshold 0.50 for Precision/Recall/F1
    - No SMOTE / ADASYN / class_weight / threshold tuning
    - No Random Forest / XGBoost / CatBoost / hyperparameter tuning / SHAP
    - No production models are saved - results are printed only
"""

import sys
import time
from pathlib import Path

# Make the project root importable so `ml.*` modules work no matter from
# which folder this script is executed (e.g. python ml/cross_validate_baseline.py).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from ml.data_cleaning import clean_credit_data
from ml.evaluate import SCALAR_METRICS, SCALAR_METRIC_LABELS, evaluate_classifier
from ml.feature_engineering import engineer_features
from ml.preprocessing import TARGET
from ml.train_baseline import (
    CONFIGS,
    RANDOM_STATE,
    TEST_SIZE,
    build_model,
    get_split,
    load_data,
)

# ---------------------------------------------------------------------------
# Global configuration
# ---------------------------------------------------------------------------
N_SPLITS = 5
CV_SEED = 42

# Metrics reported per fold (Phase 6 requirement). Accuracy is excluded here
# even though evaluate_classifier also computes it.
REPORT_METRICS = ["roc_auc", "pr_auc", "precision", "recall", "f1"]

# Phase 5 single-split (holdout) results, used only to check reproducibility
# of the CV mean. Results were printed-only in Phase 5, so we store them here
# as constants. CV itself never touches X_test / y_test.
PHASE5_SINGLE_SPLIT = {
    "full": {"roc_auc": 0.872, "pr_auc": 0.70, "recall": 0.528, "precision": 0.73, "f1": 0.612},
    "application_only": {"roc_auc": 0.819, "pr_auc": 0.64, "recall": 0.430, "precision": 0.73, "f1": 0.540},
}


# ---------------------------------------------------------------------------
# Cross-validation core (small, testable functions)
# ---------------------------------------------------------------------------
def make_folds(y_train, n_splits=N_SPLITS, random_state=CV_SEED):
    """
    Build a StratifiedKFold splitter with the Phase 6 fixed configuration.

    Returns (splitter, n_folds) so tests can assert the exact parameters.
    """
    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    n_folds = splitter.get_n_splits()
    return splitter, n_folds


def run_cv_scores(X_train, y_train, build_pipeline_fn, n_splits=N_SPLITS, time_fits=False):
    """
    Run per-fold validation for ONE model pipeline builder.

    build_pipeline_fn : callable(x_fold_train, y_fold_train) that returns a
        FRESH (unfitted) Pipeline for the fold. It receives the fold's TRAINING
        rows only, which lets callers derive fold-specific settings (e.g.
        class weights for Phase 8) without ever seeing validation data.
        Phases 6/7 pass a lambda here so every phase shares exactly the same
        fold logic, splitter, and leakage rules.

    Returns a DataFrame with one row per fold containing the fold number and
    every scalar metric from evaluate_classifier() (plus fit_time_s when
    time_fits=True).

    The model pipeline (preprocessor + classifier) is rebuilt from scratch and
    fitted on each fold's training portion only, guaranteeing that
    imputation/scaling/encoding cannot leak validation information.
    """
    splitter, n_folds = make_folds(y_train, n_splits=n_splits)
    if n_folds != n_splits:
        raise ValueError(f"Expected {n_splits} folds, got splitter with {n_folds}.")

    fold_rows = []
    for fold, (train_idx, val_idx) in enumerate(splitter.split(X_train, y_train), start=1):
        X_fold_train = X_train.iloc[train_idx]
        X_fold_val = X_train.iloc[val_idx]
        y_fold_train = y_train.iloc[train_idx]
        y_fold_val = y_train.iloc[val_idx]

        model = build_pipeline_fn(X_fold_train, y_fold_train)
        started = time.perf_counter()
        # Fitting the whole Pipeline fits preprocessor AND classifier on the
        # fold's training rows only - the validation rows stay unseen.
        model.fit(X_fold_train, y_fold_train)
        fit_time_s = time.perf_counter() - started

        y_pred = model.predict(X_fold_val)               # default threshold 0.50
        y_proba = model.predict_proba(X_fold_val)[:, 1]  # P(default)

        metrics = evaluate_classifier(y_fold_val, y_pred, y_proba)
        row = {"fold": fold}
        row.update({metric: metrics[metric] for metric in SCALAR_METRICS})
        if time_fits:
            row["fit_time_s"] = fit_time_s
        fold_rows.append(row)

    return pd.DataFrame(fold_rows).set_index("fold")


def run_cross_validation(X_train, y_train, config, n_splits=N_SPLITS):
    """
    Phase 6 wrapper: validate ONE Logistic Regression configuration.

    Delegates to the generic run_cv_scores() so Phase 6 and Phase 7 share the
    exact same fold logic. The pipeline (preprocessor + LR) is rebuilt from
    scratch for every fold.
    """
    return run_cv_scores(
        X_train,
        y_train,
        lambda X_fold_train, y_fold_train: build_model(config)[0],
        n_splits=n_splits,
    )


def summarize_fold_scores(fold_df, metrics=REPORT_METRICS):
    """
    Summarize per-fold scores into mean / std / min / max per metric.

    std uses sample standard deviation (ddof=1) - the default pandas choice.
    """
    summary_rows = []
    for metric in metrics:
        values = fold_df[metric]
        summary_rows.append(
            {
                "Metric": SCALAR_METRIC_LABELS[metric],
                "Mean": values.mean(),
                "Std": values.std(),
                "Min": values.min(),
                "Max": values.max(),
            }
        )
    return pd.DataFrame(summary_rows)


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------
def print_fold_scores(fold_df, title, metrics=REPORT_METRICS):
    """Print one metric per block with all 5 fold values, mean and std."""
    separator = "=" * 60
    print(separator)
    print(title)
    print(separator)
    for metric in metrics:
        values = fold_df[metric]
        label = SCALAR_METRIC_LABELS[metric]
        print(f"\n{label} ({metric})")
        print("-" * 60)
        for fold in fold_df.index:
            print(f"  Fold {fold} : {values[fold]:.4f}")
        print(f"  Mean       : {values.mean():.4f}")
        print(f"  Std        : {values.std():.4f}")


def print_comparison_table(full_scores, app_only_scores, metrics=REPORT_METRICS):
    """
    Print the clean summary table requested in Phase 6:

        Metric | FULL Mean | FULL Std | APPLICATION-ONLY Mean | APPLICATION-ONLY Std
    """
    rows = []
    for metric in metrics:
        rows.append(
            {
                "Metric": SCALAR_METRIC_LABELS[metric],
                "FULL Mean": f"{full_scores[metric].mean():.4f}",
                "FULL Std": f"{full_scores[metric].std():.4f}",
                "APPLICATION-ONLY Mean": f"{app_only_scores[metric].mean():.4f}",
                "APPLICATION-ONLY Std": f"{app_only_scores[metric].std():.4f}",
            }
        )
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))


def print_gap_report(full_scores, app_only_scores, metrics=("roc_auc", "pr_auc", "recall", "f1")):
    """
    Report the mean performance gaps (FULL - APPLICATION-ONLY) and whether the
    two configurations overlap once fold variability is accounted for.
    """
    print("\n" + "-" * 60)
    print("MEAN PERFORMANCE GAPS (FULL - APPLICATION-ONLY)")
    print("-" * 60)
    for metric in metrics:
        full = full_scores[metric]
        app = app_only_scores[metric]
        gap = full.mean() - app.mean()
        # Combined fold variability: does the mean gap exceed 2x the spread of
        # the more variable configuration? More signal than a hard hypothesis test.
        spread = max(full.std(), app.std())
        separation = "  (gap clear of fold variability)" if abs(gap) > 2 * spread else "  (gap within fold variability)"
        print(f"  {SCALAR_METRIC_LABELS[metric]:12s}: {gap:+.4f}{separation}")


def print_interpretation(full_scores, app_only_scores, phase5_single_split=PHASE5_SINGLE_SPLIT):
    """
    Short, cautious interpretation of the CV comparison:

    1. Does FULL consistently outperform APPLICATION-ONLY?
    2. Are the results stable across folds?
    3. Is the Phase 5 single-split result reproducible?
    """
    separator = "=" * 60
    print("\n" + separator)
    print("INTERPRETATION")
    print(separator)

    roc_gap = full_scores["roc_auc"].mean() - app_only_scores["roc_auc"].mean()
    recall_gap = full_scores["recall"].mean() - app_only_scores["recall"].mean()

    full_wins_roc_every_fold = (full_scores["roc_auc"] > app_only_scores["roc_auc"]).all()
    full_wins_recall_every_fold = (full_scores["recall"] > app_only_scores["recall"]).all()

    # Variability comparison across folds.
    roc_full_std = full_scores["roc_auc"].std()
    roc_app_std = app_only_scores["roc_auc"].std()

    # 1) Consistent outperformance?
    if full_wins_roc_every_fold and full_wins_recall_every_fold:
        consistency = ("YES - FULL beat APPLICATION-ONLY on ROC-AUC and recall in "
                       "ALL 5 folds.")
    elif full_wins_roc_every_fold:
        consistency = ("PARTIALLY - FULL beat APPLICATION-ONLY on ROC-AUC in all "
                       "5 folds, but not on every other metric.")
    else:
        consistency = "NOT ALWAYS - FULL did not win on every fold."
    print(f"\n1) Does FULL consistently outperform APPLICATION-ONLY?\n   {consistency}")

    # 2) Stability across folds.
    print(f"\n2) Are the results stable across folds?")
    print(f"   FULL            ROC-AUC folds std = {roc_full_std:.4f}")
    print(f"   APPLICATION-ONLY ROC-AUC folds std = {roc_app_std:.4f}")
    if roc_full_std < 0.01 and roc_app_std < 0.01:
        print("   Yes - fold-to-fold ROC-AUC spread is very small (< 0.01) for both.")
    elif roc_full_std < 0.02 and roc_app_std < 0.02:
        print("   Yes - fold-to-fold ROC-AUC spread is small (< 0.02) for both.")
    else:
        print("   Moderate - fold-to-fold ROC-AUC spread exceeds 0.02.")

    # 3) Reproducibility vs the Phase 5 single test-split.
    print(f"\n3) Is the Phase 5 single-split result reproducible?")
    if phase5_single_split is not None:
        phase5_roc_gap = (phase5_single_split["full"]["roc_auc"]
                          - phase5_single_split["application_only"]["roc_auc"])
        phase5_recall_gap = (phase5_single_split["full"]["recall"]
                             - phase5_single_split["application_only"]["recall"])
        cv_roc_gap_proximity = abs(roc_gap - phase5_roc_gap)
        cv_recall_gap_proximity = abs(recall_gap - phase5_recall_gap)
        print(f"   CV mean ROC-AUC gap  : {roc_gap:+.4f}   (Phase 5: {phase5_roc_gap:+.4f})")
        print(f"   CV mean Recall gap   : {recall_gap:+.4f}   (Phase 5: {phase5_recall_gap:+.4f})")
        if cv_roc_gap_proximity < 0.01 and cv_recall_gap_proximity < 0.05:
            print("   Yes - the mean gaps over 5 folds closely match the single split.")
        else:
            print("   Mostly - same direction and magnitude, some fold-to-fold shift.")
    else:
        print(f"   CV mean ROC-AUC gap : {roc_gap:+.4f} (no single-split reference provided).")

    print(f"\n-> These two fields are lender-assigned/priced. This does NOT prove")
    print("   data leakage - it means they require deployment consideration:")
    print("   if the goal is to screen applicants BEFORE a grade/rate exists,")
    print("   the APPLICATION-ONLY model is the realistic one.")
    print(separator)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main():
    print("Loading raw dataset...")
    df = load_data()
    print(f"  Loaded {len(df):,} rows.")

    # 1) Deterministic cleaning (duplicates + impossible records).
    cleaned, _ = clean_credit_data(df)

    # 2) Deterministic feature engineering (ratios, log, binary flags).
    engineered, _ = engineer_features(cleaned)

    # 3) Separate features from the target.
    X = engineered.drop(columns=[TARGET])
    y = engineered[TARGET]

    # 4) ONE stratified holdout split, SAME logic/seed as Phase 5.
    X_train, X_test, y_train, y_test = get_split(X, y)
    print(f"\nHoldout split (unused by cross-validation):")
    print(f"  X_train / y_train : {len(X_train):,} rows")
    print(f"  X_test  / y_test  : {len(X_test):,} rows (held out)")

    # 5) Cross-validate BOTH configurations on X_train / y_train only.
    fold_scores = {}
    for config in CONFIGS:
        print(f"\nRunning 5-fold StratifiedKFold CV for config: {config}")
        fold_scores[config] = run_cross_validation(X_train, y_train, config)

    # 6) Per-fold + mean/std reporting for every metric.
    print("\n" + "=" * 60)
    print("FOLD SCORES")
    print("=" * 60)
    for config in CONFIGS:
        title = f"MODEL ({config}) - PER-FOLD SCORES"
        print_fold_scores(fold_scores[config], title)

    # 7) Clean summary table + gap report.
    print("\n" + "=" * 60)
    print("SUMMARY - MEAN +/- STD PER METRIC")
    print("=" * 60)
    print_comparison_table(fold_scores["full"], fold_scores["application_only"])

    print_gap_report(fold_scores["full"], fold_scores["application_only"])

    # 8) Interpretation.
    print_interpretation(fold_scores["full"], fold_scores["application_only"])

    print("\nPhase 6 complete. No production model was saved - results are printed only.")


if __name__ == "__main__":
    main()