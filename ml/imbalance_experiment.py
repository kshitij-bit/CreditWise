"""
Phase 8 - Class imbalance strategy comparison for XGBoost and CatBoost.

Compares THREE class-imbalance approaches for the two strongest
APPLICATION-ONLY models from Phase 7 (XGBoost, CatBoost):

    A. Baseline       - no imbalance treatment
    B. Class Weighted - XGBoost: scale_pos_weight = n_neg / n_pos
                        CatBoost: balanced class_weights = [n/(2*n0), n/(2*n1)]
                        computed from the TRAINING portion of each CV fold only
    C. SMOTE          - resamples ONLY the (transformed) TRAINING portion of
                        every fold, inside an imblearn Pipeline

Run with:
    python ml/imbalance_experiment.py

Split rules
-----------
    - ONE stratified holdout split exists (test_size=0.20, random_state=42,
      stratify=y), identical logic to Phases 5-7.
    - Cross-validation uses ONLY X_train / y_train. The holdout
      X_test / y_test is never seen by any experiment.

Pipeline (per fold)
-------------------
    Fold training data
        -> fit preprocessing              (inside the Pipeline)
        -> transform fold training data
        -> [SMOTE training data only]     (SMOTE strategy only)
        -> model.fit

    Fold validation data
        -> transform with ALREADY-FITTED preprocessing
        -> model.predict / predict_proba  (never resampled)

The validation set is NEVER resampled: imblearn samplers only act during
fit() (on the training portion) and are skipped during predict/transform.

Metric priority (default is the minority/risk class)
----------------------------------------------------
    Trade-off view: Recall vs Precision vs PR-AUC vs F1. PR-AUC is primary
    because it is threshold-independent. Threshold stays at 0.50.

Design rules (deliberately enforced)
------------------------------------
    - Same fixed baseline complexity as Phase 7 (no hyperparameter tuning)
    - No GridSearchCV / RandomizedSearchCV / Optuna
    - No ADASYN / undersampling / class_weight on validation
    - No threshold tuning, no SHAP
    - The frozen test set is never evaluated
    - No production models are saved - results are printed only
"""

import sys
from pathlib import Path

# Make the project root importable so `ml.*` modules work no matter from
# which folder this script is executed (e.g. python ml/imbalance_experiment.py).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier

from ml.compare_models import prepare_data
from ml.cross_validate_baseline import (
    N_SPLITS,
    REPORT_METRICS,
    run_cv_scores,
)
from ml.evaluate import SCALAR_METRIC_LABELS
from ml.preprocessing import build_logistic_preprocessor
from ml.train_baseline import RANDOM_STATE, build_model

# ---------------------------------------------------------------------------
# Global configuration
# ---------------------------------------------------------------------------
CV_CONFIG = "application_only"

# The six planned experiments: every model x every strategy.
MODELS = ("xgboost", "catboost")
STRATEGIES = ("baseline", "class_weighted", "smote")
EXPERIMENTS = [tuple(pair) for pair in [(m, s) for m in MODELS for s in STRATEGIES]]

MODEL_LABELS = {"xgboost": "XGBoost", "catboost": "CatBoost"}
STRATEGY_LABELS = {
    "baseline": "Baseline",
    "class_weighted": "Class Weighted",
    "smote": "SMOTE",
}


# ---------------------------------------------------------------------------
# Class-weight helpers (pure functions - unit-testable)
# ---------------------------------------------------------------------------
def xgb_scale_pos_weight(y):
    """
    XGBoost scale_pos_weight from the TRAINING labels of one fold only.

    scale_pos_weight = n_negative / n_positive. No validation data is (or can
    be) used: the caller only hands this function the fold's training labels.
    """
    y_series = pd.Series(np.asarray(y))
    n_neg = int((y_series == 0).sum())
    n_pos = int((y_series == 1).sum())
    if n_pos == 0:
        raise ValueError("Fold training data has no positive class - cannot weight.")
    return n_neg / n_pos


def catboost_balanced_weights(y):
    """
    CatBoost balanced class weights from the TRAINING labels of one fold only.

    Standard balanced formula: w_i = n / (k * n_i) for k classes. Returns
    [w_class_0, w_class_1] as CatBoost expects class_weights.
    """
    y_series = pd.Series(np.asarray(y))
    total = len(y_series)
    n_0 = int((y_series == 0).sum())
    n_1 = int((y_series == 1).sum())
    if n_0 == 0 or n_1 == 0:
        raise ValueError("Fold training data must contain both classes.")
    return [total / (2.0 * n_0), total / (2.0 * n_1)]


# ---------------------------------------------------------------------------
# Fixed baseline estimators (same complexity as Phase 7, seed 42)
# ---------------------------------------------------------------------------
def build_xgboost(**params):
    """
    Build an XGBClassifier with the project's fixed seed and metrics.

    Defaults are stored in a dict and merged with `params` so callers may
    override ANY constructor argument (including n_estimators/max_depth used
    by the Phase 9 tuning grid) without a duplicate-keyword error.
    """
    defaults = {
        "n_estimators": 100,
        "random_state": RANDOM_STATE,
        "eval_metric": "logloss",
    }
    defaults.update(params)
    return XGBClassifier(**defaults)


def build_catboost(**params):
    """
    Build a CatBoostClassifier with the project's fixed seed and settings.

    Defaults are stored in a dict and merged with `params` so callers may
    override ANY constructor argument (including iterations/depth used by the
    Phase 9 tuning grid) without a duplicate-keyword error.
    """
    defaults = {
        "iterations": 100,
        "random_seed": RANDOM_STATE,
        "verbose": False,
        "allow_writing_files": False,
    }
    defaults.update(params)
    return CatBoostClassifier(**defaults)


def base_estimator(model):
    if model == "xgboost":
        return build_xgboost()
    if model == "catboost":
        return build_catboost()
    raise ValueError(f"Unknown model '{model}'.")


# ---------------------------------------------------------------------------
# Pipeline builder per (model, strategy)
# ---------------------------------------------------------------------------
def build_experiment_pipeline(model, strategy, y_fold_train=None):
    """
    Build (unfitted) the full experiment pipeline for (model, strategy).

    y_fold_train is the CURRENT fold's TRAINING labels and is used ONLY by the
    class_weighted strategy to compute weights, which guarantees weights are
    never derived from validation data. The baseline and SMOTE strategies
    ignore it.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy '{strategy}'.")

    if strategy == "baseline":
        return build_model(CV_CONFIG, estimator=base_estimator(model))[0]

    if strategy == "class_weighted":
        if y_fold_train is None:
            raise ValueError("class_weighted needs the fold's training labels.")
        if model == "xgboost":
            estimator = build_xgboost(scale_pos_weight=xgb_scale_pos_weight(y_fold_train))
        else:  # catboost
            estimator = build_catboost(class_weights=catboost_balanced_weights(y_fold_train))
        return build_model(CV_CONFIG, estimator=estimator)[0]

    if strategy == "smote":
        preprocessor, _ = build_logistic_preprocessor(CV_CONFIG)
        return ImbPipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("smote", SMOTE(random_state=RANDOM_STATE)),
                ("classifier", base_estimator(model)),
            ]
        )

    raise ValueError(f"Unhandled strategy '{strategy}'.")


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------
def run_experiments(X_train, y_train):
    """
    5-fold CV for all six experiments on X_train/y_train only.

    Returns {(model, strategy): per-fold DataFrame} including per-fold fit time.
    Every experiment uses the exact same StratifiedKFold splits (same seed).
    """
    results = {}
    for model, strategy in EXPERIMENTS:
        name = f"{MODEL_LABELS[model]} - {STRATEGY_LABELS[strategy]}"
        print(f"Running 5-fold StratifiedKFold CV for: {name}")
        results[(model, strategy)] = run_cv_scores(
            X_train,
            y_train,
            lambda X_ft, y_ft, m=model, s=strategy: build_experiment_pipeline(m, s, y_ft),
            n_splits=N_SPLITS,
            time_fits=True,
        )
    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_fold_panel(fold_df, title):
    """Compact per-experiment panel: metric x (Fold1-5, Mean, Std)."""
    separator = "=" * 72
    print(separator)
    print(title)
    print(separator)
    panel = fold_df[REPORT_METRICS].T.copy()
    panel.columns = [f"Fold {c}" for c in panel.columns]
    panel["Mean"] = panel.mean(axis=1)
    panel["Std"] = panel.std(axis=1)
    print(panel.round(4).to_string())
    print(f"  Fit time / fold : {fold_df['fit_time_s'].mean():.2f}s "
          f"+/- {fold_df['fit_time_s'].std():.2f}s")


def make_primary_table(results):
    """
    Final 6-row table grouped by model (Baseline -> Class Weighted -> SMOTE):
        Model | Strategy | ROC-AUC Mean | PR-AUC Mean | Precision Mean |
        Recall Mean | F1 Mean | ROC-AUC Std | PR-AUC Std
    """
    rows = []
    for model in MODELS:
        for strategy in STRATEGIES:
            fold_df = results[(model, strategy)]
            rows.append(
                {
                    "Model": MODEL_LABELS[model],
                    "Strategy": STRATEGY_LABELS[strategy],
                    "ROC-AUC": fold_df["roc_auc"].mean(),
                    "PR-AUC": fold_df["pr_auc"].mean(),
                    "Precision": fold_df["precision"].mean(),
                    "Recall": fold_df["recall"].mean(),
                    "F1": fold_df["f1"].mean(),
                    "ROC-AUC Std": fold_df["roc_auc"].std(),
                    "PR-AUC Std": fold_df["pr_auc"].std(),
                }
            )
    return pd.DataFrame(rows)


def make_delta_table(results):
    """
    Change relative to the SAME model's baseline:
        Model | Strategy | Δ Precision | Δ Recall | Δ F1 | Δ PR-AUC
    Baseline rows are 0.0000 by definition.
    """
    rows = []
    for model in MODELS:
        base = results[(model, "baseline")]
        for strategy in STRATEGIES:
            fold_df = results[(model, strategy)]
            if strategy == "baseline":
                deltas = {"Precision": 0.0, "Recall": 0.0, "F1": 0.0, "PR-AUC": 0.0}
            else:
                deltas = {
                    "Precision": fold_df["precision"].mean() - base["precision"].mean(),
                    "Recall": fold_df["recall"].mean() - base["recall"].mean(),
                    "F1": fold_df["f1"].mean() - base["f1"].mean(),
                    "PR-AUC": fold_df["pr_auc"].mean() - base["pr_auc"].mean(),
                }
            rows.append(
                {
                    "Model": MODEL_LABELS[model],
                    "Strategy": STRATEGY_LABELS[strategy],
                    "dPrecision": deltas["Precision"],
                    "dRecall": deltas["Recall"],
                    "dF1": deltas["F1"],
                    "dPR-AUC": deltas["PR-AUC"],
                }
            )
    return pd.DataFrame(rows)


def print_tradeoff_interpretation(results):
    """Describe the Recall/Precision/PR-AUC/F1 trade-off per strategy."""
    separator = "=" * 72
    print("\n" + separator)
    print("TRADE-OFF INTERPRETATION (strategy vs SAME model baseline)")
    print(separator)
    for model in MODELS:
        base = results[(model, "baseline")]
        for strategy in STRATEGIES[1:]:
            fold_df = results[(model, strategy)]
            d_recall = fold_df["recall"].mean() - base["recall"].mean()
            d_precision = fold_df["precision"].mean() - base["precision"].mean()
            d_f1 = fold_df["f1"].mean() - base["f1"].mean()
            d_pr = fold_df["pr_auc"].mean() - base["pr_auc"].mean()
            # Threshold-independent ranking quality unchanged if PR-AUC shift
            # stays within one fold-std of the baseline.
            pr_spread = max(base["pr_auc"].std(), fold_df["pr_auc"].std())
            if abs(d_pr) <= pr_spread:
                ranking_note = "PR-AUC unchanged within fold noise"
            elif d_pr > 0:
                ranking_note = f"PR-AUC improved +{d_pr:.4f}"
            else:
                ranking_note = f"PR-AUC worsened {d_pr:+.4f}"
            print(f"\n  {MODEL_LABELS[model]} - {STRATEGY_LABELS[strategy]}:")
            print(f"    Recall   : {d_recall:+.4f}")
            print(f"    Precision: {d_precision:+.4f}")
            print(f"    F1       : {d_f1:+.4f}")
            print(f"    {ranking_note}")
            if d_recall > 0 and d_precision < 0:
                print("    -> Operating-point shift: more defaults caught at the cost")
                print("       of lower precision.")
            elif d_recall > 0 and d_precision >= 0:
                print("    -> Strictly better recall without sacrificing precision.")
            else:
                print("    -> No clear recall benefit; treat as a robustness check.")

    # Stability across folds per experiment.
    print("\n  Fold stability (mean of ROC-AUC std and PR-AUC std):")
    for (model, strategy), fold_df in results.items():
        stability = (fold_df["roc_auc"].std() + fold_df["pr_auc"].std()) / 2
        print(f"    {MODEL_LABELS[model]:9s} {STRATEGY_LABELS[strategy]:14s}: "
              f"{stability:.4f}")


def print_recommendation(results):
    """Best per-model strategy (diversified) + overall highlights."""
    separator = "=" * 72
    print("\n" + separator)
    print("RECOMMENDATION (at most two - one per model)")
    print(separator)

    for model in MODELS:
        best_strategy = max(
            STRATEGIES, key=lambda s: results[(model, s)]["pr_auc"].mean()
        )
        fold_df = results[(model, best_strategy)]
        print(f"\n  Best {MODEL_LABELS[model]} strategy: "
              f"{STRATEGY_LABELS[best_strategy]}")
        print(f"    PR-AUC {fold_df['pr_auc'].mean():.4f} | "
              f"ROC-AUC {fold_df['roc_auc'].mean():.4f} | "
              f"Recall {fold_df['recall'].mean():.4f} | "
              f"F1 {fold_df['f1'].mean():.4f}")

    print("\n  Overall highlights across all six experiments:")
    metric_best = {
        "pr_auc": "PR-AUC",
        "roc_auc": "ROC-AUC",
        "recall": "Recall",
        "f1": "F1",
    }
    for metric, label in metric_best.items():
        best = max(EXPERIMENTS, key=lambda e: results[e][metric].mean())
        print(f"    Best {label:7s}: {MODEL_LABELS[best[0]]} - "
              f"{STRATEGY_LABELS[best[1]]} ({results[best][metric].mean():.4f})")

    print("\n  No production model is declared in this phase - the decision")
    print("  logic and threshold selection happen in later phases.")
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
    print(f"  Feature config: {CV_CONFIG} "
          "(loan_grade / loan_int_rate excluded)")

    results = run_experiments(X_train, y_train)

    # Per-experiment fold panels (Fold 1-5, mean, std).
    print("\n" + "=" * 72)
    print("FOLD-BY-FOLD SCORES")
    print("=" * 72)
    for model, strategy in EXPERIMENTS:
        title = (f"{MODEL_LABELS[model]} - {STRATEGY_LABELS[strategy]} "
                 f"PER-FOLD SCORES")
        print_fold_panel(results[(model, strategy)], title)

    # Final 6-row comparison table.
    print("\n" + "=" * 72)
    print("FINAL COMPARISON TABLE (6 experiments, APPLICATION-ONLY)")
    print("=" * 72)
    primary = make_primary_table(results)
    format_cols = ["ROC-AUC", "PR-AUC", "Precision", "Recall", "F1", "ROC-AUC Std", "PR-AUC Std"]
    display = primary.copy()
    for col in format_cols:
        display[col] = display[col].map(lambda v: f"{v:.4f}")
    print(display.to_string(index=False))

    # Deltas relative to each model's baseline.
    print("\n" + "=" * 72)
    print("DELTAS RELATIVE TO EACH MODEL'S BASELINE")
    print("=" * 72)
    deltas = make_delta_table(results).copy()
    for col in ["dPrecision", "dRecall", "dF1", "dPR-AUC"]:
        deltas[col] = deltas[col].map(lambda v: f"{v:+.4f}")
    print(deltas.to_string(index=False))

    print_tradeoff_interpretation(results)
    print_recommendation(results)

    print("\nPhase 8 complete. No production model was saved - results are printed only.")


if __name__ == "__main__":
    main()