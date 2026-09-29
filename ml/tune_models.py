"""
Phase 9 - Light hyperparameter tuning for the two Phase 8 candidates.

Tunes exactly TWO candidates on the APPLICATION-ONLY feature set using the
same 5-fold StratifiedKFold cross-validation as Phases 6-8:

    Model A - XGBoost Baseline       (no class weighting)
    Model B - CatBoost Class Weighted(fold-training-only balanced weights)

Run with:
    python ml/tune_models.py

Split rules
-----------
    - ONE stratified holdout split exists (Phase 5 logic, test_size=0.20,
      random_state=42, stratify=y). Cross-validation uses ONLY X_train.
      X_test / y_test is never seen by any tuning run.
    - The same stratified 5-fold split with seed 42 is used for every
      parameter combination, so all combos are compared on identical folds.

Search strategy (small + interpretable, deliberately NOT Optuna/grid)
----------------------------------------------------------------------
    - Each model gets ONE hard-coded "staged manual grid" of 18 combinations.
    - Stages are interpretable: 1) depth sweep, 2) inverse learning-rate /
      tree-count scaling, 3) subsampling/regularization variation around the
      most promising depth, 4) depth extremes at low LR / high budget.
    - Approximately 15-20 combos per model as agreed in the plan.

Selection rule
--------------
    - Primary    : mean PR-AUC (average precision, threshold-independent)
    - Tie-break  : mean ROC-AUC, then mean F1
    - Accuracy is never used for selection.
    - Reported metrics always use threshold 0.50 (no threshold tuning here).

Methodology guards (deliberately enforced)
------------------------------------------
    - XGBoost candidate NEVER receives scale_pos_weight (Phase 8 chose the
      BASELINE strategy for XGBoost).
    - CatBoost class weights are recomputed per fold from that fold's
      TRAINING labels only (catboost_balanced_weights).
    - Preprocessing stays inside the pipeline and is refitted per fold.
    - No SMOTE / ADASYN / SHAP / threshold tuning.
    - No production model is saved - results are printed only.
"""

import sys
from pathlib import Path

# Make the project root importable so `ml.*` modules work no matter from
# which folder this script is executed.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from ml.compare_models import prepare_data
from ml.cross_validate_baseline import N_SPLITS, run_cv_scores
from ml.evaluate import SCALAR_METRIC_LABELS
from ml.imbalance_experiment import (
    build_catboost,
    build_xgboost,
    catboost_balanced_weights,
)
from ml.train_baseline import build_model

# ---------------------------------------------------------------------------
# Global configuration
# ---------------------------------------------------------------------------
CV_CONFIG = "application_only"

# Tunable parameter keys per model (exact names passed to the estimators).
XGB_TUNED_PARAMS = ["n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree"]
CAT_TUNED_PARAMS = ["iterations", "depth", "learning_rate", "l2_leaf_reg"]

# The ONLY values the search is allowed to use (kept to the approved ranges).
XGB_SUPPORTED_VALUES = {
    "n_estimators": {100, 200, 300},
    "max_depth": {3, 4, 5},
    "learning_rate": {0.03, 0.05, 0.1},
    "subsample": {0.8, 1.0},
    "colsample_bytree": {0.8, 1.0},
}
CAT_SUPPORTED_VALUES = {
    "iterations": {100, 200, 300},
    "depth": {4, 6, 8},
    "learning_rate": {0.03, 0.05, 0.1},
    "l2_leaf_reg": {3, 5, 7},
}

# Phase 8 baseline EFFECTIVE values of the tunable parameters. These are the
# library defaults (XGBoost: max_depth=6, lr=0.3, sub/col=1.0; CatBoost:
# depth=6, lr=auto, l2=3) used by build_xgboost()/build_catboost() with only
# n_estimators=100 / iterations=100 forced. Hard-coded because
# sklearn's get_params() does not materialize unset **kwargs as defaults.
XGB_BASELINE_PARAMS = {
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.3,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
}
CAT_BASELINE_PARAMS = {
    "iterations": 100,
    "depth": 6,
    "learning_rate": None,  # None -> CatBoost auto-learning-rate
    "l2_leaf_reg": 3,
}

# ---------------------------------------------------------------------------
# Staged manual search grids (18 combos each, hard-coded for transparency)
# ---------------------------------------------------------------------------
# Stage A1: depth sweep     A2: LR/tree scaling  A3: subsample/colsample
# A4: depth extremes at low LR/high trees.
XGB_PARAM_SPACE = [
    # A1 - depth sweep (lr=0.1, n=100, sub=1.0, col=1.0)
    {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.10, "subsample": 1.0, "colsample_bytree": 1.0},
    {"n_estimators": 100, "max_depth": 4, "learning_rate": 0.10, "subsample": 1.0, "colsample_bytree": 1.0},
    {"n_estimators": 100, "max_depth": 5, "learning_rate": 0.10, "subsample": 1.0, "colsample_bytree": 1.0},
    # A2 - inverse LR/tree scaling at depth 4
    {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05, "subsample": 1.0, "colsample_bytree": 1.0},
    {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.03, "subsample": 1.0, "colsample_bytree": 1.0},
    {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.10, "subsample": 1.0, "colsample_bytree": 1.0},
    {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.05, "subsample": 1.0, "colsample_bytree": 1.0},
    {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.03, "subsample": 1.0, "colsample_bytree": 1.0},
    # A3 - subsample / colsample_bytree around the two best A2 candidates
    {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 1.0},
    {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05, "subsample": 1.0, "colsample_bytree": 0.8},
    {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8},
    {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.10, "subsample": 0.8, "colsample_bytree": 1.0},
    {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.10, "subsample": 1.0, "colsample_bytree": 0.8},
    {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.10, "subsample": 0.8, "colsample_bytree": 0.8},
    # A4 - depth extremes at low LR / high tree budgets
    {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.05, "subsample": 1.0, "colsample_bytree": 1.0},
    {"n_estimators": 200, "max_depth": 5, "learning_rate": 0.05, "subsample": 1.0, "colsample_bytree": 1.0},
    {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.03, "subsample": 1.0, "colsample_bytree": 1.0},
    {"n_estimators": 300, "max_depth": 5, "learning_rate": 0.03, "subsample": 1.0, "colsample_bytree": 1.0},
]

# Stage B1: depth sweep  B2: LR/iter scaling  B3: l2_leaf_reg variation
# B4: depth extremes at low LR/high iters + regularization.
CAT_PARAM_SPACE = [
    # B1 - depth sweep (lr=0.1, it=100, l2=3)
    {"iterations": 100, "depth": 4, "learning_rate": 0.10, "l2_leaf_reg": 3},
    {"iterations": 100, "depth": 6, "learning_rate": 0.10, "l2_leaf_reg": 3},
    {"iterations": 100, "depth": 8, "learning_rate": 0.10, "l2_leaf_reg": 3},
    # B2 - inverse LR / iteration scaling at depth 6
    {"iterations": 200, "depth": 6, "learning_rate": 0.05, "l2_leaf_reg": 3},
    {"iterations": 300, "depth": 6, "learning_rate": 0.03, "l2_leaf_reg": 3},
    {"iterations": 200, "depth": 6, "learning_rate": 0.10, "l2_leaf_reg": 3},
    {"iterations": 300, "depth": 6, "learning_rate": 0.05, "l2_leaf_reg": 3},
    {"iterations": 200, "depth": 6, "learning_rate": 0.03, "l2_leaf_reg": 3},
    # B3 - l2_leaf_reg 5/7 around the two best B2 candidates
    {"iterations": 200, "depth": 6, "learning_rate": 0.05, "l2_leaf_reg": 5},
    {"iterations": 200, "depth": 6, "learning_rate": 0.05, "l2_leaf_reg": 7},
    {"iterations": 200, "depth": 6, "learning_rate": 0.10, "l2_leaf_reg": 5},
    {"iterations": 200, "depth": 6, "learning_rate": 0.10, "l2_leaf_reg": 7},
    # B4 - depth extremes + regularization at low LR / high budgets
    {"iterations": 200, "depth": 4, "learning_rate": 0.05, "l2_leaf_reg": 3},
    {"iterations": 200, "depth": 8, "learning_rate": 0.05, "l2_leaf_reg": 3},
    {"iterations": 300, "depth": 4, "learning_rate": 0.03, "l2_leaf_reg": 3},
    {"iterations": 300, "depth": 8, "learning_rate": 0.03, "l2_leaf_reg": 3},
    {"iterations": 300, "depth": 4, "learning_rate": 0.03, "l2_leaf_reg": 5},
    {"iterations": 200, "depth": 8, "learning_rate": 0.05, "l2_leaf_reg": 7},
]

MODELS = ("xgboost", "catboost")
MODEL_LABELS = {"xgboost": "XGBoost", "catboost": "CatBoost"}
TUNED_PARAMS = {"xgboost": XGB_TUNED_PARAMS, "catboost": CAT_TUNED_PARAMS}
PARAM_SPACES = {"xgboost": XGB_PARAM_SPACE, "catboost": CAT_PARAM_SPACE}
SUPPORTED_VALUES = {"xgboost": XGB_SUPPORTED_VALUES, "catboost": CAT_SUPPORTED_VALUES}

# Metrics used for selection (accuracy is intentionally excluded).
SELECTION_METRICS = ("pr_auc", "roc_auc", "f1")
# Mean/std columns required in the tuning report.
REPORT_FIELDS = (
    "PR-AUC", "PR-AUC Std", "ROC-AUC", "ROC-AUC Std", "Precision", "Recall", "F1",
)


# ---------------------------------------------------------------------------
# Human-readable labels
# ---------------------------------------------------------------------------
def combo_label(model, params):
    """Short unique label for one parameter combination."""
    if model == "xgboost":
        return (f"d{params['max_depth']} lr{params['learning_rate']:.2f} "
                f"n{params['n_estimators']} sub{params['subsample']:g} "
                f"col{params['colsample_bytree']:g}")
    return (f"d{params['depth']} lr{params['learning_rate']:.2f} "
            f"it{params['iterations']} l2{params['l2_leaf_reg']}")


def baseline_label(model):
    if model == "xgboost":
        return "Baseline (d6 lr0.30 n100 sub1 col1)"
    return "Baseline - Class Weighted (d6 lr-auto it100 l2=3)"


# ---------------------------------------------------------------------------
# Estimator / pipeline builders
# ---------------------------------------------------------------------------
def baseline_estimator(model, y_fold_train=None):
    """
    The EXACT Phase 8 CANDIDATE estimator used as the Phase 9 reference.

    Phase 8 selected DIFFERENT strategies per model:
      - XGBoost  -> BASELINE strategy (no class weighting)
      - CatBoost -> CLASS WEIGHTED strategy, so its reference must rebuild
                    balanced weights from the fold's TRAINING labels only.
    """
    if model == "xgboost":
        return build_xgboost()
    if y_fold_train is None:
        raise ValueError("CatBoost baseline needs the fold's training labels.")
    return build_catboost(class_weights=catboost_balanced_weights(y_fold_train))


def baseline_pipeline(model, y_fold_train=None):
    """Unfitted application-only pipeline for the Phase 8 candidate baseline."""
    return build_model(CV_CONFIG, estimator=baseline_estimator(model, y_fold_train))[0]


def baseline_params(model):
    """Effective baseline values of the tunable parameters (for reporting)."""
    if model == "xgboost":
        return dict(XGB_BASELINE_PARAMS)
    return dict(CAT_BASELINE_PARAMS)


def build_tuned_pipeline(model, params, y_fold_train=None):
    """
    Build (unfitted) the application-only pipeline for one tuned combo.

    model        : "xgboost" (baseline - NO class weighting) or
                   "catboost" (class weighted from fold-training labels).
    params       : the parameter dict for this combo.
    y_fold_train : the CURRENT fold's TRAINING labels, used ONLY to compute
                   CatBoost balanced weights. Never full-train or validation.
    """
    if model == "xgboost":
        if "scale_pos_weight" in params:
            raise ValueError("XGBoost candidate must never receive scale_pos_weight.")
        estimator = build_xgboost(**params)
    else:  # catboost
        if "class_weights" in params:
            raise ValueError("class_weights must be derived per fold, not in the space.")
        if y_fold_train is None:
            raise ValueError("CatBoost tuned pipelines need the fold's training labels.")
        estimator = build_catboost(
            class_weights=catboost_balanced_weights(y_fold_train), **params
        )
    return build_model(CV_CONFIG, estimator=estimator)[0]


# ---------------------------------------------------------------------------
# Tuning runner
# ---------------------------------------------------------------------------
def run_tuning(X_train, y_train, model):
    """
    Run 5-fold CV for every combo in the model's grid, plus the un-tuned
    Phase 8 baseline (same loop/folds) for the delta comparison.

    Returns (results, baseline_df):
        results         : {combo_label: (params_dict, per-fold DataFrame)}
        baseline_df     : per-fold DataFrame for the Phase 8 baseline
    """
    results = {}
    for params in PARAM_SPACES[model]:
        label = combo_label(model, params)
        print(f"Running 5-fold StratifiedKFold CV for: "
              f"{MODEL_LABELS[model]} - {label}")
        results[label] = (
            params,
            run_cv_scores(
                X_train,
                y_train,
                lambda X_ft, y_ft, m=model, p=params: build_tuned_pipeline(m, p, y_ft),
                n_splits=N_SPLITS,
                time_fits=True,
            ),
        )

    base_label = baseline_label(model)
    print(f"Running 5-fold StratifiedKFold CV for: "
          f"{MODEL_LABELS[model]} - {base_label}")
    baseline_df = run_cv_scores(
        X_train,
        y_train,
        lambda X_ft, y_ft, m=model: baseline_pipeline(m, y_ft),
        n_splits=N_SPLITS,
        time_fits=True,
    )
    return results, baseline_df


# ---------------------------------------------------------------------------
# Selection helpers (pure - unit-testable)
# ---------------------------------------------------------------------------
def summarize_fold_df(fold_df):
    """Mean (+ std where required) summary of one per-fold DataFrame."""
    return {
        "PR-AUC": fold_df["pr_auc"].mean(),
        "PR-AUC Std": fold_df["pr_auc"].std(),
        "ROC-AUC": fold_df["roc_auc"].mean(),
        "ROC-AUC Std": fold_df["roc_auc"].std(),
        "Precision": fold_df["precision"].mean(),
        "Recall": fold_df["recall"].mean(),
        "F1": fold_df["f1"].mean(),
        "Fit/s": fold_df["fit_time_s"].mean(),
    }


def select_best_label(results):
    """
    Best combo label by (mean PR-AUC, mean ROC-AUC, mean F1).
    results : {combo_label: (params, per-fold DataFrame)}
    """
    def score(item):
        _, (_, fold_df) = item
        return tuple(fold_df[metric].mean() for metric in SELECTION_METRICS)

    best_label, _ = max(results.items(), key=score)
    return best_label


def make_deltas(best_fold_df, baseline_df):
    """Best-tuned minus Phase 8 baseline deltas for the report fields."""
    metric_map = {
        "Precision": "precision", "Recall": "recall", "F1": "f1",
        "ROC-AUC": "roc_auc", "PR-AUC": "pr_auc",
    }
    return {
        f"d{field}": best_fold_df[metric_map[field]].mean()
        - baseline_df[metric_map[field]].mean()
        for field in ("PR-AUC", "ROC-AUC", "Recall", "Precision", "F1")
    }


def choose_leader(first, second):
    """
    Head-to-head ranking of two tuned candidates.

    first/second : (name, summary_dict). Primary = PR-AUC mean, then
                   ROC-AUC mean, then F1 mean. Returns
                   (leader, backup) each as (name, summary_dict).
    """
    def score(candidate):
        return (candidate[1]["PR-AUC"], candidate[1]["ROC-AUC"], candidate[1]["F1"])

    leader, backup = sorted([first, second], key=score, reverse=True)
    return leader, backup


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------
def make_tuning_table(results):
    """Sorted (desc) table of every combo with all required fields."""
    rows = [{"Config": label, **summarize_fold_df(fold_df)}
            for label, (_, fold_df) in results.items()]
    table = pd.DataFrame(rows)
    return table.sort_values(
        ["PR-AUC", "ROC-AUC", "F1"], ascending=False
    ).reset_index(drop=True)


def format_report_table(table):
    """4-decimal formatting of the report columns (keeps Config readable)."""
    display = table.copy()
    for col in REPORT_FIELDS + ("Fit/s",):
        display[col] = display[col].map(lambda v: f"{v:.4f}")
    return display


def print_model_section(model, results, baseline_df):
    """Print one model's full report: top-5, best, baseline, deltas."""
    separator = "=" * 88
    print("\n" + separator)
    print(f"{MODEL_LABELS[model]} - TUNING REPORT (APPLICATION-ONLY)")
    print(separator)

    table = make_tuning_table(results)
    print("\nALL CONFIGURATIONS (sorted by mean PR-AUC):")
    print(format_report_table(table).to_string(index=False))

    print("\n" + separator)
    print(f"{MODEL_LABELS[model]} - TOP 5 CONFIGURATIONS")
    print(separator)
    print(format_report_table(table.head(5)).to_string(index=False))

    # Best tuned configuration.
    best_label = select_best_label(results)
    best_params, best_fold_df = results[best_label]
    best_summary = summarize_fold_df(best_fold_df)

    print("\n" + separator)
    print(f"BEST {MODEL_LABELS[model].upper()} PARAMETERS")
    print(separator)
    print(f"  {best_label}")
    for key, value in best_params.items():
        print(f"  {key:16s}: {value}")

    print("\n" + separator)
    print(f"BEST {MODEL_LABELS[model].upper()} METRICS (5-fold CV mean, "
          f"threshold 0.50)")
    print(separator)
    for field in ("PR-AUC", "PR-AUC Std", "ROC-AUC", "ROC-AUC Std",
                  "Precision", "Recall", "F1", "Fit/s"):
        print(f"  {field:12s}: {best_summary[field]:.4f}")

    # Baseline reference + deltas.
    base_summary = summarize_fold_df(baseline_df)
    print("\n" + separator)
    print(f"PHASE 8 BASELINE REFERENCE ({MODEL_LABELS[model]})")
    print(separator)
    print(f"  Parameters: {baseline_label(model)}")
    for key, value in baseline_params(model).items():
        shown = "auto (None -> CatBoost default)" if value is None else value
        print(f"  {key:16s}: {shown}")
    print("\n  Baseline metrics (5-fold CV mean, threshold 0.50):")
    for field in ("PR-AUC", "ROC-AUC", "Precision", "Recall", "F1"):
        print(f"  {field:12s}: {base_summary[field]:.4f}")

    print("\n  Improvement of best-tuned over baseline:")
    deltas = make_deltas(best_fold_df, baseline_df)
    for key, value in deltas.items():
        print(f"  {key:14s}: {value:+.4f}")

    return best_summary


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
    print(f"  CV: {N_SPLITS}-fold StratifiedKFold, random_state=42, "
          "identical folds for every combination.")

    summaries = {}
    for model in MODELS:
        results, baseline_df = run_tuning(X_train, y_train, model)
        own_best = print_model_section(model, results, baseline_df)

        # Final head-to-head needs the OTHER model's summary too, so collect
        # the near-identical info keyed by model.
        best_label = select_best_label(results)
        best_params, best_fold_df = results[best_label]
        summaries[model] = {
            "label": best_label,
            "params": best_params,
            "summary": own_best,
        }

    # Head-to-head comparison.
    separator = "=" * 88
    print("\n" + separator)
    print("HEAD-TO-HEAD: TUNED XGBOOST vs TUNED WEIGHTED CATBOOST")
    print(separator)
    print("\n  Tuned XGBoost  : "
          f"PR-AUC {summaries['xgboost']['summary']['PR-AUC']:.4f} | "
          f"ROC-AUC {summaries['xgboost']['summary']['ROC-AUC']:.4f} | "
          f"Recall {summaries['xgboost']['summary']['Recall']:.4f} | "
          f"F1 {summaries['xgboost']['summary']['F1']:.4f}")
    print(f"    Parameters: {summaries['xgboost']['label']}")
    print("\n  Tuned CatBoost : "
          f"PR-AUC {summaries['catboost']['summary']['PR-AUC']:.4f} | "
          f"ROC-AUC {summaries['catboost']['summary']['ROC-AUC']:.4f} | "
          f"Recall {summaries['catboost']['summary']['Recall']:.4f} | "
          f"F1 {summaries['catboost']['summary']['F1']:.4f}")
    print(f"    Parameters: {summaries['catboost']['label']}")

    leader, backup = choose_leader(
        (f"Tuned {MODEL_LABELS['xgboost']} (baseline)", summaries["xgboost"]["summary"]),
        (f"Tuned {MODEL_LABELS['catboost']} (class weighted)", summaries["catboost"]["summary"]),
    )
    print("\n" + separator)
    print("FINAL PHASE 9 CANDIDATES")
    print(separator)
    print(f"\n  LEADING CANDIDATE : {leader[0]}")
    print(f"    PR-AUC {leader[1]['PR-AUC']:.4f} | ROC-AUC {leader[1]['ROC-AUC']:.4f} "
          f"| F1 {leader[1]['F1']:.4f}")
    print(f"  BACKUP CANDIDATE  : {backup[0]}")
    print(f"    PR-AUC {backup[1]['PR-AUC']:.4f} | ROC-AUC {backup[1]['ROC-AUC']:.4f} "
          f"| F1 {backup[1]['F1']:.4f}")
    print("\n  No production model is declared in this phase - final selection,")
    print("  decision logic (score/band/recommendation) and threshold selection")
    print("  happen in later phases. No model artifacts were saved.")
    print(separator)


if __name__ == "__main__":
    main()