"""
Phase 5 - Model evaluation helpers for CreditWise AI.

Reusable, framework-free metric computation for trained classifiers.
Every metric is returned in one clean dictionary so callers (and tests)
can use the results programmatically.

Metrics returned
----------------
    accuracy, precision, recall, f1, roc_auc, pr_auc, confusion_matrix

ROC-AUC and PR-AUC need predicted *probabilities*, all other metrics use
hard 0/1 predictions (default threshold 0.50).
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Labels used when rendering the confusion matrix.
CONFUSION_LABELS = ["0 - No default", "1 - Default"]

# Canonical order of the scalar metrics.
SCALAR_METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
SCALAR_METRIC_LABELS = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1",
    "roc_auc": "ROC-AUC",
    "pr_auc": "PR-AUC",
}


def evaluate_classifier(y_true, y_pred, y_proba=None):
    """
    Compute all evaluation metrics for a binary classifier.

    Parameters
    ----------
    y_true : array-like
        Ground-truth labels (0 / 1).
    y_pred : array-like
        Hard model predictions at the default threshold (0 / 1).
    y_proba : array-like, optional
        Predicted probability of the positive (default) class.
        Required for ROC-AUC / PR-AUC.

    Returns
    -------
    dict with keys:
        accuracy, precision, recall, f1, roc_auc, pr_auc, confusion_matrix

    zero_division=0 is used so a precision/f1 of 0/0 does not crash.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

    if y_proba is not None:
        # Threshold-independent metrics computed from probabilities.
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
        metrics["pr_auc"] = average_precision_score(y_true, y_proba)
    else:
        # Without probabilities these cannot be computed.
        metrics["roc_auc"] = float("nan")
        metrics["pr_auc"] = float("nan")

    # Confusion matrix as a labelled DataFrame for readable output.
    cm_array = confusion_matrix(y_true, y_pred)
    metrics["confusion_matrix"] = pd.DataFrame(
        cm_array,
        index=pd.Index(CONFUSION_LABELS, name="Actual"),
        columns=pd.Index(CONFUSION_LABELS, name="Predicted"),
    )

    return metrics


def print_comparison_table(full_metrics, application_only_metrics):
    """
    Print a side-by-side table of FULL vs APPLICATION-ONLY metrics.

    The two metric dicts come from evaluate_classifier().
    """
    rows = []
    for metric in SCALAR_METRICS:
        rows.append(
            {
                "Metric": SCALAR_METRIC_LABELS[metric],
                "FULL": f"{full_metrics[metric]:.4f}",
                "APPLICATION-ONLY": f"{application_only_metrics[metric]:.4f}",
            }
        )
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))


def print_confusion_matrix(confusion_df, title):
    """Print one labeled confusion matrix under a heading."""
    print(f"\n{title}")
    print(confusion_df.to_string())


if __name__ == "__main__":
    # Quick self-check:
    #   python ml/evaluate.py
    y_true = np.array([0, 1, 1, 0, 1, 0, 0, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 0, 1, 0, 0, 0, 1, 1])
    y_proba = np.array([0.1, 0.9, 0.6, 0.2, 0.8, 0.3, 0.4, 0.1, 0.7, 0.95])
    result = evaluate_classifier(y_true, y_pred, y_proba)
    print("Self-check metrics:")
    for key, value in result.items():
        if key != "confusion_matrix":
            print(f"  {key:16s}: {value:.4f}")
    print(result["confusion_matrix"])