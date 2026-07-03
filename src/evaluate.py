"""Model evaluation, threshold selection, and metric artifacts.

Provides the metric calculations, positive-class probability extraction, and the
validation-only threshold search used by ``src.train`` to choose a model without
touching the held-out test set. Also writes the JSON/plot artifacts logged to
MLflow and surfaced by ``/model-info``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def predict_positive_probability(model: Any, X: pd.DataFrame) -> np.ndarray:
    """Return positive-class probabilities for estimators with common APIs."""
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)
        return np.asarray(probabilities)[:, 1]
    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(X))
        return 1.0 / (1.0 + np.exp(-scores))
    predictions = np.asarray(model.predict(X))
    return predictions.astype(float)


def calculate_metrics(
    y_true: pd.Series | np.ndarray,
    y_probability: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float | list[list[int]]]:
    """Calculate classification metrics with emphasis on positive class."""
    y_true_arr = np.asarray(y_true)
    y_pred = (y_probability >= threshold).astype(int)
    cm = confusion_matrix(y_true_arr, y_pred, labels=[0, 1])
    true_negative, false_positive, false_negative, true_positive = cm.ravel()
    specificity = (
        float(true_negative) / float(true_negative + false_positive)
        if (true_negative + false_positive) > 0
        else 0.0
    )
    metrics: dict[str, float | list[list[int]]] = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true_arr, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_arr, y_pred)),
        "precision_positive": float(precision_score(y_true_arr, y_pred, zero_division=0)),
        "recall_positive": float(recall_score(y_true_arr, y_pred, zero_division=0)),
        "specificity": specificity,
        "f1_positive": float(f1_score(y_true_arr, y_pred, zero_division=0)),
        "f2_positive": float(fbeta_score(y_true_arr, y_pred, beta=2, zero_division=0)),
        "confusion_matrix": cm.tolist(),
    }
    if len(set(y_true_arr.tolist())) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true_arr, y_probability))
        metrics["pr_auc"] = float(average_precision_score(y_true_arr, y_probability))
    else:
        metrics["roc_auc"] = float("nan")
        metrics["pr_auc"] = float("nan")
    return metrics


def classification_report_dict(
    y_true: pd.Series | np.ndarray,
    y_probability: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    """Return the sklearn classification report as a dictionary."""
    y_pred = (y_probability >= threshold).astype(int)
    return classification_report(y_true, y_pred, output_dict=True, zero_division=0)


def choose_best_threshold(
    y_true: pd.Series | np.ndarray,
    y_probability: np.ndarray,
    thresholds: np.ndarray | None = None,
    min_recall: float = 0.65,
) -> tuple[float, dict[str, float | list[list[int]]], list[tuple[float, dict[str, float | list[list[int]]]]]]:
    """Select threshold that maximizes F1 while keeping recall >= min_recall.

    This should be called on a *validation* split, never on the held-out test
    set, so that the chosen threshold does not leak test information. Returns
    (best_threshold, best_metrics, all_scored) so callers can log per-threshold
    results to MLflow.
    """
    if thresholds is None:
        thresholds = np.round(np.arange(0.20, 0.71, 0.02), 2)
    scored = []
    for threshold in thresholds:
        metrics = calculate_metrics(y_true, y_probability, float(threshold))
        scored.append((float(threshold), metrics))

    eligible = [
        item for item in scored
        if float(item[1]["recall_positive"]) >= min_recall
    ]
    if eligible:
        eligible.sort(key=lambda item: float(item[1]["f1_positive"]), reverse=True)
        best = eligible[0]
    else:
        scored.sort(key=lambda item: float(item[1]["f2_positive"]), reverse=True)
        best = scored[0]
    return best[0], best[1], scored


def write_json_artifact(payload: dict[str, Any], path: Path) -> None:
    """Write a JSON artifact with consistent formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, allow_nan=True)


def save_confusion_matrix_plot(
    y_true: pd.Series | np.ndarray,
    y_probability: np.ndarray,
    threshold: float,
    output_path: Path,
) -> None:
    """Save a confusion matrix plot when matplotlib is available."""
    try:
        import os
        os.environ.setdefault("MPLCONFIGDIR", str(output_path.parent / ".matplotlib"))
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay
    except Exception:
        return

    y_pred = (y_probability >= threshold).astype(int)
    display = ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
    display.ax_.set_title(f"Confusion Matrix @ {threshold:.2f}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_feature_importance_artifact(model: Any, feature_names: list[str], output_path: Path) -> None:
    """Save model feature importances or coefficients as JSON."""
    values: np.ndarray | None = None
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_)
    elif hasattr(model, "coef_"):
        values = np.asarray(model.coef_).ravel()

    if values is None:
        payload = {"feature_importance": []}
    else:
        rows = [
            {"feature": feature, "importance": float(value)}
            for feature, value in zip(feature_names, values, strict=False)
        ]
        rows.sort(key=lambda row: abs(row["importance"]), reverse=True)
        payload = {"feature_importance": rows[:50]}
    write_json_artifact(payload, output_path)
