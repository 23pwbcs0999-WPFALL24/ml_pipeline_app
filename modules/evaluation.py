"""Evaluation pipeline for classification and regression predictions."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def evaluate_model(y_true, y_pred, task_type: str):
    """Calculate suitable metrics and return them in a simple dictionary."""

    if len(y_true) != len(y_pred):
        raise ValueError("Actual and predicted values must have the same length.")

    if task_type == "classification":
        # Weighted averages support multiclass classification too.
        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision_weighted": float(
                precision_score(y_true, y_pred, average="weighted", zero_division=0)
            ),
            "recall_weighted": float(
                recall_score(y_true, y_pred, average="weighted", zero_division=0)
            ),
            "f1_weighted": float(
                f1_score(y_true, y_pred, average="weighted", zero_division=0)
            ),
        }

    elif task_type == "regression":
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        metrics = {
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "rmse": rmse,
            "r2": float(r2_score(y_true, y_pred)),
        }

    else:
        raise ValueError("task_type must be 'classification' or 'regression'.")

    return {"metrics": metrics}
