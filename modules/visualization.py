"""Small plotting functions used by the Streamlit app."""

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay


def create_correlation_heatmap(correlation):
    """Return a heatmap figure, or None when there is no numeric correlation."""
    if correlation is None or correlation.empty:
        return None

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(
        correlation,
        annot=True,
        cmap="coolwarm",
        center=0,
        fmt=".2f",
        ax=ax,
    )
    ax.set_title("Feature Correlation Heatmap")
    fig.tight_layout()
    return fig


def create_confusion_matrix(y_true, y_pred):
    """Return a confusion matrix figure for classification."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax, colorbar=False)
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    return fig


def create_regression_plot(y_true, y_pred):
    """Return an actual-vs-predicted scatter plot for regression."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_true, y_pred, alpha=0.7)

    # Draw a reference line so perfect predictions are easy to recognize.
    values = list(y_true) + list(y_pred)
    minimum = min(values)
    maximum = max(values)
    ax.plot([minimum, maximum], [minimum, maximum], linestyle="--")

    ax.set_xlabel("Actual values")
    ax.set_ylabel("Predicted values")
    ax.set_title("Actual vs Predicted")
    fig.tight_layout()
    return fig
