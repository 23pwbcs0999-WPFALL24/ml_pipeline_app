"""Model inference pipeline. This module loads a model; it never trains one."""

from pathlib import Path
import joblib
import pandas as pd


def load_model(model_path):
    """Load an already-fitted model or scikit-learn Pipeline from disk."""
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}. "
            "Place your compatible pretrained model at models/model.pkl."
        )

    # joblib is commonly used to save/load scikit-learn models.
    return joblib.load(model_path)


def make_predictions(fitted_model, X_test: pd.DataFrame):
    """Use the loaded model to predict values for the test features."""
    if not hasattr(fitted_model, "predict"):
        raise TypeError("The loaded file does not contain an object with a predict() method.")

    # The model must have been fitted before it was saved.
    return fitted_model.predict(X_test)
