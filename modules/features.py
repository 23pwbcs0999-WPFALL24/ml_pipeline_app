"""Feature analysis and train/test splitting pipeline."""

import pandas as pd
from sklearn.model_selection import train_test_split


def prepare_features(
    cleaned_df: pd.DataFrame,
    target_column: str,
    test_size: float = 0.2,
    task_type: str = "classification",
):
    """Prepare X/y, create a correlation matrix and split the dataset.

    Returns:
        X_train, X_test, y_train, y_test, correlation_matrix

    This function does not scale or encode data. Those steps must match the
    preprocessing used by the already-fitted model. Prefer storing them inside
    the model's scikit-learn Pipeline.
    """

    if target_column not in cleaned_df.columns:
        raise ValueError(f"Target column '{target_column}' does not exist.")

    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    # X contains the input columns; y contains the value to predict.
    X = cleaned_df.drop(columns=[target_column])
    y = cleaned_df[target_column]

    if X.shape[1] == 0:
        raise ValueError("The dataset needs at least one input feature besides the target.")

    if len(cleaned_df) < 5:
        raise ValueError("At least 5 usable rows are recommended for a train/test split.")

    # Correlation is calculated only for numeric columns.
    # It is useful for basic exploration, but correlation does not prove causation.
    numeric_df = cleaned_df.select_dtypes(include="number")
    correlation = numeric_df.corr() if numeric_df.shape[1] >= 2 else None

    # Stratification helps keep class proportions similar in both splits.
    # It is used only when every class has enough examples.
    stratify_values = None
    if task_type == "classification":
        class_counts = y.value_counts()
        if len(class_counts) > 1 and class_counts.min() >= 2:
            stratify_values = y

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=42,
            stratify=stratify_values,
        )
    except ValueError as error:
        # Give a useful message rather than silently making a different split.
        raise ValueError(
            "Could not split this dataset. Try a different test percentage or "
            "check whether there are enough examples in each target class. "
            f"Details: {error}"
        ) from error

    return X_train, X_test, y_train, y_test, correlation
