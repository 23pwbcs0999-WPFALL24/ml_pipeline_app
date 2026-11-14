"""Data cleaning pipeline.

This beginner version performs safe, basic cleaning only. It does not
impute feature values because imputation should be fitted on training data
and applied consistently to test data. A complete pretrained model Pipeline
can contain those preprocessing steps.
"""

import pandas as pd


def clean_data(df: pd.DataFrame, target_column: str):
    """Clean basic CSV issues and return (cleaned_df, cleaning_report).

    Steps:
    1. Make a copy so the original DataFrame remains unchanged.
    2. Remove leading/trailing spaces from column names.
    3. Remove completely empty rows and columns.
    4. Remove duplicate rows.
    5. Remove rows where the target value is missing.

    Missing values in input features are reported, not filled here. The
    loaded model should handle them through its fitted preprocessing steps.
    """

    cleaned_df = df.copy()
    original_rows = len(cleaned_df)

    # Clean accidental spaces in column names.
    cleaned_df.columns = cleaned_df.columns.astype(str).str.strip()

    # Check that the target column is still present after cleaning its name.
    if target_column not in cleaned_df.columns:
        raise ValueError(
            f"Target column '{target_column}' was not found. "
            "Check the column name and try again."
        )

    # Remove rows and columns that contain no data at all.
    cleaned_df = cleaned_df.dropna(axis=0, how="all")
    cleaned_df = cleaned_df.dropna(axis=1, how="all")

    # Remove duplicate records.
    duplicates_removed = int(cleaned_df.duplicated().sum())
    cleaned_df = cleaned_df.drop_duplicates()

    # A row without its correct target value cannot be used for supervised evaluation.
    missing_target_rows = int(cleaned_df[target_column].isna().sum())
    cleaned_df = cleaned_df.dropna(subset=[target_column])

    if cleaned_df.empty:
        raise ValueError("No rows remain after basic cleaning.")

    # Report feature missing values so the user can see what still needs handling.
    missing_values = {
        str(column): int(count)
        for column, count in cleaned_df.isna().sum().items()
        if count > 0
    }

    report = {
        "original_rows": int(original_rows),
        "rows_after_cleaning": int(len(cleaned_df)),
        "duplicate_rows_removed": duplicates_removed,
        "rows_with_missing_target_removed": missing_target_rows,
        "missing_values_remaining": missing_values,
        "column_data_types": {
            str(column): str(dtype)
            for column, dtype in cleaned_df.dtypes.items()
        },
    }

    return cleaned_df, report
