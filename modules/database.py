"""Simple SQLite storage for practice pipeline runs and predictions."""

from pathlib import Path
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime

import pandas as pd


# Store the database inside the project folder, not the current terminal folder.
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATABASE_DIR = PROJECT_DIR / "database"
DATABASE_PATH = DATABASE_DIR / "ml_pipeline.db"


@contextmanager
def get_connection():
    """Open a database connection and always close it when finished."""
    DATABASE_DIR.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row

    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def init_db():
    """Create the tables if they do not already exist."""
    with get_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS pipeline_runs (
                run_id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                target_column TEXT NOT NULL,
                created_at TEXT NOT NULL,
                original_rows INTEGER,
                cleaned_rows INTEGER,
                status TEXT NOT NULL,
                cleaned_data_json TEXT,
                feature_summary_json TEXT,
                metrics_json TEXT
            )
            """
        )

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                actual_value TEXT,
                predicted_value TEXT,
                FOREIGN KEY (run_id) REFERENCES pipeline_runs (run_id)
            )
            """
        )


def save_pipeline_run(
    run_id: str,
    filename: str,
    target_column: str,
    original_rows: int,
    cleaned_df: pd.DataFrame,
    feature_summary: dict,
    status: str = "data_prepared",
):
    """Save the cleaned dataset and feature summary for one run.

    JSON is used here to keep the beginner example easy to follow. For very
    large datasets, storing the full DataFrame in a database this way is not
    recommended.
    """

    cleaned_json = cleaned_df.to_json(orient="records", date_format="iso")
    feature_json = json.dumps(feature_summary, default=str)

    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO pipeline_runs (
                run_id, filename, target_column, created_at,
                original_rows, cleaned_rows, status,
                cleaned_data_json, feature_summary_json, metrics_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                filename,
                target_column,
                datetime.now().isoformat(timespec="seconds"),
                int(original_rows),
                int(len(cleaned_df)),
                status,
                cleaned_json,
                feature_json,
                None,
            ),
        )


def save_predictions(run_id: str, y_true, y_pred):
    """Save actual and predicted values for every test row."""
    rows_to_save = [
        (run_id, str(actual), str(predicted))
        for actual, predicted in zip(list(y_true), list(y_pred))
    ]

    with get_connection() as connection:
        # Delete any old predictions for this run before inserting new ones.
        connection.execute("DELETE FROM predictions WHERE run_id = ?", (run_id,))
        connection.executemany(
            """
            INSERT INTO predictions (run_id, actual_value, predicted_value)
            VALUES (?, ?, ?)
            """,
            rows_to_save,
        )


def update_run_results(run_id: str, metrics: dict, status: str = "completed"):
    """Save evaluation metrics and update the run status."""
    metrics_json = json.dumps(metrics, default=float)

    with get_connection() as connection:
        connection.execute(
            """
            UPDATE pipeline_runs
            SET metrics_json = ?, status = ?
            WHERE run_id = ?
            """,
            (metrics_json, status, run_id),
        )


def get_recent_runs(limit: int = 10):
    """Return a small list of the most recent runs for the Streamlit UI."""
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT run_id, filename, target_column, created_at,
                   original_rows, cleaned_rows, status
            FROM pipeline_runs
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (int(limit),),
        ).fetchall()

    return [dict(row) for row in rows]
