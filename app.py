"""Main Streamlit app for a beginner-friendly end-to-end ML workflow.

Run from the project folder with:
    streamlit run app.py

The app does NOT train a model. It expects an already-fitted model or
scikit-learn Pipeline at models/model.pkl.
"""

from pathlib import Path
import uuid

import pandas as pd
import streamlit as st

from modules.cleaning import clean_data
from modules.features import prepare_features
from modules.database import (
    init_db,
    save_pipeline_run,
    save_predictions,
    update_run_results,
    get_recent_runs,
)
from modules.model import load_model, make_predictions
from modules.evaluation import evaluate_model
from modules.visualization import (
    create_correlation_heatmap,
    create_confusion_matrix,
    create_regression_plot,
)


# Keep paths relative to this file, so the app works from any terminal folder.
PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "models" / "model.pkl"
UPLOAD_DIR = PROJECT_DIR / "data" / "uploaded"


st.set_page_config(
    page_title="ML Pipeline Practice",
    page_icon="🧪",
    layout="wide",
)

# Create the SQLite tables when the application starts.
init_db()
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


st.title(" End-to-End Machine Learning Pipeline")
st.write(
    "Upload a CSV, clean and inspect it, prepare features, store the results, "
    "and apply an existing model. This practice app does not train a model."
)


# ----------------------------- Sidebar -----------------------------------
st.sidebar.header("Pipeline stages")
st.sidebar.markdown(
    "1. CSV upload\n\n"
    "2. Data cleaning\n\n"
    "3. Feature analysis and splitting\n\n"
    "4. Database storage\n\n"
    "5. Model prediction\n\n"
    "6. Evaluation and visualization"
)

st.sidebar.divider()
st.sidebar.subheader("Existing model")
if MODEL_PATH.exists():
    st.sidebar.success("models/model.pkl found")
else:
    st.sidebar.warning("models/model.pkl not found")
    st.sidebar.caption(
        "You can still test upload, cleaning, feature preparation and database storage. "
        "Predictions require a compatible pretrained model file."
    )


# ----------------------------- CSV upload --------------------------------
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is None:
    st.info("Upload a CSV file to begin.")
else:
    try:
        # Read the uploaded file into a Pandas DataFrame.
        raw_df = pd.read_csv(uploaded_file)

        if raw_df.empty:
            st.error("The uploaded CSV has no rows.")
            st.stop()

        st.subheader("1. Dataset preview")
        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", f"{raw_df.shape[0]:,}")
        col2.metric("Columns", f"{raw_df.shape[1]:,}")
        col3.metric("Missing cells", f"{int(raw_df.isna().sum().sum()):,}")
        st.dataframe(raw_df.head(10), use_container_width=True)

        # The target is the value the model is expected to predict.
        target_column = st.selectbox(
            "Select the target column",
            options=list(raw_df.columns),
            help="Choose the column containing the correct answer/label.",
        )

        task_type = st.radio(
            "Select the problem type",
            options=["classification", "regression"],
            horizontal=True,
        )

        test_size_percent = st.slider(
            "Test data percentage",
            min_value=10,
            max_value=40,
            value=20,
            step=5,
        )
        test_size = test_size_percent / 100

        run_button = st.button("Run pipeline", type="primary", use_container_width=True)

        if run_button:
            try:
                # Give each execution a short unique ID to connect its database records.
                run_id = uuid.uuid4().hex[:10]

                # Save a copy of the original upload for this practice run.
                safe_filename = Path(uploaded_file.name).name
                saved_upload_path = UPLOAD_DIR / f"{run_id}_{safe_filename}"
                saved_upload_path.write_bytes(uploaded_file.getvalue())

                # STAGE 1: Basic cleaning.
                cleaned_df, cleaning_report = clean_data(raw_df, target_column)

                # STAGE 2: Correlation analysis, X/y separation and data splitting.
                X_train, X_test, y_train, y_test, correlation = prepare_features(
                    cleaned_df=cleaned_df,
                    target_column=target_column,
                    test_size=test_size,
                    task_type=task_type,
                )

                # Keep a small summary of the prepared data in the database.
                # We store shapes and feature names here to keep the example simple.
                feature_summary = {
                    "feature_columns": list(X_train.columns),
                    "X_train_rows": int(X_train.shape[0]),
                    "X_test_rows": int(X_test.shape[0]),
                    "y_train_rows": int(y_train.shape[0]),
                    "y_test_rows": int(y_test.shape[0]),
                }

                # STAGE 3: Save cleaned data and feature information in SQLite.
                save_pipeline_run(
                    run_id=run_id,
                    filename=safe_filename,
                    target_column=target_column,
                    original_rows=len(raw_df),
                    cleaned_df=cleaned_df,
                    feature_summary=feature_summary,
                    status="data_prepared",
                )

                # Show the cleaning and feature stages even when no model is present.
                st.success(f"Data preparation completed. Run ID: {run_id}")

                st.subheader("2. Data cleaning results")
                st.json(cleaning_report)
                st.dataframe(cleaned_df.head(10), use_container_width=True)

                st.subheader("3. Feature preparation")
                a, b, c = st.columns(3)
                a.metric("Training rows", len(X_train))
                b.metric("Testing rows", len(X_test))
                c.metric("Input features", X_train.shape[1])
                st.write("Feature columns:", list(X_train.columns))

                # Display the correlation heatmap when there are numeric columns.
                heatmap_fig = create_correlation_heatmap(correlation)
                if heatmap_fig is not None:
                    st.pyplot(heatmap_fig, use_container_width=True)
                else:
                    st.info("No numeric columns were available for a correlation heatmap.")

                st.subheader("4. Database storage")
                st.success("Cleaned data and feature summary saved to SQLite.")

                # STAGE 4: Load an already-trained model and generate predictions.
                if not MODEL_PATH.exists():
                    st.warning(
                        "Prediction and evaluation were skipped because "
                        "models/model.pkl does not exist yet. Add a compatible "
                        "pretrained model file to continue."
                    )
                else:
                    fitted_model = load_model(MODEL_PATH)
                    predictions = make_predictions(fitted_model, X_test)

                    # STAGE 5: Evaluate predictions against the actual test labels.
                    evaluation_results = evaluate_model(
                        y_true=y_test,
                        y_pred=predictions,
                        task_type=task_type,
                    )

                    # STAGE 6: Store predictions and evaluation metrics in SQLite.
                    save_predictions(run_id, y_test, predictions)
                    update_run_results(
                        run_id=run_id,
                        metrics=evaluation_results["metrics"],
                        status="completed",
                    )

                    st.subheader("5. Model predictions")
                    prediction_df = pd.DataFrame(
                        {
                            "Actual": y_test.reset_index(drop=True),
                            "Predicted": pd.Series(predictions),
                        }
                    )
                    st.dataframe(prediction_df.head(20), use_container_width=True)

                    # Allow the user to download the prediction results.
                    csv_bytes = prediction_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "Download predictions as CSV",
                        data=csv_bytes,
                        file_name=f"predictions_{run_id}.csv",
                        mime="text/csv",
                    )

                    st.subheader("6. Model evaluation")
                    metric_columns = st.columns(max(1, len(evaluation_results["metrics"])))
                    for index, (metric_name, metric_value) in enumerate(
                        evaluation_results["metrics"].items()
                    ):
                        metric_columns[index].metric(
                            metric_name.upper(), f"{metric_value:.4f}"
                        )

                    # Show a suitable chart for the selected problem type.
                    if task_type == "classification":
                        evaluation_fig = create_confusion_matrix(y_test, predictions)
                    else:
                        evaluation_fig = create_regression_plot(y_test, predictions)

                    if evaluation_fig is not None:
                        st.pyplot(evaluation_fig, use_container_width=True)

                    st.success("Pipeline completed successfully.")

            except Exception as error:
                st.error(f"The pipeline could not complete: {error}")
                st.caption(
                    "Check that the selected target, dataset columns and pretrained model "
                    "are compatible. The model must accept the uploaded feature columns."
                )

    except Exception as error:
        st.error(f"Could not read the CSV file: {error}")


# ----------------------------- Run history -------------------------------
st.divider()
st.subheader("Previous pipeline runs")
run_history = get_recent_runs(limit=10)
if run_history:
    history_df = pd.DataFrame(run_history)
    st.dataframe(history_df, use_container_width=True, hide_index=True)
else:
    st.caption("No pipeline runs have been saved yet.")
