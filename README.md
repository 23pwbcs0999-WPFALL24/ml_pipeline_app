# ML Pipeline Practice App

A beginner-friendly Streamlit project that demonstrates these stages:

1. CSV upload
2. Basic data cleaning
3. Correlation analysis and train/test split
4. SQLite storage
5. Loading an existing model and making predictions
6. Evaluation and visualization

**This project does not train a model.** Add an already-fitted compatible model
as `models/model.pkl` to enable prediction and evaluation. The model should
ideally be a complete scikit-learn `Pipeline` containing its preprocessing.

## 1. Create and activate a virtual environment (Windows PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can use the environment's Python directly:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m streamlit run app.py
```

## 2. Install packages

```powershell
python -m pip install -r requirements.txt
```

## 3. Run the app

```powershell
streamlit run app.py
```

## 4. Add your existing model

Put the model file here:

```text
models/model.pkl
```

The model must already be fitted and compatible with the CSV features. It must
have a `predict()` method. No training is performed by this app.

## Notes for this learning version

- Data cleaning removes fully empty rows/columns, duplicate rows, and rows with
  a missing target.
- Missing values in input features are reported but not filled here. For a
  correct train/test workflow, imputation and scaling should be fitted on the
  training data only. A complete model Pipeline can perform those steps.
- Correlation analysis uses numeric columns only.
- SQLite stores cleaned data and a summary of the prepared features as JSON,
  along with predictions and evaluation metrics. This is suitable for small
  practice datasets, not large production datasets.
