# Applied Data Science Project

This project builds an end-to-end credit risk workflow for Lending Club loans. It combines loan records with macroeconomic indicators, validates and transforms the data, explores the main drivers of default, and trains classification models that prioritize reducing false negatives.

## Business Goal

The target problem is binary classification: predict whether a loan will be fully paid or charged off/default. The business objective is to support safer underwriting decisions, reduce losses from approving bad loans, and keep the rejection of good borrowers under control.

## Data Sources

- Lending Club loan dataset: `data/Lending Club loan.csv`
- FRED macroeconomic indicators: CPI, unemployment rate, and federal funds rate

The collection pipeline merges the Lending Club data with the macroeconomic series and produces the combined dataset used by the rest of the project.

## Project Pipeline

The workflow is organized into the following stages:

1. `src/DataCollection.py` collects and merges the raw loan and macroeconomic data.
2. `src/DataCleaning.py` cleans the merged dataset.
3. `src/DataValidation.py` checks completeness, duplicates, categorical consistency, and relationships.
4. `src/DataTransformation.py` filters labels, engineers features, encodes categorical variables, creates train/test splits, and normalizes selected numeric fields.
5. `DataUndersampling.py` balances the training data.
6. `src/Model.py` trains and evaluates the model set with F2-focused tuning.
7. `EDA.py` generates the main visualizations, feature-to-target analysis, and dashboard outputs.

## Key Modeling Notes

- The target variable is derived from `loan_status`.
- Current loans are removed because their outcome is unresolved.
- Leakage-prone fields such as repayment outcome variables are excluded from the final modeling feature set.
- The strongest signals in the report come from `sub_grade`, `int_rate`, `term`, `fico`, `dti`, and macroeconomic indicators.
- Model selection prioritizes recall and F2-score because false negatives are the most expensive error for the lender.

## Repository Structure

- `app/` - FastAPI inference service
- `src/` - collection, cleaning, validation, transformation, and model training code
- `data/` - raw, cleaned, transformed, and split datasets
- `reports/figures/` - EDA and analysis figures used in the report
- `models/` - saved model artifact for the API
- `tests/` - unit tests for the pipeline

## Setup

Install dependencies with Poetry:

```bash
poetry install
```

## Common Workflows

Use the Makefile for the standard project commands:

```bash
make install
make preprocess
make train
make evaluate
make test
```

Available targets include:

- `make preprocess` to run collection, validation, cleaning, transformation, and undersampling
- `make train` to train and evaluate the classifier pipeline
- `make evaluate` to run the model workflow
- `make test` to run the test suite with coverage reporting
- `make all` to run preprocessing, training, and tests

You can also run the scripts directly:

```bash
poetry run python src/DataCollection.py
poetry run python src/DataCleaning.py
poetry run python src/DataValidation.py
poetry run python src/DataTransformation.py
poetry run python DataUndersampling.py
poetry run python src/Model.py
poetry run python EDA.py
```

## EDA Outputs

Running `EDA.py` produces visualizations, summary tables, and dashboard-ready outputs that support the report narrative. The generated artifacts are saved in:

- `reports/figures/`
- `reports/eda_interpretations.txt`
- `reports/model_comparison_metrics.csv`
- `reports/feature_target_numeric_correlations.csv`
- `reports/feature_target_categorical_summary.csv`
- `reports/dashboard/eda_dashboard.html`

## Validation and Model Evaluation

The validation pipeline confirms data quality before modeling. The model training workflow logs standard metrics and business metrics, including accuracy, precision, recall, ROC-AUC, F2-score, and custom loss/profit proxies.

The final model comparison is based on test F2-score, with recall emphasized more than precision because catching default risk is the primary business concern.

## CI/CD

- GitHub Actions runs tests, linting, and type checks on pushes and pull requests to `master`.
- Cloud Build is used for Docker image creation and Cloud Run deployment.
- The API reads the deployed model through `MODEL_NAME` and `MODEL_PATH` environment variables.

## Inference API

The FastAPI service in `app/main.py` loads the saved model bundle and exposes a health endpoint and a prediction endpoint. It expects the same feature order and normalization metadata used during training.
