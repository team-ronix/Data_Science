# Data Science Project

This repository contains data collection, cleaning, validation, and exploratory analysis for Lending Club loan data enriched with macroeconomic indicators.

## Setup

Install dependencies with Poetry:

```bash
poetry install
```

## Run Pipelines

```bash
poetry run python DataCollection.py
poetry run python DataCleaning.py
poetry run python DataValidation.py
poetry run python EDA.py
```

## EDA Deliverables

Running `EDA.py` produces:

- At least five major visualizations with interpretation-ready summaries
- Feature-to-target analysis outputs
- Baseline model comparison metrics
- A dashboard combining EDA findings, model comparisons, and business insights

Generated artifacts are saved in:

- `eda_outputs/plots/`
- `eda_outputs/feature_target_numeric_correlations.csv`
- `eda_outputs/feature_target_categorical_summary.csv`
- `eda_outputs/model_comparison_metrics.csv`
- `eda_outputs/eda_interpretations.txt`
- `eda_outputs/dashboard/eda_dashboard.html`