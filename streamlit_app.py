from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "model_outputs"
PLOTS_DIR = OUTPUT_DIR / "plots"
EDA_COLUMNS = ["loan_status", "loan_amnt", "annual_inc", "int_rate", "dti"]
EDA_SAMPLE_SIZE = 5000


@st.cache_data
def load_scores() -> pd.DataFrame:
    return pd.read_csv(OUTPUT_DIR / "model_scores.csv")


@st.cache_data
def load_baseline_delta(stage: str) -> pd.DataFrame:
    return pd.read_csv(OUTPUT_DIR / f"models_vs_baseline_{stage}.csv")


@st.cache_data
def load_train_data() -> pd.DataFrame:
    available_columns = pd.read_csv(DATA_DIR / "train_norm.csv", nrows=0).columns.tolist()
    use_columns = [column for column in EDA_COLUMNS if column in available_columns]
    return pd.read_csv(DATA_DIR / "train_norm.csv", usecols=use_columns)


@st.cache_data
def load_train_sample() -> pd.DataFrame:
    train = load_train_data()
    if len(train) <= EDA_SAMPLE_SIZE:
        return train.copy()
    return train.sample(n=EDA_SAMPLE_SIZE, random_state=42)


def format_metric(value: float) -> str:
    return f"{value:.3f}"


def show_overview(scores: pd.DataFrame) -> None:
    test_scores = scores[scores["Stage"] == "test"].copy()
    best_row = test_scores.sort_values("F2-Score", ascending=False).iloc[0]

    st.title("Loan Risk Dashboard")
    st.write(
        "This dashboard shows the main data patterns, model results, and business takeaways for the loan risk project."
    )

    first, second, third = st.columns(3)
    first.metric("Best Test Model", str(best_row["Model"]))
    second.metric("Best F2 Score", format_metric(float(best_row["F2-Score"])))
    third.metric("Best ROC-AUC", format_metric(float(best_row["ROC-AUC"])))

    st.subheader("What this model does")
    st.write(
        "The model helps estimate if a loan is more likely to be fully paid or charged off. The goal is to support better lending decisions."
    )


def show_eda(train: pd.DataFrame) -> None:
    st.header("EDA Findings")
    train_sample = load_train_sample()

    if "loan_status" in train.columns:
        status_map = {0: "Charge Off", 1: "Fully Paid"}
        status_counts = train["loan_status"].map(status_map).value_counts().reset_index()
        status_counts.columns = ["Loan Status", "Count"]
        fig = px.bar(
            status_counts,
            x="Loan Status",
            y="Count",
            color="Loan Status",
            title="Loan Status Distribution",
        )
        st.plotly_chart(fig, use_container_width=True)

    numeric_candidates = [
        column for column in ["loan_amnt", "annual_inc", "int_rate", "dti"] if column in train.columns
    ]
    if len(numeric_candidates) >= 2:
        fig = px.scatter(
            train_sample,
            x=numeric_candidates[0],
            y=numeric_candidates[1],
            color="loan_status" if "loan_status" in train.columns else None,
            title=f"{numeric_candidates[0]} vs {numeric_candidates[1]}",
            opacity=0.5,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Scatter chart uses a random sample of up to {EDA_SAMPLE_SIZE:,} rows for speed.")

    summary_columns = [column for column in ["loan_amnt", "annual_inc", "int_rate", "dti"] if column in train.columns]
    if summary_columns:
        st.subheader("Simple Data Summary")
        st.dataframe(train[summary_columns].describe().round(2), use_container_width=True)


def show_model_results(scores: pd.DataFrame) -> None:
    st.header("Model Performance")

    stage = st.radio("Choose dataset", ["test", "train"], horizontal=True)
    stage_scores = scores[scores["Stage"] == stage].copy().sort_values("F2-Score", ascending=False)
    st.dataframe(stage_scores.round(3), use_container_width=True)

    fig = px.bar(
        stage_scores,
        x="Model",
        y=["Accuracy", "Recall", "F2-Score", "ROC-AUC"],
        barmode="group",
        title=f"Model Comparison on {stage.title()} Data",
    )
    st.plotly_chart(fig, use_container_width=True)

    comparison_plot = PLOTS_DIR / f"model_comparison_{stage}.png"
    roc_plot = PLOTS_DIR / f"roc_curves_comparison_{stage}.png"

    left, right = st.columns(2)
    if comparison_plot.exists():
        left.image(str(comparison_plot), caption=f"Score comparison ({stage})")
    if roc_plot.exists():
        right.image(str(roc_plot), caption=f"ROC curves ({stage})")


def show_business_insights(scores: pd.DataFrame) -> None:
    st.header("Business Insights")

    test_scores = scores[scores["Stage"] == "test"].copy().sort_values("F2-Score", ascending=False)
    best_row = test_scores.iloc[0]
    baseline_delta = load_baseline_delta("test").sort_values("Delta F2-Score", ascending=False)

    st.write(f"Best model on test data: {best_row['Model']}")
    st.write(f"Recall: {format_metric(float(best_row['Recall']))}")
    st.write(f"F2 score: {format_metric(float(best_row['F2-Score']))}")
    st.write(
        "A higher recall and F2 score means the model is better at finding good loans, while still keeping mistakes visible."
    )

    st.subheader("Improvement over baseline")
    st.dataframe(baseline_delta.round(3), use_container_width=True)

    top_gain = baseline_delta.iloc[0]
    st.info(
        f"Compared with the baseline, {top_gain['Model']} gives the biggest F2 improvement on test data."
    )


def main() -> None:
    st.set_page_config(page_title="Loan Risk Dashboard", layout="wide")

    scores = load_scores()
    train = load_train_data()

    show_overview(scores)
    tab1, tab2, tab3 = st.tabs(["EDA", "Model Results", "Business Insights"])

    with tab1:
        show_eda(train)
    with tab2:
        show_model_results(scores)
    with tab3:
        show_business_insights(scores)


if __name__ == "__main__":
    main()