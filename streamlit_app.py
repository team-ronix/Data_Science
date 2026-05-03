from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "model_outputs"
PLOTS_DIR = OUTPUT_DIR / "plots"
FIGURES_DIR = ROOT / "reports" / "figures"
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


_DEFAULT_STATUSES = {
    "charged off", "default",
    "late (31-120 days)", "late (16-30 days)",
    "does not meet the credit policy. status:charged off",
}
_NON_DEFAULT_STATUSES = {
    "fully paid",
    "does not meet the credit policy. status:fully paid",
}
_EDA_RAW_COLS = ["loan_status", "int_rate", "sub_grade", "term", "issue_d", "loan_amnt", "emp_length", "purpose"]


@st.cache_data
def load_raw_eda() -> pd.DataFrame:
    path = DATA_DIR / "merged_df_cleaned.csv"
    available = pd.read_csv(path, nrows=0).columns.tolist()
    cols = [c for c in _EDA_RAW_COLS if c in available]
    df = pd.read_csv(path, usecols=cols)
    if "loan_status" in df.columns:
        status = df["loan_status"].astype(str).str.strip().str.lower()
        df = df[status != "current"].copy()
        status = df["loan_status"].astype(str).str.strip().str.lower()
        df["target_default"] = None
        df.loc[status.isin(_DEFAULT_STATUSES), "target_default"] = 1
        df.loc[status.isin(_NON_DEFAULT_STATUSES), "target_default"] = 0
        df["target_default"] = pd.to_numeric(df["target_default"], errors="coerce")
        df = df.dropna(subset=["target_default"])
        df["target_default"] = df["target_default"].astype(int)
    return df


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


def show_home(scores: pd.DataFrame) -> None:
    test_scores = scores[scores["Stage"] == "test"].copy()
    best_row = test_scores.sort_values("F2-Score", ascending=False).iloc[0]

    st.header("Project Overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Best Model", str(best_row["Model"]))
    c2.metric("F2 Score", format_metric(float(best_row["F2-Score"])))
    c3.metric("ROC-AUC", format_metric(float(best_row["ROC-AUC"])))

    st.markdown(
        "The goal is to predict whether a loan will be **Fully Paid** or **Charged Off**. "
        "Below are the five most important patterns found in the data."
    )

    df = load_raw_eda()

    # 1. Class imbalance
    st.subheader("1. Class Imbalance")
    if "target_default" in df.columns:
        n0 = int((df["target_default"] == 0).sum())
        n1 = int((df["target_default"] == 1).sum())
        fig_imb = make_subplots(rows=1, cols=2, specs=[[{"type": "bar"}, {"type": "pie"}]])
        fig_imb.add_trace(go.Bar(
            x=["Fully Paid", "Charge Off"], y=[n0, n1],
            marker_color=["#4C72B0", "#DD8452"],
            text=[f"{v:,}<br>({100*v/(n0+n1):.1f}%)" for v in [n0, n1]],
            textposition="outside", name=""
        ), row=1, col=1)
        fig_imb.add_trace(go.Pie(
            labels=["Fully Paid", "Charge Off"], values=[n0, n1],
            marker_colors=["#4C72B0", "#DD8452"], hole=0.3, name=""
        ), row=1, col=2)
        fig_imb.update_layout(title="Class Imbalance", showlegend=False, height=380)
        st.plotly_chart(fig_imb, use_container_width=True)

    # 2. Charge-Off Rate by sub-grade
    st.subheader("2. Charge-Off Rate by Loan Sub-Grade")
    if "sub_grade" in df.columns and "target_default" in df.columns:
        grade_stats = (
            df.groupby("sub_grade")["target_default"]
            .agg(["mean", "count"])
            .rename(columns={"mean": "default_rate", "count": "n_loans"})
            .reset_index()
            .sort_values("sub_grade")
        )
        grade_stats["default_pct"] = (grade_stats["default_rate"] * 100).round(1)
        fig_grade = make_subplots(rows=1, cols=2, subplot_titles=("Charge-Off Rate (%)", "Loan Volume"))
        fig_grade.add_trace(go.Bar(
            x=grade_stats["sub_grade"], y=grade_stats["default_pct"],
            marker_color="#E74C3C", name="Charge-Off Rate",
            hovertemplate="%{x}: %{y:.1f}%"
        ), row=1, col=1)
        fig_grade.add_trace(go.Bar(
            x=grade_stats["sub_grade"], y=grade_stats["n_loans"],
            marker_color="#3498DB", name="Loan Count",
            hovertemplate="%{x}: %{y:,}"
        ), row=1, col=2)
        fig_grade.update_layout(showlegend=False, height=380)
        st.plotly_chart(fig_grade, use_container_width=True)

    # 3. Interest rate by default status
    st.subheader("3. Interest Rate vs Default Status")
    if "int_rate" in df.columns and "target_default" in df.columns:
        int_df = df[["int_rate", "target_default"]].dropna().copy()
        if int_df["int_rate"].dtype == object:
            int_df["int_rate"] = int_df["int_rate"].str.replace("%", "").astype(float)
        int_df["Status"] = int_df["target_default"].map({0: "Fully Paid", 1: "Charge Off"})
        fig_rate = px.box(
            int_df, x="Status", y="int_rate",
            color="Status", color_discrete_map={"Fully Paid": "#4C72B0", "Charge Off": "#DD8452"},
            labels={"int_rate": "Interest Rate (%)"},
            title="Interest Rate Distribution by Outcome",
            points=False,
        )
        fig_rate.update_layout(showlegend=False, height=380)
        st.plotly_chart(fig_rate, use_container_width=True)

    # 4. Charge-Off Rate by term
    st.subheader("4. Loan Term Effect")
    if "term" in df.columns and "target_default" in df.columns:
        term_stats = (
            df.groupby("term")["target_default"]
            .agg(["mean", "count"])
            .rename(columns={"mean": "default_rate", "count": "n"})
            .reset_index()
        )
        term_stats["default_pct"] = (term_stats["default_rate"] * 100).round(1)
        term_stats["term_label"] = term_stats["term"].astype(str).str.strip()
        fig_term = make_subplots(
            rows=1, cols=2, subplot_titles=("Charge-Off Rate (%)", "Loan Count"),
            specs=[[{"type": "bar"}, {"type": "bar"}]]
        )
        fig_term.add_trace(go.Bar(
            x=term_stats["term_label"], y=term_stats["default_pct"],
            marker_color="#E74C3C", text=term_stats["default_pct"].map(lambda v: f"{v:.1f}%"),
            textposition="outside", name="Charge-Off Rate"
        ), row=1, col=1)
        fig_term.add_trace(go.Bar(
            x=term_stats["term_label"], y=term_stats["n"],
            marker_color="#3498DB", name="Count"
        ), row=1, col=2)
        fig_term.update_layout(showlegend=False, height=380)
        st.plotly_chart(fig_term, use_container_width=True)

    # 5. Employment length
    st.subheader("5. Employment Length")
    if "emp_length" in df.columns and "target_default" in df.columns:
        emp_df = df[["emp_length", "target_default"]].dropna(subset=["emp_length"]).copy()
        import re
        emp_df["emp_yrs"] = emp_df["emp_length"].astype(str).apply(
            lambda x: 0 if "< 1" in x else (10 if "10+" in x else int(re.sub(r"\D", "", x) or -1))
        )
        emp_df = emp_df[emp_df["emp_yrs"] >= 0]
        emp_stats = (
            emp_df.groupby("emp_yrs")["target_default"]
            .agg(["mean", "count"])
            .rename(columns={"mean": "charge_off_rate", "count": "n"})
            .reset_index().sort_values("emp_yrs")
        )
        emp_stats["charge_off_pct"] = (emp_stats["charge_off_rate"] * 100).round(2)
        emp_stats["label"] = emp_stats["emp_yrs"].apply(lambda x: "< 1 yr" if x == 0 else ("10+ yrs" if x == 10 else f"{x} yrs"))
        fig_emp = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Loan Volume by Employment Length", "Charge-Off Rate by Employment Length"),
            specs=[[{"type": "bar"}, {"type": "scatter"}]],
        )
        fig_emp.add_trace(go.Bar(
            x=emp_stats["label"], y=emp_stats["n"],
            marker_color="#5DADE2", name="Loan Count",
            hovertemplate="%{x}: %{y:,} loans"
        ), row=1, col=1)
        fig_emp.add_trace(go.Scatter(
            x=emp_stats["label"], y=emp_stats["charge_off_pct"],
            mode="lines+markers", line_color="#E74C3C", name="Charge-Off Rate",
            hovertemplate="%{x}: %{y:.2f}%"
        ), row=1, col=2)
        fig_emp.update_yaxes(title_text="Count", row=1, col=1)
        fig_emp.update_yaxes(title_text="Charge-Off Rate (%)", row=1, col=2)
        fig_emp.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig_emp, use_container_width=True)

    # 6. Loan purpose
    st.subheader("6. Loan Purpose")
    if "purpose" in df.columns and "target_default" in df.columns:
        purpose_stats = (
            df.dropna(subset=["purpose"]).groupby("purpose")["target_default"]
            .agg(["mean", "count"])
            .rename(columns={"mean": "charge_off_rate", "count": "n"})
            .reset_index()
        )
        purpose_stats["charge_off_pct"] = (purpose_stats["charge_off_rate"] * 100).round(1)
        by_rate = purpose_stats.sort_values("charge_off_pct")
        by_vol  = purpose_stats.sort_values("n")
        fig_purp = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Charge-Off Rate by Purpose", "Loan Volume by Purpose"),
        )
        fig_purp.add_trace(go.Bar(
            x=by_rate["charge_off_pct"], y=by_rate["purpose"],
            orientation="h", marker_color="#E74C3C", name="Charge-Off Rate",
            hovertemplate="%{y}: %{x:.1f}%"
        ), row=1, col=1)
        fig_purp.add_trace(go.Bar(
            x=by_vol["n"], y=by_vol["purpose"],
            orientation="h", marker_color="#3498DB", name="Loan Count",
            hovertemplate="%{y}: %{x:,}"
        ), row=1, col=2)
        fig_purp.update_xaxes(title_text="Charge-Off Rate (%)", row=1, col=1)
        fig_purp.update_xaxes(title_text="Loans", row=1, col=2)
        fig_purp.update_layout(showlegend=False, height=460)
        st.plotly_chart(fig_purp, use_container_width=True)

    # 7. Temporal trends
    st.subheader("7. Temporal Trends")
    if "issue_d" in df.columns and "target_default" in df.columns and "loan_amnt" in df.columns:
        tmp = df[["issue_d", "target_default", "loan_amnt"]].copy()
        tmp["issue_d"] = pd.to_datetime(tmp["issue_d"], errors="coerce")
        tmp = tmp.dropna(subset=["issue_d"])
        tmp["ym"] = tmp["issue_d"].dt.to_period("M").astype(str)
        monthly = (
            tmp.groupby("ym")
            .agg(n_loans=("loan_amnt", "count"),
                 default_rate=("target_default", "mean"),
                 avg_loan=("loan_amnt", "mean"))
            .reset_index()
        )
        monthly["default_pct"] = (monthly["default_rate"] * 100).round(2)
        fig_time = make_subplots(
            rows=3, cols=1, shared_xaxes=True,
            subplot_titles=("Monthly Loan Originations", "Monthly Charge-Off Rate (%)", "Avg Loan Amount ($)"),
            vertical_spacing=0.08,
        )
        fig_time.add_trace(go.Scatter(x=monthly["ym"], y=monthly["n_loans"],
            mode="lines", line_color="#3498DB", name="Originations"), row=1, col=1)
        fig_time.add_trace(go.Scatter(x=monthly["ym"], y=monthly["default_pct"],
            mode="lines", line_color="#E74C3C", name="Charge-Off Rate"), row=2, col=1)
        fig_time.add_trace(go.Scatter(x=monthly["ym"], y=monthly["avg_loan"],
            mode="lines", line_color="#27AE60", name="Avg Loan"), row=3, col=1)
        fig_time.update_layout(showlegend=False, height=560)
        st.plotly_chart(fig_time, use_container_width=True)


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

    st.title("Loan Risk Dashboard") 
    tab0, tab1, tab2, tab3 = st.tabs(["Home", "EDA", "Model Results", "Business Insights"])

    with tab0:
        show_home(scores)
    with tab1:
        show_eda(train)
    with tab2:
        show_model_results(scores)
    with tab3:
        show_business_insights(scores)


if __name__ == "__main__":
    main()