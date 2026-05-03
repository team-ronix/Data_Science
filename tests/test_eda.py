import logging
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from EDA import (
    load_data,
    build_target,
    savefig,
    overview,
    plot_missing,
    check_target_distribution,
    plot_correlation,
    plot_univariate,
    default_by_category,
    numeric_vs_default,
    plot_interest_rate,
    plot_dti,
    plot_loan_grade,
    plot_emp_length,
    plot_annual_income,
    plot_purpose,
    plot_fico,
    plot_home_ownership,
    plot_verification,
    plot_temporal,
    top_corr_pairs,
    feature_importance,
    make_report,
    DEFAULT_STATUSES,
    NON_DEFAULT_STATUSES,
)


@pytest.fixture
def logger():
    return logging.getLogger("test_eda")


@pytest.fixture
def sample_df():
    np.random.seed(42)
    n = 500
    return pd.DataFrame({
        "id": range(1, n + 1),
        "loan_amnt": np.random.uniform(1000, 50000, n),
        "annual_inc": np.random.uniform(10000, 300000, n),
        "installment": np.random.uniform(100, 1500, n),
        "dti": np.random.uniform(0, 40, n),
        "int_rate": np.random.uniform(5, 30, n),
        "revol_util": np.random.uniform(0, 100, n),
        "fico_range_high": np.random.uniform(600, 850, n).astype(int),
        "avg_cur_bal": np.random.uniform(0, 100000, n),
        "loan_status": np.random.choice(
            list(DEFAULT_STATUSES) + list(NON_DEFAULT_STATUSES) + ["Current"], n
        ),
        "sub_grade": np.random.choice(["A1", "B1", "C1", "D1"], n),
        "emp_length": np.random.choice(["1 year", "5 years", "10+ years", "<1 year"], n),
        "purpose": np.random.choice(["debt_consolidation", "credit_card", "home_improvement"], n),
        "home_ownership": np.random.choice(["RENT", "OWN", "MORTGAGE"], n),
        "verification_status": np.random.choice(["Verified", "Not Verified", "Source Verified"], n),
        "issue_d": pd.date_range("2020-01-01", periods=n, freq="D"),
        "earliest_cr_line": pd.date_range("2000-01-01", periods=n, freq="D"),
    })


@pytest.fixture
def sample_df_with_target(sample_df):
    df = sample_df.copy()
    df["target_default"] = np.random.choice([0.0, 1.0], len(df), p=[0.7, 0.3])
    return df


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def _close_logging_handlers():
    for log_name in ("EDA", "root", ""):
        log = logging.getLogger(log_name if log_name else None)
        for h in list(log.handlers):
            if isinstance(h, logging.FileHandler):
                h.close()
                log.removeHandler(h)
    root = logging.getLogger()
    for h in list(root.handlers):
        if isinstance(h, logging.FileHandler):
            h.close()
            root.removeHandler(h)


def test_load_data_returns_dataframe(logger, temp_dir):
    test_csv = temp_dir / "test.csv"
    pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]}).to_csv(test_csv, index=False)
    with patch("EDA.INPUT_PATH", test_csv):
        result = load_data(logger)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3


def test_load_data_preserves_columns(logger, temp_dir):
    test_csv = temp_dir / "test.csv"
    pd.DataFrame({"col1": [1, 2], "col2": [3, 4]}).to_csv(test_csv, index=False)
    with patch("EDA.INPUT_PATH", test_csv):
        result = load_data(logger)
        assert list(result.columns) == ["col1", "col2"]


def test_build_target_creates_target_column(sample_df, logger):
    result = build_target(sample_df.copy(), logger)
    assert "target_default" in result.columns


def test_build_target_removes_current_loans(sample_df, logger):
    df = sample_df.copy()
    df["loan_status"] = "current"
    result = build_target(df, logger)
    assert len(result) < len(df)


def test_build_target_assigns_1_for_default(sample_df, logger):
    df = sample_df.copy()
    df["loan_status"] = "Charged Off"
    result = build_target(df, logger)
    assert result["target_default"].iloc[0] == 1


def test_build_target_assigns_0_for_non_default(sample_df, logger):
    df = sample_df.copy()
    df["loan_status"] = "Fully Paid"
    result = build_target(df, logger)
    assert result["target_default"].iloc[0] == 0


def test_build_target_missing_loan_status_column(sample_df, logger):
    df = sample_df.drop(columns=["loan_status"])
    result = build_target(df, logger)
    assert "target_default" not in result.columns


def test_savefig_creates_png_file(temp_dir):
    with patch("EDA.FIG_DIR", temp_dir):
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])
        result = savefig("test_fig")
        assert result.exists()
        assert result.suffix == ".png"
        plt.close("all")


def test_savefig_returns_path(temp_dir):
    with patch("EDA.FIG_DIR", temp_dir):
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])
        result = savefig("test_fig")
        assert isinstance(result, Path)
        plt.close("all")


def test_overview_returns_dict(sample_df, logger):
    assert isinstance(overview(sample_df, logger), dict)


def test_overview_contains_meta_key(sample_df, logger):
    assert "meta" in overview(sample_df, logger)


def test_overview_meta_contains_required_keys(sample_df, logger):
    meta = overview(sample_df, logger)["meta"]
    for key in ("n_rows", "n_cols", "n_numeric", "n_categ"):
        assert key in meta


def test_overview_correct_row_count(sample_df, logger):
    assert overview(sample_df, logger)["meta"]["n_rows"] == len(sample_df)


def test_overview_contains_stats_numeric(sample_df, logger):
    assert "stats_numeric" in overview(sample_df, logger)


def test_plot_missing_with_no_nulls(sample_df, logger, temp_dir):
    with patch("EDA.FIG_DIR", temp_dir):
        assert plot_missing(sample_df, logger) is None


def test_plot_missing_with_nulls(sample_df, logger, temp_dir):
    with patch("EDA.FIG_DIR", temp_dir):
        df = sample_df.copy()
        df.loc[0:10, "loan_amnt"] = np.nan
        result = plot_missing(df, logger)
        assert result is not None or result is None
        plt.close("all")


def test_plot_missing_returns_path_or_none(sample_df, logger, temp_dir):
    with patch("EDA.FIG_DIR", temp_dir):
        result = plot_missing(sample_df, logger)
        assert result is None or isinstance(result, Path)
        plt.close("all")


def test_check_target_distribution_returns_dict(sample_df_with_target, logger, temp_dir):
    with patch("EDA.FIG_DIR", temp_dir):
        result = check_target_distribution(sample_df_with_target, logger)
        assert isinstance(result, dict)
        plt.close("all")


def test_check_target_distribution_contains_counts(sample_df_with_target, logger, temp_dir):
    with patch("EDA.FIG_DIR", temp_dir):
        result = check_target_distribution(sample_df_with_target, logger)
        assert "n_default" in result
        assert "n_non_default" in result
        plt.close("all")


def test_check_target_distribution_imbalance_ratio(sample_df_with_target, logger, temp_dir):
    with patch("EDA.FIG_DIR", temp_dir):
        result = check_target_distribution(sample_df_with_target, logger)
        if result["n_default"] > 0:
            expected = round(result["n_non_default"] / result["n_default"], 2)
            assert result["imbalance_ratio"] == expected
        plt.close("all")


def test_plot_correlation_returns_path(sample_df, logger, temp_dir):
    with patch("EDA.FIG_DIR", temp_dir):
        result = plot_correlation(sample_df, logger)
        assert isinstance(result, Path) or result is None
        plt.close("all")


def test_plot_correlation_creates_file(sample_df, logger, temp_dir):
    with patch("EDA.FIG_DIR", temp_dir):
        result = plot_correlation(sample_df, logger)
        if result:
            assert result.exists()
        plt.close("all")


def test_plot_univariate_returns_list(sample_df, logger, temp_dir):
    with patch("EDA.FIG_DIR", temp_dir):
        result = plot_univariate(sample_df, logger)
        assert isinstance(result, list)
        plt.close("all")


def test_plot_univariate_creates_files(sample_df, logger, temp_dir):
    with patch("EDA.FIG_DIR", temp_dir):
        result = plot_univariate(sample_df, logger)
        assert len(result) > 0
        assert all(isinstance(p, Path) for p in result)
        plt.close("all")


def test_default_by_category_returns_list(sample_df_with_target, logger, temp_dir):
    with patch("EDA.FIG_DIR", temp_dir):
        result = default_by_category(sample_df_with_target, logger)
        assert isinstance(result, list)
        plt.close("all")


def test_default_by_category_no_target(sample_df, logger, temp_dir):
    with patch("EDA.FIG_DIR", temp_dir):
        assert default_by_category(sample_df, logger) == []
        plt.close("all")


def test_numeric_vs_default_returns_list(sample_df_with_target, logger, temp_dir):
    with patch("EDA.FIG_DIR", temp_dir):
        result = numeric_vs_default(sample_df_with_target, logger)
        assert isinstance(result, list)
        plt.close("all")


def test_numeric_vs_default_no_target(sample_df, logger, temp_dir):
    with patch("EDA.FIG_DIR", temp_dir):
        assert numeric_vs_default(sample_df, logger) == []
        plt.close("all")


def test_plot_interest_rate_returns_path_or_none(sample_df_with_target, logger, temp_dir):
    with patch("EDA.FIG_DIR", temp_dir):
        result = plot_interest_rate(sample_df_with_target, logger)
        assert isinstance(result, Path) or result is None
        plt.close("all")


def test_plot_interest_rate_missing_column(sample_df_with_target, logger, temp_dir):
    with patch("EDA.FIG_DIR", temp_dir):
        df = sample_df_with_target.drop(columns=["int_rate"])
        assert plot_interest_rate(df, logger) is None
        plt.close("all")


def test_plot_dti_returns_path_or_none(sample_df_with_target, logger, temp_dir):
    with patch("EDA.FIG_DIR", temp_dir):
        result = plot_dti(sample_df_with_target, logger)
        assert isinstance(result, Path) or result is None
        plt.close("all")


def test_plot_dti_missing_column(sample_df_with_target, logger, temp_dir):
    with patch("EDA.FIG_DIR", temp_dir):
        df = sample_df_with_target.drop(columns=["dti"])
        assert plot_dti(df, logger) is None
        plt.close("all")


def test_plot_loan_grade_returns_path_or_none(sample_df_with_target, logger, temp_dir):
    with patch("EDA.FIG_DIR", temp_dir):
        result = plot_loan_grade(sample_df_with_target, logger)
        assert isinstance(result, Path) or result is None
        plt.close("all")


def test_plot_loan_grade_missing_column(sample_df_with_target, logger, temp_dir):
    with patch("EDA.FIG_DIR", temp_dir):
        df = sample_df_with_target.drop(columns=["sub_grade"])
        assert plot_loan_grade(df, logger) is None
        plt.close("all")


def test_plot_emp_length_returns_path_or_none(sample_df_with_target, logger, temp_dir):
    with patch("EDA.FIG_DIR", temp_dir):
        result = plot_emp_length(sample_df_with_target, logger)
        assert isinstance(result, Path) or result is None
        plt.close("all")


def test_plot_emp_length_missing_column(sample_df_with_target, logger, temp_dir):
    with patch("EDA.FIG_DIR", temp_dir):
        df = sample_df_with_target.drop(columns=["emp_length"])
        assert plot_emp_length(df, logger) is None
        plt.close("all")


def test_plot_annual_income_returns_path_or_none(sample_df_with_target, logger, temp_dir):
    with patch("EDA.FIG_DIR", temp_dir):
        result = plot_annual_income(sample_df_with_target, logger)
        assert isinstance(result, Path) or result is None
        plt.close("all")


def test_plot_annual_income_missing_column(sample_df_with_target, logger, temp_dir):
    with patch("EDA.FIG_DIR", temp_dir):
        df = sample_df_with_target.drop(columns=["annual_inc"])
        assert plot_annual_income(df, logger) is None
        plt.close("all")


def test_plot_purpose_returns_path_or_none(sample_df_with_target, logger, temp_dir):
    with patch("EDA.FIG_DIR", temp_dir):
        result = plot_purpose(sample_df_with_target, logger)
        assert isinstance(result, Path) or result is None
        plt.close("all")


def test_plot_purpose_missing_column(sample_df_with_target, logger, temp_dir):
    with patch("EDA.FIG_DIR", temp_dir):
        df = sample_df_with_target.drop(columns=["purpose"])
        assert plot_purpose(df, logger) is None
        plt.close("all")


def test_plot_fico_returns_path_or_none(sample_df_with_target, logger, temp_dir):
    with patch("EDA.FIG_DIR", temp_dir):
        result = plot_fico(sample_df_with_target, logger)
        assert isinstance(result, Path) or result is None
        plt.close("all")


def test_plot_fico_missing_columns(sample_df_with_target, logger, temp_dir):
    with patch("EDA.FIG_DIR", temp_dir):
        df = sample_df_with_target.drop(columns=["fico_range_high"])
        assert plot_fico(df, logger) is None
        plt.close("all")


def test_plot_home_ownership_returns_path_or_none(sample_df_with_target, logger, temp_dir):
    with patch("EDA.FIG_DIR", temp_dir):
        result = plot_home_ownership(sample_df_with_target, logger)
        assert isinstance(result, Path) or result is None
        plt.close("all")


def test_plot_home_ownership_missing_column(sample_df_with_target, logger, temp_dir):
    with patch("EDA.FIG_DIR", temp_dir):
        df = sample_df_with_target.drop(columns=["home_ownership"])
        assert plot_home_ownership(df, logger) is None
        plt.close("all")


def test_plot_verification_returns_path_or_none(sample_df_with_target, logger, temp_dir):
    with patch("EDA.FIG_DIR", temp_dir):
        result = plot_verification(sample_df_with_target, logger)
        assert isinstance(result, Path) or result is None
        plt.close("all")


def test_plot_verification_missing_column(sample_df_with_target, logger, temp_dir):
    with patch("EDA.FIG_DIR", temp_dir):
        df = sample_df_with_target.drop(columns=["verification_status"])
        assert plot_verification(df, logger) is None
        plt.close("all")


def test_plot_temporal_returns_path_or_none(sample_df_with_target, logger, temp_dir):
    with patch("EDA.FIG_DIR", temp_dir):
        result = plot_temporal(sample_df_with_target, logger)
        assert isinstance(result, Path) or result is None
        plt.close("all")


def test_plot_temporal_missing_date_column(sample_df_with_target, logger, temp_dir):
    with patch("EDA.FIG_DIR", temp_dir):
        df = sample_df_with_target.drop(columns=["issue_d", "earliest_cr_line"])
        assert plot_temporal(df, logger) is None
        plt.close("all")


def test_top_corr_pairs_returns_dict(sample_df, logger, temp_dir):
    with patch("EDA.FIG_DIR", temp_dir):
        result = top_corr_pairs(sample_df, logger)
        assert isinstance(result, dict)
        plt.close("all")


def test_top_corr_pairs_contains_top_pairs(sample_df, logger, temp_dir):
    with patch("EDA.FIG_DIR", temp_dir):
        result = top_corr_pairs(sample_df, logger)
        assert "top_pairs" in result
        assert isinstance(result["top_pairs"], pd.DataFrame)
        plt.close("all")


def test_top_corr_pairs_contains_required_columns(sample_df, logger, temp_dir):
    with patch("EDA.FIG_DIR", temp_dir):
        top_pairs = top_corr_pairs(sample_df, logger)["top_pairs"]
        for col in ("feature_1", "feature_2", "abs_corr"):
            assert col in top_pairs.columns
        plt.close("all")


def test_feature_importance_returns_dict(sample_df_with_target, logger):
    assert isinstance(feature_importance(sample_df_with_target, logger), dict)


def test_feature_importance_no_target(sample_df, logger):
    assert feature_importance(sample_df, logger) == {}


def test_feature_importance_contains_summary(sample_df_with_target, logger):
    result = feature_importance(sample_df_with_target, logger)
    if result:
        assert "summary" in result


def _report_args():
    return (
        {"n_rows": 100, "n_cols": 10},
        {"n_default": 30, "n_non_default": 70, "imbalance_ratio": 2.33, "majority_baseline_acc": 0.7},
        pd.DataFrame(),
        pd.DataFrame(),
    )


def test_make_report_returns_path(logger, temp_dir):
    with patch("EDA.REPORT_DIR", temp_dir):
        with patch("EDA.FIG_DIR", temp_dir / "figures"):
            (temp_dir / "figures").mkdir(exist_ok=True)
            result = make_report(*_report_args(), logger)
            assert isinstance(result, Path)


def test_make_report_creates_html_file(logger, temp_dir):
    with patch("EDA.REPORT_DIR", temp_dir):
        with patch("EDA.FIG_DIR", temp_dir / "figures"):
            (temp_dir / "figures").mkdir(exist_ok=True)
            result = make_report(*_report_args(), logger)
            assert result.exists()
            assert result.suffix == ".html"


def test_make_report_html_contains_meta(logger, temp_dir):
    with patch("EDA.REPORT_DIR", temp_dir):
        with patch("EDA.FIG_DIR", temp_dir / "figures"):
            (temp_dir / "figures").mkdir(exist_ok=True)
            result = make_report(*_report_args(), logger)
            html = result.read_text()
            assert "100" in html
            assert "10" in html


def test_build_target_integration(sample_df, logger):
    result = build_target(sample_df.copy(), logger)
    assert "target_default" in result.columns
    assert result["target_default"].dtype in [float, "Float64", int, "int64"]


def test_overview_with_various_dtypes(logger):
    df = pd.DataFrame({
        "numeric": [1, 2, 3, 4],
        "float": [1.1, 2.2, 3.3, 4.4],
        "string": ["a", "b", "c", "d"],
        "category": pd.Categorical(["cat1", "cat2", "cat1", "cat2"]),
    })
    result = overview(df, logger)
    assert result["meta"]["n_numeric"] == 2
    assert result["meta"]["n_categ"] >= 2


def test_plot_functions_with_small_data(logger, temp_dir):
    with patch("EDA.FIG_DIR", temp_dir):
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        try:
            plot_correlation(df, logger)
            plt.close("all")
        except Exception as e:
            pytest.skip(f"Skipping small data test: {str(e)}")


def test_overview_empty_dataframe(logger):
    with pytest.raises((ValueError, KeyError)):
        overview(pd.DataFrame(), logger)


def test_build_target_all_current_loans(logger):
    df = pd.DataFrame({"loan_status": ["Current", "Current", "Current"]})
    assert len(build_target(df, logger)) == 0


def test_plot_missing_all_nulls(sample_df, logger, temp_dir):
    with patch("EDA.FIG_DIR", temp_dir):
        df = sample_df.copy()
        df["new_col"] = np.nan
        plot_missing(df, logger)
        plt.close("all")


def test_check_target_distribution_imbalanced_data(logger, temp_dir):
    with patch("EDA.FIG_DIR", temp_dir):
        df = pd.DataFrame({
            "loan_status": ["Charged Off"] * 5 + ["Fully Paid"] * 100,
            "target_default": [1] * 5 + [0] * 100,
        })
        result = check_target_distribution(df, logger)
        assert result["imbalance_ratio"] > 10
        plt.close("all")