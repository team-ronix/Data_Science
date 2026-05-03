import logging
import pandas as pd
import pytest

import src.DataTransformation as dt_module
from src.DataTransformation import (
    create_fico_feature,
    encode_emp_length,
    encode_pymnt_plan,
    encode_sub_grade,
    encode_term,
    filter_loan_status,
    load_input_data,
    log_step,
    normalize_home_ownership,
    normalize_train_test,
    one_hot_encode_columns,
    save_business_statistics,
    save_transformation_outputs,
    setup_logging,
    split_transformed_data,
    transform_data,
)


def make_logger():
    logger = logging.getLogger("DataTransformationTest")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    logger.setLevel(logging.INFO)
    return logger


def _make_transformed_df(n_good: int = 60, n_bad: int = 40) -> pd.DataFrame:
    rows = n_good + n_bad
    return pd.DataFrame(
        {
            "loan_status": [0] * n_good + [1] * n_bad,
            "loan_amnt": [1000.0] * rows,
            "fico": [700.0] * rows,
        }
    )


def test_load_input_data_uses_read_csv(monkeypatch):
    expected = pd.DataFrame({"loan_status": ["Fully Paid"]})
    monkeypatch.setattr(pd, "read_csv", lambda path: expected)

    result = load_input_data()

    assert result.equals(expected)


def test_load_input_data_accepts_custom_path(tmp_path):
    csv = tmp_path / "custom.csv"
    pd.DataFrame({"x": [1, 2]}).to_csv(csv, index=False)

    result = load_input_data(path=csv)

    assert list(result.columns) == ["x"]
    assert len(result) == 2


def test_filter_loan_status_removes_current_and_default():
    df = pd.DataFrame(
        {"loan_status": ["Fully Paid", "Current", "Default", "In Grace Period", "Charged Off"]}
    )

    result = filter_loan_status(df)

    assert len(result) == 3
    assert "Current" not in result["loan_status"].values
    assert "Default" not in result["loan_status"].values


def test_filter_loan_status_encodes_good_as_0_bad_as_1():
    df = pd.DataFrame({"loan_status": ["Fully Paid", "In Grace Period", "Charged Off"]})

    result = filter_loan_status(df)

    assert result["loan_status"].tolist() == [0, 0, 1]


def test_filter_loan_status_includes_policy_exception_as_good():
    df = pd.DataFrame(
        {"loan_status": ["Does not meet the credit policy. Status:Fully Paid", "Charged Off"]}
    )

    result = filter_loan_status(df)

    assert result["loan_status"].tolist() == [0, 1]


def test_create_fico_feature_builds_average_and_drops_unused_columns():
    df = pd.DataFrame(
        {
            "fico_range_high": [700, 720],
            "fico_range_low": [680, 700],
            "id": [1, 2],
            "last_pymnt_d": ["2020-01-01", "2020-02-01"],
        }
    )

    result = create_fico_feature(df)

    assert result["fico"].tolist() == [690.0, 710.0]
    assert "id" not in result.columns
    assert "last_pymnt_d" not in result.columns
    assert "fico_range_low" not in result.columns
    assert "fico_range_high" not in result.columns


def test_encode_term_and_pymnt_plan():
    df = pd.DataFrame({"term": ["36 months", "60 months"], "pymnt_plan": ["y", "n"]})

    result = encode_pymnt_plan(encode_term(df))

    assert result["term"].tolist() == [1, 0]
    assert result["pymnt_plan"].tolist() == [1, 0]


def test_encode_sub_grade_produces_integer_column():
    df = pd.DataFrame({"sub_grade": ["B2", "A1", "C3"]})

    result = encode_sub_grade(df)

    assert pd.api.types.is_integer_dtype(result["sub_grade"])


def test_encode_emp_length_standard_years():
    df = pd.DataFrame({"emp_length": ["5 years", "3 years"]})

    result = encode_emp_length(df)

    assert result["emp_length"].tolist() == [5, 3]


def test_encode_emp_length_special_values():
    df = pd.DataFrame({"emp_length": ["10+ years", "< 1 year", "Unknown"]})

    result = encode_emp_length(df)

    assert result["emp_length"].tolist() == [10, 0, -1]


def test_encode_sub_grade_and_emp_length_combined():
    df = pd.DataFrame(
        {
            "sub_grade": ["B2", "A1", "C3"],
            "emp_length": ["10+ years", "< 1 year", "5 years"],
        }
    )

    result = encode_emp_length(encode_sub_grade(df))

    assert pd.api.types.is_integer_dtype(result["sub_grade"])
    assert result["emp_length"].tolist() == [10, 0, 5]


def test_normalize_home_ownership_and_one_hot_encode_columns():
    df = pd.DataFrame(
        {
            "home_ownership": ["RENT", "NONE", "OWN"],
            "verification_status": ["Verified", "Not Verified", "Source Verified"],
        }
    )

    result = normalize_home_ownership(df)
    result = one_hot_encode_columns(result, ["home_ownership", "verification_status"])

    assert "home_ownership_OTHER" in result.columns
    assert "verification_status_Verified" in result.columns
    assert "verification_status_Source Verified" in result.columns


def test_normalize_home_ownership_keeps_standard_values():
    df = pd.DataFrame({"home_ownership": ["RENT", "MORTGAGE", "OWN", "NONE", "ANY"]})
    result = normalize_home_ownership(df)
    assert result["home_ownership"].tolist() == ["RENT", "MORTGAGE", "OWN", "OTHER", "OTHER"]


def test_one_hot_encode_columns_no_op_when_columns_absent():
    df = pd.DataFrame({"a": [1, 2]})

    result = one_hot_encode_columns(df, ["nonexistent"])

    assert list(result.columns) == ["a"]


def test_one_hot_encode_columns_with_logger_does_not_raise():
    df = pd.DataFrame({"cat": ["x", "y"]})
    logger = make_logger()

    result = one_hot_encode_columns(df, ["cat"], logger=logger)

    assert "cat_x" in result.columns
    assert "cat_y" in result.columns


def test_transform_data_runs_end_to_end(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(dt_module, "BASE_DIR", tmp_path)

    df_in = pd.DataFrame(
        {
            "loan_status": ["Fully Paid", "Current", "In Grace Period"],
            "fico_range_high": [700, 710, 720],
            "fico_range_low": [680, 700, 710],
            "term": ["36 months", "60 months", "36 months"],
            "sub_grade": ["A1", "B2", "A1"],
            "emp_length": ["10+ years", "< 1 year", "5 years"],
            "pymnt_plan": ["y", "n", "y"],
            "home_ownership": ["RENT", "NONE", "OWN"],
            "verification_status": ["Verified", "Source Verified", "Not Verified"],
            "purpose": ["debt_consolidation", "credit_card", "home_improvement"],
            "total_pymnt": [1200.0, 1100.0, 1300.0],
            "loan_amnt": [1000.0, 1000.0, 1000.0],
        }
    )
    monkeypatch.setattr(pd, "read_csv", lambda *args, **kwargs: df_in)

    logger = make_logger()
    result = transform_data(logger)

    assert "loan_status" in result.columns
    assert result["loan_status"].isin([0, 1]).all()
    assert "fico" in result.columns
    assert any(col.startswith("home_ownership_") for col in result.columns)
    assert any(col.startswith("verification_status_") for col in result.columns)
    assert any(col.startswith("purpose_") for col in result.columns)
    assert pd.api.types.is_integer_dtype(result["term"])
    assert pd.api.types.is_integer_dtype(result["sub_grade"])


def test_normalize_train_test_scales_using_train_stats():
    train = pd.DataFrame({"loan_amnt": [10.0, 30.0], "fico": [50.0, 150.0]})
    test = pd.DataFrame({"loan_amnt": [20.0], "fico": [100.0]})

    train_norm, test_norm = normalize_train_test(train, test)

    assert pytest.approx(train_norm.loc[0, "loan_amnt"]) == 0.0
    assert pytest.approx(train_norm.loc[1, "loan_amnt"]) == 1.0
    assert pytest.approx(test_norm.loc[0, "loan_amnt"]) == 0.5


def test_normalize_train_test_handles_zero_range():
    train = pd.DataFrame({"loan_amnt": [100.0, 100.0], "fico": [50.0, 50.0]})
    test = pd.DataFrame({"loan_amnt": [100.0], "fico": [50.0]})

    train_norm, test_norm = normalize_train_test(train, test)

    assert all(train_norm["loan_amnt"] == 0)
    assert all(train_norm["fico"] == 0)
    assert all(test_norm["loan_amnt"] == 0)
    assert all(test_norm["fico"] == 0)


def test_normalize_train_test_ignores_loan_status_column():
    train = pd.DataFrame({"loan_amnt": [0.0, 100.0], "loan_status": [0, 1]})
    test = pd.DataFrame({"loan_amnt": [50.0], "loan_status": [1]})

    train_norm, test_norm = normalize_train_test(train, test)

    assert train_norm["loan_status"].tolist() == [0, 1]
    assert test_norm["loan_status"].tolist() == [1]


def test_setup_logging_returns_logger(tmp_path, monkeypatch):
    monkeypatch.setattr(dt_module, "LOG_PATH", tmp_path / "test.log")
    logger = setup_logging()

    assert isinstance(logger, logging.Logger)
    assert logger.name == "DataTransformationPipeline"


def test_log_step_logs_dataframe_stats():
    logger = make_logger()
    df = pd.DataFrame({"a": [1, 2, None], "b": [4, 5, 6]})
    log_step(logger, "Test Step", df)


def test_save_business_statistics_creates_csv_with_correct_columns(tmp_path, monkeypatch):
    monkeypatch.setattr(dt_module, "BASE_DIR", tmp_path)
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)

    data = pd.DataFrame(
        {
            "loan_status": [0, 0, 1, 1],
            "total_pymnt": [1200.0, 1400.0, 100.0, 200.0],
            "loan_amnt": [1000.0, 1000.0, 1000.0, 1000.0],
        }
    )

    csv_path = tmp_path / "data" / "business_statistics.csv"
    save_business_statistics(make_logger(), data, csv_path)
    assert csv_path.exists()
    stats_df = pd.read_csv(csv_path)
    assert set(stats_df.columns) == {"avg_loan_profit", "avg_loan_loss", "avg_loan_amount"}


def test_save_business_statistics_calculates_correct_values(tmp_path, monkeypatch):
    monkeypatch.setattr(dt_module, "BASE_DIR", tmp_path)
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)

    data = pd.DataFrame(
        {
            "loan_status": [0, 0, 1, 1],
            "total_pymnt": [1200.0, 1400.0, 100.0, 200.0],
            "loan_amnt": [1000.0, 1000.0, 1000.0, 1000.0],
        }
    )
    saved_path = tmp_path / "data" / "business_statistics.csv"
    save_business_statistics(make_logger(), data, saved_path)
    stats_df = pd.read_csv(saved_path)
    # avg_loan_profit = mean(1200,1400) - mean(1000,1000) = 300
    assert pytest.approx(stats_df["avg_loan_profit"].iloc[0], rel=0.01) == 300.0
    # avg_loan_loss = mean(1000,1000) - mean(100,200) = 850
    assert pytest.approx(stats_df["avg_loan_loss"].iloc[0], rel=0.01) == 850.0
    # avg_loan_amount = mean of all loan_amnt = 1000
    assert pytest.approx(stats_df["avg_loan_amount"].iloc[0], rel=0.01) == 1000.0



def test_split_transformed_data_creates_stratified_split():
    df = pd.DataFrame(
        {
            "loan_status": [0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
            "feature1": range(10),
        }
    )

    train, test = split_transformed_data(df, test_size=0.2, random_state=42)

    assert len(train) + len(test) == 10
    assert len(test) == 2
    assert "loan_status" in train.columns
    assert "feature1" in train.columns


def test_split_transformed_data_stratification_maintains_class_ratio():
    df = pd.DataFrame(
        {
            "loan_status": [0] * 80 + [1] * 20,
            "feature1": range(100),
        }
    )

    train, test = split_transformed_data(df, test_size=0.2, random_state=42)
    class_ratio_train = (train["loan_status"] == 0).sum() / len(train)
    class_ratio_test = (test["loan_status"] == 0).sum() / len(test)
    assert 0.75 < class_ratio_train < 0.85
    assert 0.75 < class_ratio_test < 0.85


def test_save_transformation_outputs_writes_all_csv_files(tmp_path, monkeypatch):
    output_path = tmp_path / "transformed.csv"
    train_output_path = tmp_path / "train.csv"
    train_norm_output_path = tmp_path / "train_norm.csv"
    test_output_path = tmp_path / "test.csv"

    df = _make_transformed_df()
    train_norm, test_norm = save_transformation_outputs(
        make_logger(), 
        df,
        output_path=output_path,
        train_output_path=train_output_path,
        train_norm_output_path=train_norm_output_path,
        test_output_path=test_output_path,
    )

    assert (tmp_path / "transformed.csv").exists()
    assert (tmp_path / "train.csv").exists()
    assert (tmp_path / "train_norm.csv").exists()
    assert (tmp_path / "test.csv").exists()
    assert "loan_status" in train_norm.columns
    assert "loan_status" in test_norm.columns


def test_save_transformation_outputs_returns_normalized_splits(tmp_path, monkeypatch):
    output_path = tmp_path / "transformed.csv"
    train_output_path = tmp_path / "train.csv"
    train_norm_output_path = tmp_path / "train_norm.csv"
    test_output_path = tmp_path / "test.csv"

    df = _make_transformed_df()
    train_norm, test_norm = save_transformation_outputs(
        make_logger(), 
        df, 
        output_path=output_path, 
        train_output_path=train_output_path, 
        train_norm_output_path=train_norm_output_path, 
        test_output_path=test_output_path
    )

    assert len(train_norm) + len(test_norm) == len(df)
    # loan_amnt was constant (1000), zero-range → normalized to 0
    assert (train_norm["loan_amnt"] == 0).all()
    assert (test_norm["loan_amnt"] == 0).all()