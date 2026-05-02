import logging

import pandas as pd
import pytest

from DataTransformation import (
    create_fico_feature,
    encode_emp_length,
    encode_pymnt_plan,
    encode_sub_grade,
    encode_term,
    filter_loan_status,
    load_input_data,
    normalize_home_ownership,
    normalize_train_test,
    one_hot_encode_columns,
    transform_data,
)


def make_logger():
    logger = logging.getLogger("DataTransformationTest")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    logger.setLevel(logging.INFO)
    return logger


def test_load_input_data_uses_read_csv(monkeypatch):
    expected = pd.DataFrame({"loan_status": ["Fully Paid"]})
    monkeypatch.setattr(pd, "read_csv", lambda path: expected)

    result = load_input_data()

    assert result.equals(expected)


def test_filter_loan_status_removes_current_and_default_and_encodes_good_status():
    df = pd.DataFrame(
        {
            "loan_status": [
                "Fully Paid",
                "Current",
                "Default",
                "In Grace Period",
                "Charged Off",
            ]
        }
    )

    result = filter_loan_status(df)

    assert result["loan_status"].tolist() == [1, 1, 0]


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


def test_encode_sub_grade_and_emp_length():
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


def test_transform_data_runs_end_to_end(monkeypatch):
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
        }
    )
    monkeypatch.setattr(pd, "read_csv", lambda *args, **kwargs: df_in)

    logger = make_logger()
    result = transform_data(logger)

    assert "loan_status" in result.columns
    assert result["loan_status"].isin([0, 1]).all()
    assert "fico" in result.columns
    assert any(column.startswith("home_ownership_") for column in result.columns)
    assert any(column.startswith("verification_status_") for column in result.columns)
    assert any(column.startswith("purpose_") for column in result.columns)
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
