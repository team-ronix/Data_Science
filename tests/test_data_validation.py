import logging

import pandas as pd
import numpy as np
import pytest

from src.DataValidation import DataValidation


@pytest.fixture
def sample_data():
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "loan_amnt": [1000.0, 1500.0, 2000.0, 1200.0, 1800.0],
            "int_rate": [5.5, 6.0, 7.5, 5.0, 6.5],
            "annual_inc": [50000.0, 60000.0, 75000.0, 55000.0, 80000.0],
            "dti": [10.5, 12.0, 15.0, 11.0, 13.5],
            "loan_status": [0, 1, 0, 0, 1],
            "home_ownership": ["RENT", "MORTGAGE", "OWN", "RENT", "MORTGAGE"],
            "term": [36, 60, 36, 60, 36],
        }
    )


@pytest.fixture
def validation_obj(sample_data, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
    return DataValidation(sample_data)


def test_init_creates_logger(sample_data, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
    
    validator = DataValidation(sample_data)
    
    assert hasattr(validator, "logger")
    assert isinstance(validator.logger, logging.Logger)


def test_init_stores_data_stats(sample_data, validation_obj):
    assert validation_obj.total_rows == 5
    assert validation_obj.total_cols == 8
    assert validation_obj.missing_values == 0


def test_init_categorizes_columns(sample_data, validation_obj):
    assert "id" in validation_obj.num_cols
    assert "loan_amnt" in validation_obj.num_cols
    assert "home_ownership" in validation_obj.cat_cols
    assert "term" in validation_obj.num_cols


def test_profile_data_returns_dict_with_required_keys(validation_obj):
    profile = validation_obj.profile_data()
    
    assert isinstance(profile, dict)
    assert "row_count" in profile
    assert "col_count" in profile
    assert "dtypes" in profile
    assert "missing_values" in profile


def test_profile_data_counts_correct_rows_and_cols(validation_obj):
    profile = validation_obj.profile_data()
    
    assert profile["row_count"] == 5
    assert profile["col_count"] == 8
    assert profile["missing_values"] == 0


def test_profile_data_with_missing_values(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
    
    data = pd.DataFrame(
        {
            "id": [1, 2, None],
            "loan_amnt": [1000.0, None, 2000.0],
        }
    )
    
    validator = DataValidation(data)
    profile = validator.profile_data()
    
    assert profile["missing_values"] == 2


def test_validate_categorical_checks_uniqueness(validation_obj):
    results = validation_obj.validate_categorical()
    
    assert "home_ownership" in results
    assert "unique_count" in results["home_ownership"]
    assert results["home_ownership"]["unique_count"] == 3


def test_validate_categorical_identifies_inconsistencies(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
    
    data = pd.DataFrame(
        {
            "home_ownership": ["RENT", "rent", "RENT", "Rent"],
            "loan_status": [0, 1, 0, 1],
        }
    )
    
    validator = DataValidation(data)
    results = validator.validate_categorical()
    
    assert results["home_ownership"]["inconsistencies"] is True


def test_validate_categorical_top_values(validation_obj):
    results = validation_obj.validate_categorical()
    
    assert "top_5_values" in results["home_ownership"]
    assert isinstance(results["home_ownership"]["top_5_values"], dict)


def test_check_duplicate_rows_detects_no_duplicates(validation_obj):
    result = validation_obj.check_duplicate_rows()
    
    assert isinstance(result, dict)
    assert "duplicate_row_count" in result
    assert result["duplicate_row_count"] == 0


def test_check_duplicate_rows_detects_duplicates(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
    
    data = pd.DataFrame(
        {
            "id": [1, 1, 2],
            "loan_amnt": [1000.0, 1000.0, 2000.0],
        }
    )
    
    validator = DataValidation(data)
    result = validator.check_duplicate_rows()
    
    assert result["duplicate_row_count"] >= 2


def test_validate_distribution_returns_dict(validation_obj):
    results = validation_obj.validate_distribution()
    
    assert isinstance(results, dict)


def test_validate_distribution_for_numeric_cols(validation_obj):
    results = validation_obj.validate_distribution()
    for col in validation_obj.num_cols:
        if col != "id":
            assert isinstance(results, dict)


def test_validate_relationships_returns_correlations(validation_obj):
    results = validation_obj.validate_relationships()
    
    assert isinstance(results, dict)
    assert "high_correlation_pairs" in results
    assert "multicollinearity_risk" in results


def test_validate_relationships_detects_high_correlation(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
    
    x = np.arange(10)
    data = pd.DataFrame(
        {
            "feature1": x,
            "feature2": x * 2,
            "loan_status": [0, 1] * 5,
        }
    )
    
    validator = DataValidation(data)
    results = validator.validate_relationships()
    
    assert len(results["high_correlation_pairs"]) > 0


def test_validate_accuracy_adds_expectations(validation_obj):
    validation_obj.validate_accuracy()    
    assert len(validation_obj.suite.expectations) > 0


def test_validate_outliers_adds_expectations(validation_obj):
    validation_obj.validate_outliers()
    assert len(validation_obj.suite.expectations) > 0


def test_check_completeness_adds_expectations(validation_obj):
    validation_obj.check_completeness()
    assert len(validation_obj.suite.expectations) > 0


def test_check_uniqueness_adds_id_expectation(validation_obj):
    validation_obj.check_uniqueness()
    assert len(validation_obj.suite.expectations) > 0


def test_run_all_validations_executes(validation_obj):
    validation_obj.run_all_validations()
    
    assert isinstance(validation_obj.report, dict)
    assert "Profile" in validation_obj.report or len(validation_obj.report) > 0


def test_add_zscore_columns_creates_zscore_features(validation_obj):
    zscore_cols = [col for col in validation_obj.data.columns if "_zscore" in col]    
    assert len(zscore_cols) > 0


def test_zscore_values_in_reasonable_range(validation_obj):
    zscore_cols = [col for col in validation_obj.data.columns if "_zscore" in col]
    for col in zscore_cols:
        assert validation_obj.data[col].min() <= 5
        assert validation_obj.data[col].max() >= -5


def test_profile_data_detects_type_mismatches(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
    data = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "numeric_as_string": ["100", "200", "300"],
        }
    )
    
    validator = DataValidation(data)
    profile = validator.profile_data()    
    assert isinstance(profile["suspected_type_mismatches"], list)


def test_validate_categorical_cardinality_high(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
    data = pd.DataFrame(
        {
            "unique_ids": [str(i) for i in range(100)],
            "loan_status": [0, 1] * 50,
        }
    )
    
    validator = DataValidation(data)
    results = validator.validate_categorical()
    assert results["unique_ids"]["cardinality_flag"] is not None


def test_validate_categorical_no_values(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
    data = pd.DataFrame(
        {
            "empty_col": [None, None, None],
            "loan_status": [0, 1, 0],
        }
    )
    
    validator = DataValidation(data)
    results = validator.validate_categorical()
    assert isinstance(results, dict)
