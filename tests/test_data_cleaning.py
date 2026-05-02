import pandas as pd
import pytest

from DataCleaning import (
    DataCleaningPipeline,
    clip_outliers,
    coerce_numeric_columns,
    finalize_types,
    handle_missing_values,
    normalize_columns,
    parse_dates,
    remove_duplicates,
    repair_logical_relationships,
    strip_and_standardize_strings,
)


def test_normalize_columns_trims_whitespace():
    df = pd.DataFrame({"  id  ": [1], " term ": ["36 months"]})

    result = normalize_columns(df)

    assert list(result.columns) == ["id", "term"]


def test_strip_and_standardize_strings_handles_null_sentinels_and_spaces():
    df = pd.DataFrame(
        {
            "term": [" 36   months ", "", "nan", "None", "ok"],
            "sub_grade": [" A1 ", "B2", "  C3  ", "null", "D4"],
        }
    )

    result = strip_and_standardize_strings(df)

    assert result.loc[0, "term"] == "36 months"
    assert pd.isna(result.loc[1, "term"])
    assert pd.isna(result.loc[2, "term"])
    assert pd.isna(result.loc[3, "term"])
    assert result.loc[4, "term"] == "ok"
    assert result.loc[0, "sub_grade"] == "A1"
    assert pd.isna(result.loc[3, "sub_grade"])


def test_parse_dates_coerces_invalid_values():
    df = pd.DataFrame(
        {
            "issue_d": ["2020-01-01", "not-a-date"],
            "earliest_cr_line": ["2015-05-01", "2018-01-10"],
        }
    )

    result = parse_dates(df, date_columns=["issue_d", "earliest_cr_line", "missing_col"])

    assert pd.api.types.is_datetime64_any_dtype(result["issue_d"])
    assert pd.api.types.is_datetime64_any_dtype(result["earliest_cr_line"])
    assert pd.isna(result.loc[1, "issue_d"])


def test_coerce_numeric_columns_strips_noise_and_percent_signs():
    df = pd.DataFrame(
        {
            "loan_amnt": ["$1,200", "300"],
            "int_rate": ["12.5%", "7%"],
            "dti": ["10.2", "bad"],
        }
    )

    result = coerce_numeric_columns(
        df,
        numeric_candidates={"loan_amnt", "int_rate", "dti"},
        percent_columns={"int_rate"},
    )

    assert result["loan_amnt"].tolist() == [1200.0, 300.0]
    assert result["int_rate"].tolist() == [12.5, 7.0]
    assert result.loc[0, "dti"] == 10.2
    assert pd.isna(result.loc[1, "dti"])


def test_repair_logical_relationships_swaps_fico_and_cleans_values():
    df = pd.DataFrame(
        {
            "fico_range_low": [750, 650],
            "fico_range_high": [700, 680],
            "loan_amnt": [-10, 100],
            "pymnt_plan": [" Y ", "n"],
        }
    )

    result = repair_logical_relationships(df, non_negative_columns=["loan_amnt"])

    assert result.loc[0, "fico_range_low"] == 700
    assert result.loc[0, "fico_range_high"] == 750
    assert pd.isna(result.loc[0, "loan_amnt"])
    assert result.loc[0, "pymnt_plan"] == "y"


def test_handle_missing_values_drops_mandatory_and_fills_expected_fields():
    df = pd.DataFrame(
        {
            "id": [1, 2, None],
            "issue_d": pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01"]),
            "CPI": [None, 250.0, None],
            "delinq_2yrs": [None, 1, None],
            "loan_amnt": [None, 200.0, None],
            "term": [None, "36 months", None],
        }
    )

    result = handle_missing_values(
        df,
        mandatory_cols=["id", "issue_d"],
        indicator_columns=["CPI"],
        string_columns=["term"],
        positive_amount_columns=["loan_amnt"],
        rate_columns=[],
    )

    assert len(result) == 2
    assert set(result["id"].tolist()) == {1, 2}
    assert result["CPI"].tolist() == [250.0, 250.0]
    assert result["delinq_2yrs"].tolist() == [0.0, 1.0]
    assert result["loan_amnt"].tolist() == [200.0, 200.0]
    assert result["term"].tolist()[0] == "Unknown"


def test_remove_duplicates_prefers_first_id_and_row():
    df = pd.DataFrame(
        {
            "id": [1, 1, 2, 2],
            "loan_amnt": [100, 200, 300, 300],
        }
    )

    result = remove_duplicates(df)

    assert len(result) == 2
    assert result["id"].tolist() == [1, 2]
    assert result["loan_amnt"].tolist() == [100, 300]


def test_clip_outliers_caps_extreme_values():
    df = pd.DataFrame({"loan_amnt": [10, 11, 12, 1000]})

    result = clip_outliers(df, columns=["loan_amnt"])

    assert result["loan_amnt"].max() < 1000
    assert result["loan_amnt"].min() == 10


def test_finalize_types_converts_expected_columns_to_nullable_int():
    df = pd.DataFrame(
        {
            "id": [1.0, 2.0],
            "delinq_2yrs": [0.0, 1.0],
            "fico_range_low": [700.0, 710.0],
            "fico_range_high": [750.0, 760.0],
        }
    )

    result = finalize_types(df, count_columns=["delinq_2yrs"])

    assert str(result["id"].dtype) == "Int64"
    assert str(result["delinq_2yrs"].dtype) == "Int64"
    assert str(result["fico_range_low"].dtype) == "Int64"
    assert str(result["fico_range_high"].dtype) == "Int64"


def test_pipeline_execute_runs_steps_and_builds_report(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "log").mkdir(parents=True, exist_ok=True)

    def add_one_column(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["b"] = 1
        return out

    pipeline = DataCleaningPipeline(name="unit-test-pipeline")
    pipeline.add_step("Add Column", add_one_column)

    input_df = pd.DataFrame({"a": [1, 2]})
    result = pipeline.execute(input_df, verbose=False)

    assert "b" in result.columns
    report = pipeline.get_execution_report()
    assert len(report) == 1
    assert report.loc[0, "step"] == "Add Column"
    assert report.loc[0, "status"] == "success"


def test_pipeline_execute_raises_and_logs_failure(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "log").mkdir(parents=True, exist_ok=True)

    def fail_step(df: pd.DataFrame) -> pd.DataFrame:
        raise ValueError("boom")

    pipeline = DataCleaningPipeline(name="unit-test-failure")
    pipeline.add_step("Failing Step", fail_step)

    with pytest.raises(ValueError, match="boom"):
        pipeline.execute(pd.DataFrame({"a": [1]}), verbose=False)

    report = pipeline.get_execution_report()
    assert len(report) == 1
    assert report.loc[0, "step"] == "Failing Step"
    assert report.loc[0, "status"] == "failed"


    
