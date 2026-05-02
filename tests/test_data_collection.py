from datetime import datetime

import pandas as pd
import pytest

from DataCollection import DataCollectionPipeline


class DummyResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise ValueError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


class DummySession:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def get(self, target_url, params=None, timeout=None):
        self.calls.append({"target_url": target_url, "params": params, "timeout": timeout})
        return self.response


@pytest.fixture()
def pipeline(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
    return DataCollectionPipeline()


def test_collect_from_dataset_returns_requested_columns_and_parses_dates(pipeline, tmp_path):
    csv_path = tmp_path / "dataset.csv"
    source = pd.DataFrame(
        {
            "id": [1, 2],
            "issue_d": ["2020-01-01", "invalid-date"],
            "loan_amnt": [1000, 2000],
            "extra": ["x", "y"],
        }
    )
    source.to_csv(csv_path, index=False)

    result = pipeline.collect_from_dataset(["id", "issue_d", "loan_amnt"], str(csv_path))

    assert list(result.columns) == ["id", "issue_d", "loan_amnt"]
    assert pd.api.types.is_datetime64_any_dtype(result["issue_d"])
    assert result.loc[0, "issue_d"] == datetime(2020, 1, 1)
    assert pd.isna(result.loc[1, "issue_d"])


def test_collect_from_dataset_returns_empty_when_file_missing(pipeline):
    result = pipeline.collect_from_dataset(["id"], "missing.csv")

    assert result.empty


def test_collect_from_dataset_returns_empty_when_no_requested_columns_exist(pipeline, tmp_path):
    csv_path = tmp_path / "dataset.csv"
    pd.DataFrame({"other": [1, 2]}).to_csv(csv_path, index=False)

    result = pipeline.collect_from_dataset(["id", "loan_amnt"], str(csv_path))

    assert result.empty


def test_collect_api_query_parses_observations_and_uses_column_name(pipeline):
    response = DummyResponse(
        {
            "observations": [
                {"date": "2020-01-01", "value": "1.5"},
                {"date": None, "value": "2.0"},
                {"date": "2020-02-01", "value": "1.7"},
            ]
        }
    )
    pipeline.session = DummySession(response)

    records = pipeline._collect_api_query(
        "https://example.com/api",
        {"series_id": "TEST_SERIES", "column_name": "My Metric"},
    )

    assert records == [
        {"Year": 2020, "Month": 1, "My Metric": "1.5"},
        {"Year": 2020, "Month": 2, "My Metric": "1.7"},
    ]


def test_collect_api_query_returns_empty_on_http_error(pipeline):
    response = DummyResponse({}, status_code=500)
    pipeline.session = DummySession(response)

    records = pipeline._collect_api_query("https://example.com/api", {"series_id": "TEST_SERIES"})

    assert records == []


def test_collect_from_api_injects_api_key_and_file_type(pipeline, monkeypatch):
    captured = {}

    def fake_query(target_url, params):
        captured["target_url"] = target_url
        captured["params"] = params.copy()
        return [{"Year": 2020, "Month": 1, "CPI": "1.0"}]

    monkeypatch.setattr(pipeline, "_collect_api_query", fake_query)
    monkeypatch.setenv("API_KEY", "secret-key")

    result = pipeline.collect_from_api(
        "https://example.com/api",
        {"series_id": "CPIAUCSL", "column_name": "CPI"},
    )

    assert captured["target_url"] == "https://example.com/api"
    assert captured["params"]["api_key"] == "secret-key"
    assert captured["params"]["file_type"] == "json"
    assert list(result.columns) == ["Year", "Month", "CPI"]
    assert result.loc[0, "CPI"] == "1.0"


def test_merge_indicators_merges_non_empty_frames(pipeline):
    df1 = pd.DataFrame({"Year": [2020], "Month": [1], "CPI": [1.0]})
    df2 = pd.DataFrame({"Year": [2020], "Month": [1], "Unemployment Rate": [5.0]})
    df3 = pd.DataFrame()

    result = pipeline.merge_indicators([df1, df2, df3])

    assert list(result.columns) == ["Year", "Month", "CPI", "Unemployment Rate"]
    assert result.loc[0, "CPI"] == 1.0
    assert result.loc[0, "Unemployment Rate"] == 5.0


def test_merge_indicators_returns_empty_when_all_empty(pipeline):
    result = pipeline.merge_indicators([pd.DataFrame(), pd.DataFrame()])

    assert result.empty


def test_merge_indicators_with_loans_merges_on_issue_date(pipeline):
    loan_df = pd.DataFrame(
        {
            "id": [1, 2],
            "issue_d": pd.to_datetime(["2020-01-15", "2020-02-20"]),
            "loan_amnt": [1000, 2000],
        }
    )
    indicators_df = pd.DataFrame(
        {
            "Year": [2020, 2020],
            "Month": [1, 2],
            "CPI": [1.5, 1.7],
        }
    )

    result = pipeline.merge_indicators_with_loans(loan_df, indicators_df)

    assert "issue_year" not in result.columns
    assert "issue_month" not in result.columns
    assert "Year" not in result.columns
    assert "Month" not in result.columns
    assert result.loc[0, "CPI"] == 1.5
    assert result.loc[1, "CPI"] == 1.7


def test_merge_indicators_with_loans_returns_indicators_when_loan_df_empty(pipeline):
    loan_df = pd.DataFrame()
    indicators_df = pd.DataFrame({"Year": [2020], "Month": [1], "CPI": [1.5]})

    result = pipeline.merge_indicators_with_loans(loan_df, indicators_df)

    assert result.equals(indicators_df)


def test_merge_indicators_with_loans_returns_loan_df_when_indicators_empty(pipeline):
    loan_df = pd.DataFrame({"id": [1], "issue_d": pd.to_datetime(["2020-01-01"]), "loan_amnt": [1000]})
    indicators_df = pd.DataFrame()

    result = pipeline.merge_indicators_with_loans(loan_df, indicators_df)

    assert result.equals(loan_df)
