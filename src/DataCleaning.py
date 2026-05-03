from __future__ import annotations
import argparse
import logging
import pickle
from pathlib import Path
from typing import Iterable

import pandas as pd
import os

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_PATH = BASE_DIR / "data/merged_df.csv"
DEFAULT_OUTPUT_PATH = BASE_DIR / "data/merged_df_cleaned.csv"

DATE_COLUMNS = ["issue_d", "earliest_cr_line", "last_pymnt_d"]

STRING_COLUMNS = [
    "term", "sub_grade", "emp_title", "emp_length", "home_ownership",
    "verification_status", "loan_status", "pymnt_plan", "purpose",
    "initial_list_status",
]

COUNT_COLUMNS = [
    "delinq_2yrs", "inq_last_6mths", "mths_since_last_delinq",
    "open_acc", "pub_rec", "total_acc", "acc_now_delinq",
    "mort_acc", "pub_rec_bankruptcies",
]

POSITIVE_AMOUNT_COLUMNS = [
    "loan_amnt", "funded_amnt", "funded_amnt_inv", "installment",
    "annual_inc", "revol_bal", "total_pymnt", "recoveries", "avg_cur_bal",
]

RATE_COLUMNS = ["int_rate", "revol_util", "dti"]

INDICATOR_COLUMNS = ["CPI", "Unemployment Rate", "Federal Funds Rate"]

_NON_NEGATIVE_COLUMNS = POSITIVE_AMOUNT_COLUMNS + RATE_COLUMNS + COUNT_COLUMNS

_NUMERIC_CANDIDATES = {
    "fico_range_low", "fico_range_high",
    *COUNT_COLUMNS, *POSITIVE_AMOUNT_COLUMNS, *RATE_COLUMNS, *INDICATOR_COLUMNS,
}

_PERCENT_COLUMNS = {"int_rate", "revol_util"}


class DataCleaningPipeline:
    def __init__(self, name: str = "Data Cleaning Pipeline") -> None:
        self.name = name
        self.steps: list[dict] = []
        self.execution_log: list[dict] = []
        self._setup_logging()
        self.logger.info(f"Pipeline '{self.name}' initialized.")

    def _setup_logging(self) -> None:
        os.makedirs("logs", exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler("./logs/DataCleaningPipeline.log"),
                logging.StreamHandler(),
            ],
        )
        self.logger = logging.getLogger("DataCleaningPipeline")

    def add_step(self, name: str, function, **kwargs) -> "DataCleaningPipeline":
        self.steps.append({"name": name, "function": function, "kwargs": kwargs})
        return self

    def execute(self, df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
        result = df.copy()
        self.execution_log = []

        for step in self.steps:
            if verbose:
                print(f"{step['name']}...")
            self.logger.info(f"Running step: {step['name']}")
            try:
                rows_before = len(result)
                nulls_before = result.isna().sum().sum()

                result = step["function"](result, **step["kwargs"])

                rows_after = len(result)
                nulls_after = result.isna().sum().sum()

                self.execution_log.append(
                    {
                        "step": step["name"],
                        "rows_before": rows_before,
                        "rows_after": rows_after,
                        "rows_changed": rows_after - rows_before,
                        "nulls_before": nulls_before,
                        "nulls_after": nulls_after,
                        "status": "success",
                    }
                )
                self.logger.info(
                    f"Step '{step['name']}' completed. "
                    f"Rows: {rows_before} -> {rows_after} | "
                    f"Nulls: {nulls_before} -> {nulls_after}"
                )
            except Exception as e:
                self.execution_log.append(
                    {"step": step["name"], "status": "failed", "error": str(e)}
                )
                self.logger.error(f"Step '{step['name']}' failed: {e}")
                if verbose:
                    print(f"Error: {e}")
                raise

        if verbose:
            print(f"Pipeline '{self.name}' completed successfully.")
        self.logger.info(f"Pipeline '{self.name}' completed successfully.")
        return result

    def get_execution_report(self) -> pd.DataFrame:
        return pd.DataFrame(self.execution_log)

    def save_pipeline(self, filename: str) -> None:
        with open(filename, "wb") as f:
            pickle.dump(self.steps, f)
        print(f"Pipeline saved to {filename}")
        self.logger.info(f"Pipeline saved to {filename}")

    def load_pipeline(self, filename: str) -> "DataCleaningPipeline":
        with open(filename, "rb") as f:
            self.steps = pickle.load(f)
        print(f"Pipeline loaded from {filename}")
        self.logger.info(f"Pipeline loaded from {filename}")
        return self


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    before = list(df.columns)
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]
    changed = [b for b, a in zip(before, df.columns) if b != a]
    if changed:
        print(f"Renamed {len(changed)} column(s): {changed}")
    return df


def strip_and_standardize_strings(df: pd.DataFrame) -> pd.DataFrame:
    _NULL_SENTINELS = {"", "nan", "None", "null"}
    df = df.copy()
    object_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for col in object_cols:
        series = (
            df[col]
            .astype("string")
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)
        )
        df[col] = series.where(~series.isin(_NULL_SENTINELS), pd.NA)  # type: ignore[call-overload]
    return df


def parse_dates(df: pd.DataFrame, date_columns: list[str] = DATE_COLUMNS) -> pd.DataFrame:
    df = df.copy()
    for col in date_columns:
        if col not in df.columns:
            continue
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def coerce_numeric_columns(
    df: pd.DataFrame,
    numeric_candidates: set[str] = _NUMERIC_CANDIDATES,
    percent_columns: set[str] = _PERCENT_COLUMNS,
) -> pd.DataFrame:
    df = df.copy()
    present_cols = [col for col in numeric_candidates if col in df.columns]
    for col in present_cols:
        series = df[col]
        is_percent = col in percent_columns
        if not pd.api.types.is_numeric_dtype(series):
            series = series.astype("string")
            if is_percent:
                series = series.str.replace("%", "", regex=False)
            series = (
                series
                .str.replace(r"[$,]", "", regex=True)
                .str.replace(r"[^0-9eE+\-.]", "", regex=True)
            )
        df[col] = pd.to_numeric(series, errors="coerce")
    return df


def repair_logical_relationships(
    df: pd.DataFrame,
    non_negative_columns: list[str] = _NON_NEGATIVE_COLUMNS,
) -> pd.DataFrame:
    df = df.copy()
    for col in non_negative_columns:
        if col not in df.columns:
            continue
        df.loc[df[col] < 0, col] = pd.NA

    if "fico_range_low" in df.columns and "fico_range_high" in df.columns:
        mask = (
            df["fico_range_low"].notna()
            & df["fico_range_high"].notna()
            & (df["fico_range_low"] > df["fico_range_high"])
        )
        if mask.any():
            df.loc[mask, ["fico_range_low", "fico_range_high"]] = (
                df.loc[mask, ["fico_range_high", "fico_range_low"]].values
            )

    if "pymnt_plan" in df.columns:
        df["pymnt_plan"] = df["pymnt_plan"].astype("string").str.strip().str.lower()

    return df


def handle_missing_values(
    df: pd.DataFrame,
    mandatory_cols: list[str] | None = None,
    indicator_columns: list[str] = INDICATOR_COLUMNS,
    string_columns: list[str] = STRING_COLUMNS,
    positive_amount_columns: list[str] = POSITIVE_AMOUNT_COLUMNS,
    rate_columns: list[str] = RATE_COLUMNS,
) -> pd.DataFrame:
    df = df.copy()

    if mandatory_cols is None:
        mandatory_cols = [col for col in ("id", "issue_d") if col in df.columns]
    if mandatory_cols:
        df = df.dropna(subset=mandatory_cols)

    if "issue_d" in df.columns:
        df = df.sort_values("issue_d").reset_index(drop=True)

    ind_cols = [col for col in indicator_columns if col in df.columns]
    if ind_cols:
        df[ind_cols] = df[ind_cols].ffill().bfill()

    zero_fill = [
        col for col in
        ("delinq_2yrs", "inq_last_6mths", "mths_since_last_delinq", "pub_rec", "acc_now_delinq")
        if col in df.columns
    ]
    df[zero_fill] = df[zero_fill].fillna(0)

    median_fill = [
        col for col in df.columns
        if col in (
            set(positive_amount_columns) | set(rate_columns) | set(indicator_columns)
            | {"open_acc", "total_acc", "fico_range_low", "fico_range_high"}
        )
    ]
    for col in median_fill:
        if df[col].notna().any():
            df[col] = df[col].fillna(df[col].median())

    for col in string_columns:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "id" in df.columns:
        df = df.drop_duplicates(subset=["id"], keep="first")
    df = df.drop_duplicates(keep="first")
    return df


def clip_outliers(
    df: pd.DataFrame,
    columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    df = df.copy()
    target_columns = list(columns) if columns is not None else [
        "loan_amnt", "funded_amnt", "funded_amnt_inv", "installment",
        "annual_inc", "dti", "revol_bal", "revol_util", "total_pymnt",
        "recoveries", "avg_cur_bal", "int_rate",
        "CPI", "Unemployment Rate", "Federal Funds Rate",
    ]
    for col in target_columns:
        if col not in df.columns:
            continue
        series = pd.Series(pd.to_numeric(df[col], errors="coerce"), index=df.index)
        if series.dropna().empty:
            continue
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            continue
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        df[col] = series.clip(lower=lower, upper=upper)
    return df


def finalize_types(
    df: pd.DataFrame,
    count_columns: list[str] = COUNT_COLUMNS,
) -> pd.DataFrame:
    df = df.copy()
    int64_columns = (
        (["id"] if "id" in df.columns else [])
        + [col for col in count_columns if col in df.columns]
        + [col for col in ("fico_range_low", "fico_range_high") if col in df.columns]
    )
    for col in int64_columns:
        df[col] = df[col].round().astype("Int64")
    return df


def clean_merged_dataset(
    input_path: str | Path = RAW_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> pd.DataFrame:
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df_raw = pd.read_csv(input_path, low_memory=False)
    print(f"Raw data loaded: {df_raw.shape[0]} rows x {df_raw.shape[1]} columns")
    print(f"Shape: {df_raw.shape} | Missing values: {df_raw.isna().sum().sum()}")

    pipeline = DataCleaningPipeline("Merged Dataset Cleaning")

    (
        pipeline
        .add_step("Normalize Column Names", normalize_columns)
        .add_step("Strip & Standardize Strings", strip_and_standardize_strings)
        .add_step("Parse Dates", parse_dates)
        .add_step("Coerce Numeric Columns", coerce_numeric_columns)
        .add_step("Repair Logical Relationships", repair_logical_relationships)
        .add_step("Handle Missing Values", handle_missing_values)
        .add_step("Remove Duplicates", remove_duplicates)
        .add_step("Clip Outliers", clip_outliers)
        .add_step("Finalize Types", finalize_types)
    )

    df_cleaned = pipeline.execute(df_raw)

    print(f"Shape: {df_cleaned.shape} | Missing values: {df_cleaned.isna().sum().sum()}")

    report = pipeline.get_execution_report()
    print(report.to_string(index=False))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_cleaned.to_csv(output_path, index=False)
    print(f"Cleaned data saved to: {output_path}")

    return df_cleaned


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean merged dataset")
    parser.add_argument("--input", default="data/merged_df.csv",
                        help="Path to input merged dataset")
    parser.add_argument("--output", default="data/merged_df_cleaned.csv",
                        help="Path to save cleaned dataset")
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    clean_merged_dataset(input_path=Path(args.input), output_path=Path(args.output))