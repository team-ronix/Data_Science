from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import pandas as pd
import os


BASE_DIR = Path(__file__).resolve().parent
RAW_PATH = BASE_DIR / "data/merged_df.csv"
DEFAULT_OUTPUT_PATH = BASE_DIR / "data/merged_df_cleaned.csv"

DATE_COLUMNS = ["issue_d", "earliest_cr_line", "last_pymnt_d"]

STRING_COLUMNS = [
    "term", "sub_grade", "emp_title", "emp_length", "home_ownership",
    "verification_status", "loan_status", "pymnt_plan", "purpose",
]

COUNT_COLUMNS = [
    "delinq_2yrs", "inq_last_6mths", "mths_since_last_delinq",
    "open_acc", "pub_rec", "total_acc", "acc_now_delinq",
]

POSITIVE_AMOUNT_COLUMNS = [
    "loan_amnt", "funded_amnt", "funded_amnt_inv", "installment",
    "annual_inc", "revol_bal", "total_pymnt", "recoveries", "avg_cur_bal",
]

RATE_COLUMNS = ["int_rate", "revol_util", "dti"]

INDICATOR_COLUMNS = ["CPI", "Unemployment Rate", "Federal Funds Rate"]

_NON_NEGATIVE_COLUMNS = POSITIVE_AMOUNT_COLUMNS + RATE_COLUMNS + COUNT_COLUMNS

_NUMERIC_CANDIDATES = {
    "id", "fico_range_low", "fico_range_high",
    *COUNT_COLUMNS, *POSITIVE_AMOUNT_COLUMNS, *RATE_COLUMNS, *INDICATOR_COLUMNS,
}

_PERCENT_COLUMNS = {"int_rate", "revol_util"}


class DataCleaningPipeline:
    def __init__(
        self,
        input_path: str | Path = RAW_PATH,
        output_path: str | Path = DEFAULT_OUTPUT_PATH,
    ) -> None:
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self._setup_logging()
        self.logger.info("Data Cleaning Pipeline initialized.")
        self.logger.info(f"Input path  : {self.input_path}")
        self.logger.info(f"Output path : {self.output_path}")

    def _setup_logging(self) -> None:
        os.makedirs("logs", exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler("logs/DataCleaningPipeline.log"),
                logging.StreamHandler(),
            ],
        )
        self.logger = logging.getLogger("DataCleaningPipeline")
        self.logger.info("Logger initialized successfully.")

    def load_data(self) -> pd.DataFrame:
        if not self.input_path.exists():
            self.logger.error(f"Input file not found: {self.input_path}")
            raise FileNotFoundError(f"Input file not found: {self.input_path}")
        self.logger.info(f"Loading raw data from: {self.input_path}")
        df = pd.read_csv(self.input_path, low_memory=False)
        self.logger.info(f"Raw data loaded successfully: {df.shape[0]:,} rows x {df.shape[1]} columns.")
        return df

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Starting cleaning pipeline.")
        cleaned = df.copy()
        cleaned = self._normalize_columns(cleaned)
        cleaned = self._strip_and_standardize_strings(cleaned)
        cleaned = self._parse_dates(cleaned)
        cleaned = self._coerce_numeric_columns(cleaned)
        cleaned = self._repair_logical_relationships(cleaned)
        cleaned = self._handle_missing_values(cleaned)
        cleaned = self._remove_duplicates(cleaned)
        cleaned = self._clip_outliers(cleaned)
        cleaned = self._finalize_types(cleaned)
        self.logger.info("Cleaning pipeline completed successfully.")
        return cleaned

    def save(self, df: pd.DataFrame) -> Path:
        self.logger.info(f"Saving cleaned data to: {self.output_path}")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.output_path, index=False)
        self.logger.info(f"Cleaned data saved successfully: {df.shape[0]:,} rows x {df.shape[1]} columns.")
        return self.output_path

    def run(self) -> pd.DataFrame:
        self.logger.info("Pipeline run started.")
        raw = self.load_data()
        self.logger.info(f"Raw shape: {raw.shape}")
        cleaned = self.clean(raw)
        self.logger.info(f"Cleaned shape: {cleaned.shape}")
        self.save(cleaned)
        self.logger.info("Pipeline run finished.")
        return cleaned

    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Normalizing column names (strip whitespace).")
        df = df.copy()
        before = list(df.columns)
        df.columns = [str(col).strip() for col in df.columns]
        changed = [b for b, a in zip(before, df.columns) if b != a]
        if changed:
            self.logger.info(f"Renamed {len(changed)} column(s) with leading/trailing whitespace: {changed}")
        else:
            self.logger.info("No column names required renaming.")
        return df

    def _strip_and_standardize_strings(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Stripping and standardizing string columns.")
        df = df.copy()
        _NULL_SENTINELS = {"", "nan", "None", "null"}
        object_cols = df.select_dtypes(include=["object"]).columns.tolist()
        self.logger.info(f"Found {len(object_cols)} object column(s) to process.")
        nullified_total = 0
        for col in object_cols:
            series = (
                df[col]
                .astype("string")
                .str.strip()
                .str.replace(r"\s+", " ", regex=True)
            )
            nullified = series.isin(_NULL_SENTINELS).sum()
            if nullified:
                nullified_total += nullified
                self.logger.info(f"  Column '{col}': replaced {nullified} sentinel value(s) with NA.")
            df[col] = series.where(~series.isin(_NULL_SENTINELS), pd.NA)
        self.logger.info(f"String standardization complete. Total sentinel values nullified: {nullified_total}.")
        return df

    def _parse_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info(f"Parsing date columns: {DATE_COLUMNS}.")
        df = df.copy()
        for col in DATE_COLUMNS:
            if col not in df.columns:
                self.logger.warning(f"  Date column '{col}' not found in DataFrame. Skipping.")
                continue
            before_nulls = df[col].isna().sum()
            df[col] = pd.to_datetime(df[col], errors="coerce")
            after_nulls = df[col].isna().sum()
            new_nulls = after_nulls - before_nulls
            if new_nulls:
                self.logger.warning(f"  Column '{col}': {new_nulls} value(s) could not be parsed and were set to NaT.")
            else:
                self.logger.info(f"  Column '{col}': parsed successfully with no new nulls.")
        return df

    def _coerce_numeric_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Coercing numeric columns.")
        df = df.copy()
        present_cols = [col for col in _NUMERIC_CANDIDATES if col in df.columns]
        missing_cols = [col for col in _NUMERIC_CANDIDATES if col not in df.columns]
        if missing_cols:
            self.logger.warning(f"  Numeric candidate columns not found in DataFrame: {sorted(missing_cols)}")
        self.logger.info(f"  Processing {len(present_cols)} numeric column(s).")
        for col in present_cols:
            series = df[col]
            is_percent = col in _PERCENT_COLUMNS
            if not pd.api.types.is_numeric_dtype(series):
                series = series.astype("string")
                if is_percent:
                    series = series.str.replace("%", "", regex=False)
                series = (
                    series
                    .str.replace(r"[$,]", "", regex=True)
                    .str.replace(r"[^0-9eE+\-.]", "", regex=True)
                )
            before_nulls = pd.to_numeric(df[col], errors="coerce").isna().sum()
            series = pd.to_numeric(series, errors="coerce")
            after_nulls = series.isna().sum()
            new_nulls = after_nulls - before_nulls
            if new_nulls > 0:
                self.logger.warning(f"  Column '{col}': {new_nulls} value(s) could not be coerced and were set to NaN.")
            if is_percent and series.dropna().gt(1).any():
                series = series / 100
                self.logger.info(f"  Column '{col}': percent values divided by 100.")
            df[col] = series
        self.logger.info("Numeric coercion complete.")
        return df

    def _repair_logical_relationships(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Repairing logical relationships.")
        df = df.copy()
        total_nullified = 0
        for col in _NON_NEGATIVE_COLUMNS:
            if col not in df.columns:
                continue
            negatives = (df[col] < 0).sum()
            if negatives:
                self.logger.warning(f"  Column '{col}': nullified {negatives} negative value(s).")
                df.loc[df[col] < 0, col] = pd.NA
                total_nullified += negatives
        if total_nullified == 0:
            self.logger.info("  No negative values found in non-negative columns.")

        if "fico_range_low" in df.columns and "fico_range_high" in df.columns:
            mask = (
                df["fico_range_low"].notna()
                & df["fico_range_high"].notna()
                & (df["fico_range_low"] > df["fico_range_high"])
            )
            if mask.any():
                self.logger.warning(f"  Swapped {mask.sum()} row(s) where fico_range_low > fico_range_high.")
                df.loc[mask, ["fico_range_low", "fico_range_high"]] = (
                    df.loc[mask, ["fico_range_high", "fico_range_low"]].values
                )
            else:
                self.logger.info("  FICO range order is consistent across all rows.")

        if "pymnt_plan" in df.columns:
            df["pymnt_plan"] = df["pymnt_plan"].astype("string").str.strip().str.lower()
            self.logger.info("  Column 'pymnt_plan': normalized to lowercase.")

        self.logger.info("Logical relationship repair complete.")
        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Handling missing values.")
        df = df.copy()

        mandatory = [col for col in ("id", "issue_d") if col in df.columns]
        if mandatory:
            before = len(df)
            df = df.dropna(subset=mandatory)
            dropped = before - len(df)
            if dropped:
                self.logger.warning(f"  Dropped {dropped} row(s) missing mandatory column(s): {mandatory}.")
            else:
                self.logger.info(f"  No rows dropped for mandatory columns: {mandatory}.")

        if "issue_d" in df.columns:
            df = df.sort_values("issue_d").reset_index(drop=True)
            self.logger.info("  DataFrame sorted by 'issue_d'.")

        indicator_cols = [col for col in INDICATOR_COLUMNS if col in df.columns]
        if indicator_cols:
            df[indicator_cols] = df[indicator_cols].ffill().bfill()
            self.logger.info(f"  Forward/backward filled indicator column(s): {indicator_cols}.")

        zero_fill = [
            col for col in
            ("delinq_2yrs", "inq_last_6mths", "mths_since_last_delinq", "pub_rec", "acc_now_delinq")
            if col in df.columns
        ]
        df[zero_fill] = df[zero_fill].fillna(0)
        self.logger.info(f"  Zero-filled column(s): {zero_fill}.")

        median_fill = [
            col for col in df.columns
            if col in (
                set(POSITIVE_AMOUNT_COLUMNS) | set(RATE_COLUMNS) | set(INDICATOR_COLUMNS)
                | {"open_acc", "total_acc", "fico_range_low", "fico_range_high"}
            )
        ]
        for col in median_fill:
            if df[col].notna().any():
                missing_count = df[col].isna().sum()
                if missing_count:
                    median_val = df[col].median()
                    df[col] = df[col].fillna(median_val)
                    self.logger.info(f"  Column '{col}': filled {missing_count} missing value(s) with median ({median_val:.4f}).")

        for col in STRING_COLUMNS:
            if col in df.columns:
                missing_count = df[col].isna().sum()
                if missing_count:
                    df[col] = df[col].fillna("Unknown")
                    self.logger.info(f"  Column '{col}': filled {missing_count} missing value(s) with 'Unknown'.")

        self.logger.info("Missing value handling complete.")
        return df

    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Removing duplicate rows.")
        df = df.copy()
        before = len(df)

        if "id" in df.columns:
            df = df.drop_duplicates(subset=["id"], keep="first")
            id_dupes = before - len(df)
            if id_dupes:
                self.logger.warning(f"  Removed {id_dupes} duplicate(s) based on 'id' column.")
            else:
                self.logger.info("  No duplicate 'id' values found.")

        before_full = len(df)
        df = df.drop_duplicates(keep="first")
        full_dupes = before_full - len(df)
        if full_dupes:
            self.logger.warning(f"  Removed {full_dupes} fully duplicate row(s).")
        else:
            self.logger.info("  No fully duplicate rows found.")

        total_removed = before - len(df)
        self.logger.info(f"Duplicate removal complete. Total rows removed: {total_removed}.")
        return df

    def _clip_outliers(
        self,
        df: pd.DataFrame,
        columns: Iterable[str] | None = None,
    ) -> pd.DataFrame:
        self.logger.info("Clipping outliers using IQR method (1.5x IQR).")
        df = df.copy()
        target_columns = list(columns) if columns is not None else [
            "loan_amnt", "funded_amnt", "funded_amnt_inv", "installment",
            "annual_inc", "dti", "revol_bal", "revol_util", "total_pymnt",
            "recoveries", "avg_cur_bal", "int_rate",
            "CPI", "Unemployment Rate", "Federal Funds Rate",
        ]
        clipped_total = 0
        for col in target_columns:
            if col not in df.columns:
                self.logger.warning(f"  Column '{col}' not found. Skipping outlier clipping.")
                continue
            series = pd.to_numeric(df[col], errors="coerce")
            if series.dropna().empty:
                self.logger.warning(f"  Column '{col}' has no numeric values. Skipping.")
                continue
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            if pd.isna(iqr) or iqr == 0:
                self.logger.warning(f"  Column '{col}': IQR is 0 or NaN. Skipping clipping.")
                continue
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            clipped = ((series < lower) | (series > upper)).sum()
            df[col] = series.clip(lower=lower, upper=upper)
            if clipped:
                clipped_total += clipped
                self.logger.info(f"  Column '{col}': clipped {clipped} outlier(s) to [{lower:.4f}, {upper:.4f}].")
            else:
                self.logger.info(f"  Column '{col}': no outliers detected.")
        self.logger.info(f"Outlier clipping complete. Total values clipped: {clipped_total}.")
        return df

    def _finalize_types(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Finalizing column data types.")
        df = df.copy()
        int64_columns = (
            (["id"] if "id" in df.columns else [])
            + [col for col in COUNT_COLUMNS if col in df.columns]
            + [col for col in ("fico_range_low", "fico_range_high") if col in df.columns]
        )
        for col in int64_columns:
            df[col] = df[col].round().astype("Int64")
            self.logger.info(f"  Column '{col}': cast to Int64.")
        self.logger.info(f"Type finalization complete. {len(int64_columns)} column(s) cast to Int64.")
        return df


def clean_merged_dataset(
    input_path: str | Path = RAW_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> pd.DataFrame:
    return DataCleaningPipeline(input_path=input_path, output_path=output_path).run()


if __name__ == "__main__":
    clean_merged_dataset()