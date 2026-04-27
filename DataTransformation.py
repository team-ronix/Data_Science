from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR / "data" / "merged_df_cleaned.csv"
OUTPUT_PATH = BASE_DIR / "data" / "merged_df_transformed.csv"
LOG_PATH = BASE_DIR / "logs/DataTransformationPipeline.log"

GOOD_STATUS = [
    "Fully Paid",
    "In Grace Period",
    "Does not meet the credit policy. Status:Fully Paid",
]

DROP_COLS = [
    "id",
    "last_pymnt_d",
    "total_pymnt",
    "recoveries",
    "emp_title",
    "funded_amnt_inv",
    "fico_range_low",
    "fico_range_high",
    "issue_d",
    "earliest_cr_line",
]


def setup_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(LOG_PATH),
            logging.StreamHandler(),
        ],
        force=True,
    )
    return logging.getLogger("DataTransformationPipeline")


def log_step(logger: logging.Logger, step_name: str, dataframe: pd.DataFrame) -> None:
    logger.info(
        "%s completed | rows=%s columns=%s nulls=%s",
        step_name,
        len(dataframe),
        len(dataframe.columns),
        int(dataframe.isna().sum().sum()),
    )


def transform_data(logger: logging.Logger) -> pd.DataFrame:
    logger.info("Loading input data from %s", INPUT_PATH)
    data = pd.read_csv(INPUT_PATH)
    log_step(logger, "Load data", data)

    logger.info("Filtering loan_status values")
    data = data[data["loan_status"] != "Current"].copy()
    data = data[data["loan_status"] != "Default"].copy()
    data["loan_status"] = data["loan_status"].apply(lambda value: 1 if value in GOOD_STATUS else 0)
    log_step(logger, "Filter and encode loan_status", data)

    logger.info("Creating fico average and dropping unused columns")
    data["fico"] = (data["fico_range_high"] + data["fico_range_low"]) / 2
    data.drop(columns=DROP_COLS, inplace=True)
    log_step(logger, "FICO feature engineering", data)

    logger.info("Encoding term")
    data["term"] = data["term"].apply(lambda value: value == "36 months").astype(int)
    log_step(logger, "Encode term", data)

    logger.info("Encoding sub_grade")
    sub_grades = data["sub_grade"].unique()
    sorted_sub_grades = np.sort(sub_grades)[::-1]
    dict_sub_grades = {grade: index for index, grade in enumerate(sorted_sub_grades)}
    data["sub_grade"] = data["sub_grade"].apply(lambda value: dict_sub_grades[value])
    log_step(logger, "Encode sub_grade", data)

    logger.info("Encoding emp_length")
    dict_emp_length: dict[str, int] = {}
    for unique_value in data["emp_length"].unique():
        if unique_value == "10+ years":
            dict_emp_length[unique_value] = 10
        elif unique_value == "< 1 year":
            dict_emp_length[unique_value] = 0
        elif unique_value == "Unknown":
            dict_emp_length[unique_value] = -1
        else:
            dict_emp_length[unique_value] = int(unique_value.split()[0])

    data["emp_length"] = data["emp_length"].apply(lambda value: dict_emp_length[value])
    log_step(logger, "Encode emp_length", data)

    logger.info("Encoding pymnt_plan")
    data["pymnt_plan"] = data["pymnt_plan"].apply(lambda value: value == "y").astype(int)
    log_step(logger, "Encode pymnt_plan", data)

    logger.info("Normalizing and one-hot encoding home_ownership")
    keep_home_ownership = ["RENT", "MORTGAGE", "OWN"]
    data["home_ownership"] = data["home_ownership"].where(
        data["home_ownership"].isin(keep_home_ownership),
        other="OTHER",
    )
    data = pd.get_dummies(data, columns=["home_ownership"], drop_first=False)
    log_step(logger, "Encode home_ownership", data)

    logger.info("One-hot encoding verification_status")
    data = pd.get_dummies(data, columns=["verification_status"], drop_first=False)
    log_step(logger, "Encode verification_status", data)

    logger.info("One-hot encoding purpose")
    data = pd.get_dummies(data, columns=["purpose"], drop_first=False)
    log_step(logger, "Encode purpose", data)

    object_cols = data.select_dtypes(include=["object"]).columns.tolist()
    logger.info("Remaining object columns: %s", object_cols)
    logger.info("Transformation complete")
    return data


def main() -> pd.DataFrame:
    logger = setup_logging()
    logger.info("Starting data transformation pipeline")
    merged_df_transformed = transform_data(logger)
    logger.info("Saving transformed data to %s", OUTPUT_PATH)
    merged_df_transformed.to_csv(OUTPUT_PATH, index=False)
    logger.info("Saved transformed data with shape %s", merged_df_transformed.shape)
    return merged_df_transformed


merged_df_transformed: pd.DataFrame | None = None


if __name__ == "__main__":
    merged_df_transformed = main()
    print(merged_df_transformed.head())