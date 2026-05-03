from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR / "data" / "merged_df_cleaned.csv"
OUTPUT_PATH = BASE_DIR / "data" / "merged_df_transformed.csv"
TRAIN_OUTPUT_PATH = BASE_DIR / "data" / "train.csv"
TRAIN_NORM_OUTPUT_PATH = BASE_DIR / "data" / "train_norm.csv"
TEST_OUTPUT_PATH = BASE_DIR / "data" / "test.csv"
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

MINMAX_COLS = [
    "loan_amnt",
    "funded_amnt",
    "int_rate",
    "installment",
    "annual_inc",
    "dti",
    "delinq_2yrs",
    "inq_last_6mths",
    "mths_since_last_delinq",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
    "avg_cur_bal",
    "acc_now_delinq",
    "CPI",
    "Unemployment Rate",
    "Federal Funds Rate",
    "fico",
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

def save_business_statistics(logger: logging.Logger, data: pd.DataFrame) -> None:
    accepted_loans = data[data["loan_status"] == 0]
    rejected_loans = data[data["loan_status"] == 1]
    # Use only accepted loans to calculate average profit as total_pymnt > loan_amnt
    avg_loan_profit = accepted_loans["total_pymnt"].mean() - accepted_loans["loan_amnt"].mean()
    avg_loan_loss = rejected_loans["loan_amnt"].mean() - rejected_loans["total_pymnt"].mean()
    avg_loan_amount = data["loan_amnt"].mean()
    statistics = pd.DataFrame({
        "avg_loan_profit": [avg_loan_profit],
        "avg_loan_loss": [avg_loan_loss],
        "avg_loan_amount": [avg_loan_amount]
    })
    statistics.to_csv(BASE_DIR / "data" / "business_statistics.csv", index=False)

def load_input_data(path: Path = INPUT_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def filter_loan_status(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data = data[data["loan_status"] != "Current"].copy()
    data = data[data["loan_status"] != "Default"].copy()
    data["loan_status"] = data["loan_status"].apply(
        lambda value: 0 if value in GOOD_STATUS else 1
    )
    return data


def create_fico_feature(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["fico"] = (data["fico_range_high"] + data["fico_range_low"]) / 2
    return data.drop(columns=DROP_COLS, errors="ignore")


def encode_term(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["term"] = data["term"].apply(lambda value: value == "36 months").astype(int)
    return data


def encode_sub_grade(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    sub_grades = data["sub_grade"].unique()
    sorted_sub_grades = np.sort(sub_grades)[::-1]
    dict_sub_grades = {grade: index for index, grade in enumerate(sorted_sub_grades)}
    data["sub_grade"] = data["sub_grade"].apply(lambda value: dict_sub_grades[value])
    return data


def encode_emp_length(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
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
    return data


def encode_pymnt_plan(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["pymnt_plan"] = data["pymnt_plan"].apply(lambda value: value == "y").astype(int)
    return data


def one_hot_encode_columns(
    data: pd.DataFrame,
    columns: list[str],
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    data = data.copy()
    present_columns = [column for column in columns if column in data.columns]
    if not present_columns:
        return data
    if logger is not None:
        logger.info("One-hot encoding columns: %s", present_columns)
    return pd.get_dummies(data, columns=present_columns, drop_first=False)


def normalize_home_ownership(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    keep_home_ownership = ["RENT", "MORTGAGE", "OWN"]
    data["home_ownership"] = data["home_ownership"].where(
        data["home_ownership"].isin(keep_home_ownership),
        other="OTHER",
    )
    return data


def transform_data(logger: logging.Logger) -> pd.DataFrame:
    logger.info("Loading input data from %s", INPUT_PATH)
    data = load_input_data()
    log_step(logger, "Load data", data)

    logger.info("Filtering loan_status values")
    data = filter_loan_status(data)
    log_step(logger, "Filter and encode loan_status", data)
    
    logger.info("Saving business statistics before transformation and dropping unused columns")
    save_business_statistics(logger, data)

    logger.info("Creating fico average and dropping unused columns")
    data = create_fico_feature(data)
    log_step(logger, "FICO feature engineering", data)

    logger.info("Encoding term")
    data = encode_term(data)
    log_step(logger, "Encode term", data)

    logger.info("Encoding sub_grade")
    data = encode_sub_grade(data)
    log_step(logger, "Encode sub_grade", data)

    logger.info("Encoding emp_length")
    data = encode_emp_length(data)
    log_step(logger, "Encode emp_length", data)

    logger.info("Encoding pymnt_plan")
    data = encode_pymnt_plan(data)
    log_step(logger, "Encode pymnt_plan", data)

    logger.info("Normalizing and one-hot encoding home_ownership")
    data = normalize_home_ownership(data)
    data = one_hot_encode_columns(data, ["home_ownership"], logger)
    log_step(logger, "Encode home_ownership", data)

    logger.info("One-hot encoding verification_status")
    data = one_hot_encode_columns(data, ["verification_status"], logger)
    log_step(logger, "Encode verification_status", data)

    logger.info("One-hot encoding purpose")
    data = one_hot_encode_columns(data, ["purpose"], logger)
    log_step(logger, "Encode purpose", data)

    object_cols = data.select_dtypes(include=["object"]).columns.tolist()
    logger.info("Remaining object columns: %s", object_cols)
    logger.info("Transformation complete")
    return data


def normalize_train_test(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_cols = [
        column for column in MINMAX_COLS if column in train.columns and column != "loan_status"
    ]

    train = train.copy()
    test = test.copy()

    train_min = train[feature_cols].min()
    train_max = train[feature_cols].max()
    train_range = (train_max - train_min).replace(0, 1)

    train[feature_cols] = (train[feature_cols] - train_min) / train_range
    test[feature_cols] = (test[feature_cols] - train_min) / train_range

    return train, test


def split_transformed_data(
    merged_df_transformed: pd.DataFrame,
    test_size: float = 0.1,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train, test = train_test_split(
        merged_df_transformed,
        test_size=test_size,
        random_state=random_state,
        stratify=merged_df_transformed["loan_status"],
    )
    return train, test


def save_transformation_outputs(
    logger: logging.Logger,
    merged_df_transformed: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    logger.info("Saving transformed data to %s", OUTPUT_PATH)
    merged_df_transformed.to_csv(OUTPUT_PATH, index=False)
    logger.info("Saved transformed data with shape %s", merged_df_transformed.shape)

    logger.info("Splitting transformed data into train/test sets")
    train, test = split_transformed_data(merged_df_transformed)

    logger.info("Applying min-max normalization using train data only")
    train.to_csv(TRAIN_OUTPUT_PATH)
    train, test = normalize_train_test(train, test)

    logger.info("Saving normalized train data to %s", TRAIN_NORM_OUTPUT_PATH)
    train.to_csv(TRAIN_NORM_OUTPUT_PATH, index=False)
    logger.info("Saving normalized test data to %s", TEST_OUTPUT_PATH)
    test.to_csv(TEST_OUTPUT_PATH, index=False)

    logger.info("Saved train/test splits with shapes train=%s test=%s", train.shape, test.shape)
    return train, test


def main() -> pd.DataFrame:
    logger = setup_logging()
    logger.info("Starting data transformation pipeline")
    merged_df_transformed = transform_data(logger)
    save_transformation_outputs(logger, merged_df_transformed)
    return merged_df_transformed


merged_df_transformed: pd.DataFrame | None = None


if __name__ == "__main__":
    merged_df_transformed = main()