import argparse
import logging
import os
from typing import Any, Dict
import pandas as pd
import great_expectations as gx
from scipy.stats import zscore, kstest, skew, kurtosis


REQUIRED_NON_NULL_COLS = [
    "id", "loan_amnt", "funded_amnt", "term", "int_rate", "installment",
    "sub_grade", "home_ownership", "annual_inc", "verification_status",
    "issue_d", "loan_status", "purpose", "dti", "fico_range_low",
    "fico_range_high", "open_acc", "pub_rec", "revol_bal", "revol_util",
    "total_acc", "total_pymnt",
]

RATE_COLUMNS = [
    "int_rate", "revol_util", "Unemployment Rate",
    "Federal Funds Rate", "dti",
]

_NUMERIC_RATIO_THRESHOLD = 0.8

_ZSCORE_THRESHOLD = 3


class DataValidation:
    def __init__(self, data: pd.DataFrame):
        self._setup_logging()
        self.data = data.copy()
        self.total_rows, self.total_cols = self.data.shape
        self.missing_values = self.data.isnull().sum().sum()
        self.num_cols = self.data.select_dtypes(include=["number"]).columns.tolist()
        self.cat_cols = self.data.select_dtypes(include=["object"]).columns.tolist()
        self.report: Dict[str, Any] = {}

        self._add_zscore_columns()

        self.context = gx.get_context(mode="ephemeral")
        data_source = self.context.data_sources.add_pandas(name="my_pandas_source")
        data_asset = data_source.add_dataframe_asset(name="loan_asset")
        batch_def = data_asset.add_batch_definition_whole_dataframe("my_batch")
        batch_def.get_batch(batch_parameters={"dataframe": self.data})
        self.suite = self.context.suites.add(
            gx.ExpectationSuite(name="loan_validation_suite")
        )
        self.validation_def = self.context.validation_definitions.add(
            gx.ValidationDefinition(
                name="loan_validation",
                data=batch_def,
                suite=self.suite,
            )
        )

    def _setup_logging(self) -> None:
        os.makedirs("logs", exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler("logs/DataValidationPipeline.log"),
                logging.StreamHandler(),
            ],
        )
        self.logger = logging.getLogger("DataValidationPipeline")
        self.logger.info("Logger initialized successfully.")

    def _add_zscore_columns(self) -> None:
        self.logger.info("Pre-computing z-score columns.")
        for col in self.num_cols:
            series = pd.to_numeric(self.data[col], errors="coerce")
            self.data[f"{col}_zscore"] = pd.Series(
                zscore(series, nan_policy="omit"),
                index=self.data.index,
            )

    def profile_data(self) -> Dict[str, Any]:
        self.logger.info("Starting basic data profiling (shape, dtypes, nulls, duplicates).")
        dtype_map = {col: str(dtype) for col, dtype in self.data.dtypes.items()}
        suspected_type_mismatches = []
        for col in self.cat_cols:
            coerced = pd.to_numeric(self.data[col], errors="coerce")
            non_null_ratio = coerced.notna().sum() / max(self.data[col].notna().sum(), 1)
            if non_null_ratio > _NUMERIC_RATIO_THRESHOLD:
                suspected_type_mismatches.append(col)
        profile = {
            "row_count": self.total_rows,
            "col_count": self.total_cols,
            "dtypes": dtype_map,
            "suspected_type_mismatches": suspected_type_mismatches,
            "missing_values": self.missing_values
        }
        self.logger.info(f"Profiling complete  {self.total_rows} rows, {self.total_cols} cols.")
        return profile

    def validate_categorical(self) -> Dict[str, Any]:
        self.logger.info("Starting categorical column validation.")
        results = {}
        for col in self.cat_cols:
            series = self.data[col].dropna().astype(str)
            series = series[series.str.lower() != "nan"]

            unique_vals = series.nunique()
            top_values = series.value_counts().head(5).to_dict()
            stripped = series.str.strip().str.lower()
            normalised_unique = stripped.nunique()
            has_inconsistencies = normalised_unique < unique_vals
            cardinality_flag = None
            if unique_vals == len(series):
                cardinality_flag = "all_unique  possible ID column"
            elif unique_vals == 1:
                cardinality_flag = "constant  zero variance"
            elif unique_vals > 0.5 * len(series):
                cardinality_flag = "high_cardinality"
            results[col] = {
                "unique_count": unique_vals,
                "top_5_values": top_values,
                "inconsistencies": has_inconsistencies,
                "cardinality_flag": cardinality_flag,
            }
            self.suite.add_expectation(
                gx.expectations.ExpectColumnValuesToNotMatchRegex(
                    column=col, regex=r"^\s*$"
                )
            )
        self.logger.info("Categorical validation completed successfully.")
        return results

    def check_duplicate_rows(self) -> Dict[str, Any]:
        self.logger.info("Checking for duplicate rows...")
        duplicate_mask = self.data.duplicated(keep=False)
        duplicate_count = int(duplicate_mask.sum())
        sample_indices = self.data[duplicate_mask].index.tolist()[:5]
        result = {
            "duplicate_row_count": duplicate_count,
            "sample_duplicate_indices": sample_indices,
        }
        self.logger.info(f"Duplicate row check complete, {duplicate_count} duplicate row(s) found.")
        return result

    def validate_distribution(self) -> Dict[str, Any]:
        self.logger.info("Starting distribution validation using Kolmogorov-Smirnov test.")
        results = {}
        for col in self.num_cols:
            series = pd.to_numeric(self.data[col], errors="coerce").dropna()
            if len(series) < 2:
                self.logger.warning(f"Not enough data for distribution validation on column: {col}")
                continue
            ks_statistic, p_value = kstest(series, "norm", args=(series.mean(), series.std()))
            results[col] = {
                "ks_statistic": round(float(ks_statistic), 6),
                "p_value": round(float(p_value), 6),
                "skewness": round(float(skew(series)), 4),
                "kurtosis": round(float(kurtosis(series)), 4),
            }
        self.logger.info("Distribution validation completed successfully.")
        return results

    def validate_relationships(self) -> Dict[str, Any]:
        self.logger.info("Starting relationship validation using correlation analysis.")
        relationship_results: Dict[str, Any] = {
            "high_correlation_pairs": [],
            "multicollinearity_risk": "low",
        }
        numeric_df = self.data[self.num_cols].apply(pd.to_numeric, errors="coerce")
        corr_matrix = numeric_df.corr(method="pearson")
        high_corr_threshold = 0.8
        for i, col_a in enumerate(self.num_cols):
            for col_b in self.num_cols[i + 1:]:
                corr_value = corr_matrix.loc[col_a, col_b]
                if pd.notna(corr_value) and abs(corr_value) >= high_corr_threshold:
                    relationship_results["high_correlation_pairs"].append(
                        {
                            "column_a": col_a,
                            "column_b": col_b,
                            "correlation": float(round(corr_value, 4)),
                        }
                    )
        high_corr_count = len(relationship_results["high_correlation_pairs"])
        if 10 <= high_corr_count < 30:
            relationship_results["multicollinearity_risk"] = "moderate"
        elif high_corr_count >= 30:
            relationship_results["multicollinearity_risk"] = "high"
        self.logger.info("Relationship validation completed successfully.")
        return relationship_results


    def validate_accuracy(self) -> None:
        self.logger.info("Adding accuracy expectations to GX suite.")
        for col in self.num_cols:
            self.suite.add_expectation(
                gx.expectations.ExpectColumnValuesToBeBetween(column=col, min_value=0)
            )
        for col in RATE_COLUMNS:
            if col in self.data.columns:
                self.suite.add_expectation(
                    gx.expectations.ExpectColumnValuesToBeBetween(
                        column=col, min_value=0, max_value=100
                    )
                )
        self.logger.info("Accuracy expectations added successfully.")

    def validate_outliers(self) -> None:
        self.logger.info("Adding outlier expectations to GX suite.")
        for col in self.num_cols:
            series = pd.to_numeric(self.data[col], errors="coerce")
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            self.suite.add_expectation(
                gx.expectations.ExpectColumnValuesToBeBetween(
                    column=col,
                    min_value=lower_bound,
                    max_value=upper_bound,
                )
            )
            zscore_col = f"{col}_zscore"
            self.suite.add_expectation(
                gx.expectations.ExpectColumnValuesToBeBetween(
                    column=zscore_col,
                    min_value=-_ZSCORE_THRESHOLD,
                    max_value=_ZSCORE_THRESHOLD,
                )
            )
        self.logger.info("Outlier expectations added successfully.")

    def check_completeness(self) -> None:
        self.logger.info("Adding completeness expectations for required columns.")
        for col in REQUIRED_NON_NULL_COLS:
            if col in self.data.columns:
                self.suite.add_expectation(
                    gx.expectations.ExpectColumnValuesToNotBeNull(column=col)
                )
        self.logger.info("Completeness expectations added successfully.")

    def check_uniqueness(self) -> None:
        self.logger.info("Adding uniqueness expectation for 'id' column.")
        self.suite.add_expectation(
            gx.expectations.ExpectColumnValuesToBeUnique(column="id")
        )
        self.logger.info("Uniqueness expectation added successfully.")

    def _print_gx_results(self, results) -> None:
        """Print the Great Expectations pass/fail table."""
        success = results.success
        print("=" * 58)
        print("DATA VALIDATION REPORT (Great Expectations v1.x)")
        print("=" * 58)
        print(f"Overall Result : {'PASSED' if success else 'FAILED'}")
        print("=" * 58)

        for exp_result in results.results:
            exp_type = exp_result.expectation_config.type
            col = exp_result.expectation_config.kwargs.get("column", "table-level")
            status = "PASS" if exp_result.success else "FAIL"
            print(f"\n[{status}] {exp_type}")
            print(f"   Column : {col}")
            if not exp_result.success and exp_result.result:
                r = exp_result.result
                if r.get("unexpected_count"):
                    print(f"   Issues : {r['unexpected_count']} unexpected values")
                if r.get("partial_unexpected_list"):
                    print(f"   Sample : {r['partial_unexpected_list'][:3]}")

    def _print_statistical_report(self) -> None:
        for dimension, result in self.report.items():
            print(f"\n[INFO] {dimension} Validation Results:")
            if not result:
                print("  - No results available")
                continue

            if dimension == "Profile":
                print(f"  - Row count: {result['row_count']}")
                print(f"  - Column count: {result['col_count']}")
                print(f"  - Missing values: {result['missing_values']}")
                if result["suspected_type_mismatches"]:
                    print(f"  - Suspected type mismatches: {result['suspected_type_mismatches']}")
                print("  - Data types:")
                for col, dtype in result["dtypes"].items():
                    print(f"    * {col}: {dtype}")

            elif dimension == "Duplicates":
                print(f"  - Duplicate row count: {result['duplicate_row_count']}")
                if result["sample_duplicate_indices"]:
                    print(f"  - Sample indices: {result['sample_duplicate_indices']}")

            elif dimension == "Categorical":
                for col_name, metrics in result.items():
                    print(f"  - {col_name}:")
                    print(f"unique_count: {metrics['unique_count']}")
                    print(f"top_5_values: {metrics['top_5_values']}")
                    print(f"inconsistencies: {metrics['inconsistencies']}")
                    if metrics["cardinality_flag"]:
                        print(f"cardinality: {metrics['cardinality_flag']}")

            elif dimension == "Distribution":
                for col_name, metrics in result.items():
                    metrics_text = ", ".join(
                        f"{k}={v}" for k, v in metrics.items()
                    )
                    print(f"  - {col_name}: {metrics_text}")

            elif dimension == "Relationships":
                print(f"  - Multicollinearity Risk: {result['multicollinearity_risk']}")
                if result["high_correlation_pairs"]:
                    print("  - Highly correlated pairs (|corr| >= 0.8):")
                    for pair in result["high_correlation_pairs"]:
                        print(
                            f"* {pair['column_a']} & {pair['column_b']}"
                            f" (corr={pair['correlation']})"
                        )

        print("\n" + "=" * 58)

    def run_all_validations(self) -> None:
        self.validate_accuracy()
        self.validate_outliers()
        self.check_completeness()
        self.check_uniqueness()

        self.report = {
            "Profile": self.profile_data(),
            "Duplicates": self.check_duplicate_rows(),
            "Categorical": self.validate_categorical(),
            "Distribution": self.validate_distribution(),
            "Relationships": self.validate_relationships(),
        }

        gx_results = self.validation_def.run(batch_parameters={"dataframe": self.data})
        self._print_gx_results(gx_results)
        self._print_statistical_report()

        self.context.build_data_docs()
        self.context.open_data_docs()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate merged dataset")
    parser.add_argument("--input", default="data/merged_df.csv",
                        help="Path to input dataset to validate")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    validator = DataValidation(df)
    validator.run_all_validations()