import logging
from typing import Any, Dict, List
import pandas as pd
import great_expectations as gx
from scipy.stats import zscore
import numpy as np



class DataValidation:
    def __init__(self, data: pd.DataFrame):
        self._setup_logging()
        self.data = data.copy()
        self.num_cols = self.data.select_dtypes(include=['number']).columns.tolist()
        self.cat_cols = self.data.select_dtypes(include=['object']).columns.tolist()
        self.report: Dict[str, Dict[str, Any]] = {}
        
        self.context = gx.get_context(mode="ephemeral")
        self.data_source = self.context.data_sources.add_pandas(name="my_pandas_source")
        self.data_asset = self.data_source.add_dataframe_asset(name="loan_asset")
        self.batch_def = self.data_asset.add_batch_definition_whole_dataframe("my_batch")
        self.batch = self.batch_def.get_batch(batch_parameters={"dataframe": self.data})
        self.suite = self.context.suites.add(
            gx.ExpectationSuite(name="loan_validation_suite")
        )
        self.validation_def = self.context.validation_definitions.add(
            gx.ValidationDefinition(
                name="loan_validation",
                data=self.batch_def,
                suite=self.suite
            )
        )

    def _setup_logging(self) -> None:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler("DataValidationPipeline.log"),
                logging.StreamHandler(),
            ],
        )
        self.logger = logging.getLogger("DataValidationPipeline")
        self.logger.info("Logger initialized successfully.")
        
        
    def validate_outliers(self):
        results = {}
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
                    max_value=upper_bound
                )
            )

            # Keep z-score output aligned to the original dataframe index.
            non_null_count = int(series.notna().sum())
            std_dev = float(series.std(ddof=0)) if non_null_count else 0.0
            if non_null_count < 2 or std_dev == 0.0:
                zscore_series = pd.Series(np.nan, index=self.data.index)
            else:
                zscore_series = pd.Series(
                    zscore(series, nan_policy="omit"),
                    index=self.data.index,
                )

            self.data[f"{col}_zscore_outliers"] = zscore_series
            threshold = 3
            self.suite.add_expectation(
                gx.expectations.ExpectColumnValuesToBeBetween(
                    column=f"{col}_zscore_outliers",
                    min_value=-threshold,
                    max_value=threshold
                )
            )
        
    def run_all_validations(self):
        self.validate_outliers()
        results = self.validation_def.run(batch_parameters={"dataframe": self.data})
        self._print_report(results)
        self.context.build_data_docs()
        self.context.open_data_docs()
        
    def _print_report(self, results):
        result_dict = results.describe()
        success     = results.success

        print("=" * 58)
        print("    DATA VALIDATION REPORT  (Great Expectations v1.x)")
        print("=" * 58)
        print(f"  Overall Result : {'PASSED' if success else 'FAILED'}")
        print("=" * 58)

        for exp_result in results.results:
            exp_type = exp_result.expectation_config.type
            col      = exp_result.expectation_config.kwargs.get("column", "table-level")
            passed   = exp_result.success
            status   = "PASS" if passed else "FAIL"

            print(f"\n[{status}] {exp_type}")
            print(f"   Column : {col}")

            if not passed and exp_result.result:
                r = exp_result.result
                if r.get("unexpected_count"):
                    print(f"   Issues : {r['unexpected_count']} unexpected values")
                if r.get("partial_unexpected_list"):
                    print(f"   Sample : {r['partial_unexpected_list'][:3]}")
        print("\n" + "=" * 58)


            

if __name__ == "__main__":
    df = pd.read_csv("data/merged_df.csv")
    validator = DataValidation(df)
    validator.run_all_validations()
