import logging
from typing import Any, Dict, List
import pandas as pd
import great_expectations as gx
from scipy.stats import zscore, kstest, skew, kurtosis
import numpy as np





class DataValidation:
    def __init__(self, data: pd.DataFrame):
        self._setup_logging()
        self.data = data.copy()
        self.num_cols = self.data.select_dtypes(include=['number']).columns.tolist()
        self.cat_cols = self.data.select_dtypes(include=['object']).columns.tolist()
        self.report = {}
        
        self.context = gx.get_context(mode="ephemeral")
        data_source = self.context.data_sources.add_pandas(name="my_pandas_source")
        data_asset = data_source.add_dataframe_asset(name="loan_asset")
        batch_def = data_asset.add_batch_definition_whole_dataframe("my_batch")
        batch = batch_def.get_batch(batch_parameters={"dataframe": self.data})
        self.suite = self.context.suites.add(
            gx.ExpectationSuite(name="loan_validation_suite")
        )
        self.validation_def = self.context.validation_definitions.add(
            gx.ValidationDefinition(
                name="loan_validation",
                data=batch_def,
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
        
    def validate_accuracy(self):
        self.logger.info("Starting accuracy validation using domain-specific rules.")
        self.data['max'] = 100
        self.data['min'] = 0
        for col in self.num_cols:
            self.suite.add_expectation(
                gx.expectations.ExpectColumnPairValuesAToBeGreaterThanB(
                    column_A=col,
                    column_B="min"
                )
            )
        rate_columns = ["int_rate", "revol_util", "Unemployment Rate", "CPI", "Federal Funds Rate", "dti"]
        for col in rate_columns:
            self.suite.add_expectation(
                gx.expectations.ExpectColumnPairValuesAToBeGreaterThanB(
                    column_A="max",
                    column_B=col
                )
            )
        self.logger.info("Accuracy validation expectations added to the suite successfully.")
        
    def validate_outliers(self):
        self.logger.info("Starting outlier validation using IQR and Z-score methods.")
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

            zscore_series = pd.Series(
                zscore(series, nan_policy="omit"),
                index=self.data.index,
            )
            self.data[f"{col} zscore"] = zscore_series
            threshold = 3
            self.suite.add_expectation(
                gx.expectations.ExpectColumnValuesToBeBetween(
                    column=f"{col} zscore",
                    min_value=-threshold,
                    max_value=threshold
                )
            )
        self.logger.info("Outlier validation expectations added to the suite successfully.")
        
    def validate_distribution(self):
        self.logger.info("Starting distribution validation using Kolmogorov-Smirnov test.")
        results = {}
        for col in self.num_cols:
            series = pd.to_numeric(self.data[col], errors="coerce").dropna()
            if len(series) < 2:
                self.logger.warning(f"Not enough data to perform distribution validation for column: {col}")
                continue
            ks_statistic, p_value = kstest(series, 'norm', args=(series.mean(), series.std()))
            skewness = skew(series)
            kurtosis_value = kurtosis(series)
            results[col] = {
                "p_value": p_value,
                "skewness": skewness,
                "kurtosis": kurtosis_value,
            }
        self.logger.info("Distribution validation completed successfully.")
        return results
        
    def validate_relationships(self):
        self.logger.info("Starting relationship validation using correlation analysis.")
        relationship_results = {
            "high_correlation_pairs": [],
            "multicollinearity_risk": "low"
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
        if high_corr_count >= 10 and high_corr_count < 30:    relationship_results["multicollinearity_risk"] = "moderate"
        elif high_corr_count >= 30:  relationship_results["multicollinearity_risk"] = "high"

        self.logger.info("Relationship validation completed successfully.")
        return relationship_results
    
    def check_completeness(self):
        self.logger.info("Checking completeness of columns...")
        for col in self.data.columns:
            self.suite.add_expectation(
                gx.expectations.ExpectColumnValuesToNotBeNull(column=col)
            )
        self.logger.info("Completeness expectations added to the suite successfully.")
        
    def check_uniqueness(self):
        self.logger.info("Checking uniqueness of 'id' column...")
        self.suite.add_expectation(
            gx.expectations.ExpectColumnValuesToBeUnique(column="id")
        )
        self.logger.info("Uniqueness expectation for 'id' column added to the suite successfully.")
    

    def run_all_validations(self):
        self.validate_accuracy()
        self.validate_outliers()
        self.check_completeness()
        self.check_uniqueness()
        self.report = {
            "Distribution": self.validate_distribution(),
            "Relationships": self.validate_relationships(),
        }
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
        
        for dimension, result in self.report.items():
            print(f"\n[INFO] {dimension} Validation Results:")
            if not result:
                print("  - No results available")
                continue

            # Distribution output: {column_name: {metric: value}}
            if isinstance(result, dict) and result and all(isinstance(v, dict) for v in result.values()):
                for col_name, metrics in list(result.items()):
                    metrics_text = ", ".join(
                        [f"{k}={v if isinstance(v, (int, float)) else v}" for k, v in metrics.items()]
                    )
                    print(f"  - {col_name}: {metrics_text}")
                continue

            # Relationship output containing lists and scalar values
            for key, value in result.items():
                if isinstance(value, list):
                    print(f"  - {key}: {len(value)} item(s)")
                    for item in value:
                        print(f"    * {item}")
                else:
                    print(f"  - {key}: {value}")
                    
        print("\n" + "=" * 58)



if __name__ == "__main__":
    df = pd.read_csv("data/merged_df.csv")
    validator = DataValidation(df)
    validator.run_all_validations()
