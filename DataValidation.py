import logging
from typing import Any, Dict, List
import pandas as pd
from scipy import stats as scipy_stats
import great_expectations as gx



class DataValidation:
    def __init__(self, data: pd.DataFrame):
        self._setup_logging()
        self.data = data.copy()
        self.report: Dict[str, Dict[str, Any]] = {}
        context = gx.get_context(mode="ephemeral")
        data_source = context.data_sources.add_pandas(name="my_pandas_source")
        data_asset = data_source.add_dataframe_asset(name="users_asset")
        batch_def = data_asset.add_batch_definition_whole_dataframe("my_batch")
        batch = batch_def.get_batch(batch_parameters={"dataframe": df})
        self.suite = context.suites.add(
            gx.ExpectationSuite(name="loan_validation_suite")
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


    

if __name__ == "__main__":
    df = pd.read_csv("data/merged_df.csv")
    validator = DataValidation(df)
    validator.run_all_validations()
    validator.print_report()