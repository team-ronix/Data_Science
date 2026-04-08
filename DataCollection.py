import requests
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from urllib.robotparser import RobotFileParser
from urllib.parse import urlparse, urljoin

import logging
import os

from datetime import datetime
import time
from dotenv import load_dotenv

load_dotenv()


class DataCollectionPipeline:
    def __init__(self):
        self._setup_logging()
        self._create_session()
        self.logger.info("Data Collection Pipeline initialized.")        
        
    def _create_session(self):
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        self.session.headers.update(
            {
                "Accept": "application/vnd.github+json",
                "User-Agent": "DataCollectionPipeline/1.0",
            }
        )
        self.logger.info("HTTP session initialized successfully with custom headers.")
    
    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('DataCollectionPipeline.log'), 
                logging.StreamHandler(), 
            ],
        )
        self.logger = logging.getLogger("DataCollectionPipeline")
        self.logger.info("Logger initialized successfully.")  
    
    def collect_from_dataset(self, wanted_cols: list, source_dataset_path: str = 'data/Lending Club loan.csv') -> pd.DataFrame:
        self.logger.info(f"Collecting from database: {source_dataset_path}")
        try:
            if not os.path.exists(source_dataset_path):
                raise FileNotFoundError(f"Dataset not found at path: {source_dataset_path}")
            # low_memory=False avoids chunk-based mixed-type inference warnings on wide CSV files.
            df = pd.read_csv(source_dataset_path, low_memory=False)
            missing_cols = [col for col in wanted_cols if col not in df.columns]
            if missing_cols:
                self.logger.warning(f"Missing columns in dataset: {missing_cols}")
            if 'issue_d' in df.columns:
                df['issue_d'] = pd.to_datetime(df['issue_d'], errors='coerce')
            available_cols = [col for col in wanted_cols if col in df.columns]
            if not available_cols:
                self.logger.error("None of the requested columns exist in the dataset.")
                return pd.DataFrame()
            return df[available_cols]
        except Exception as e:
            self.logger.error(f"Database error: {e}")
            return pd.DataFrame()
    
    def _collect_api_query(self, target_url: str, params: dict):
        try:
            response = self.session.get(target_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            observations = data.get('observations', [])
            new_indicators = []
            for item in observations:
                date_str = item.get('date', None)
                if date_str is None:    continue
                date = datetime.strptime(date_str, "%Y-%m-%d")
                data = { 'Year': date.year,     'Month': date.month}
                if params.get('series_id') == 'CPIAUCSL':   data['CPI'] = item.get('value', None)
                elif params.get('series_id') == 'UNRATE':   data['Unemployment Rate'] = item.get('value', None)
                elif params.get('series_id') == 'FEDFUNDS':   data['Federal Funds Rate'] = item.get('value', None)
                else:   continue
                new_indicators.append(data)
            return new_indicators
        except Exception as e:
            self.logger.error(f"API collection error: {e}")
     
    def collect_from_api(self, API_url: str, params: dict, target_url: str) -> pd.DataFrame:
        self.logger.info(f"Collecting from API: {API_url}")     
        params['api_key'] = os.getenv('API_KEY')
        params['file_type'] = 'json'
        indicators = self._collect_api_query(target_url, params)    
        self.logger.info(f"Data is collected for API with params: {params}")        
        time.sleep(1) 
        df = pd.DataFrame(indicators)
        if df.empty:
            self.logger.info("No new records collected from API.")
            return df
        self.logger.info(f"Indicators collected: {len(df)} records")
        return df
    
    def merge_indicators(self, df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Merging indicators.")
        if df1.empty or df2.empty:
            self.logger.warning("One or more indicator DataFrames are empty. Skipping merge.")
            return pd.DataFrame()
        merged_df = pd.merge(df1, df2, on=['Year', 'Month'], how='outer')
        self.logger.info(f"Merged dataset shape: {merged_df.shape}")
        return merged_df
    
    def merge_indicators_with_loans(self, loan_df: pd.DataFrame, indicators_df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Merging loan dataset with indicators.")
        if loan_df.empty:
            self.logger.warning("Loan dataset is empty. Skipping merge.")
            return indicators_df
        if indicators_df.empty:
            self.logger.warning("Indicators dataset is empty. Skipping merge.")
            return loan_df
        loan_df['issue_year'] = loan_df['issue_d'].dt.year
        loan_df['issue_month'] = loan_df['issue_d'].dt.month
        merged_df = pd.merge(loan_df, indicators_df, left_on=['issue_year', 'issue_month'], right_on=['Year', 'Month'], how='left')
        merged_df.drop(columns=['issue_year', 'issue_month', 'Year', 'Month'], inplace=True)
        self.logger.info(f"Merged dataset shape: {merged_df.shape}")
        return merged_df


if __name__ == "__main__":
    pipeline = DataCollectionPipeline()
    wanted_cols = [
        'id', 'loan_amnt','funded_amnt','funded_amnt_inv','term','int_rate','installment','sub_grade','emp_title',
        'emp_length','home_ownership','annual_inc','verification_status','issue_d','loan_status','pymnt_plan',
        'purpose','dti','delinq_2yrs','earliest_cr_line','fico_range_low','fico_range_high','inq_last_6mths',
        'mths_since_last_delinq','open_acc','pub_rec','revol_bal','revol_util','total_acc','total_pymnt',
        'recoveries','last_pymnt_d','avg_cur_bal', 'acc_now_delinq'
    ]
    loan_df = pipeline.collect_from_dataset(wanted_cols)
    series_ids = ['CPIAUCSL', 'UNRATE', 'FEDFUNDS']
    target_url = "https://api.stlouisfed.org/fred/series/observations"
    start_date = loan_df['issue_d'].min().strftime('%Y-%m-%d')
    end_date = loan_df['issue_d'].max().strftime('%Y-%m-%d')
    indicators_dfs = []
    for series_id in series_ids:
        params = {
            'series_id': series_id,
            'observation_start': start_date,
            'observation_end': end_date
        }
        df = pipeline.collect_from_api(target_url, params, target_url)
        indicators_dfs.append(df)
    merged_indicators_df = pipeline.merge_indicators(indicators_dfs[0], indicators_dfs[1])
    merged_indicators_df = pipeline.merge_indicators(merged_indicators_df, indicators_dfs[2])
    merged_indicators_df.to_csv(f'data/indicators_df.csv', index=False)
    merged_df = pipeline.merge_indicators_with_loans(loan_df, merged_indicators_df)
    merged_df.to_csv(f'data/merged_df.csv', index=False)

    