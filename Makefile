.DEFAULT_GOAL := help

PYTHON := poetry run python
PYTEST := poetry run pytest

# Data paths
DATA_DIR ?= data
LOAN_DATA ?= $(DATA_DIR)/"Lending Club loan.csv"
INDICATORS_DATA ?= $(DATA_DIR)/indicators_df.csv
MERGED_DATA ?= $(DATA_DIR)/merged_df.csv
CLEANED_DATA ?= $(DATA_DIR)/merged_df_cleaned.csv
TRANSFORMED_DATA ?= $(DATA_DIR)/merged_df_transformed.csv
TRAIN_DATA ?= $(DATA_DIR)/train.csv
TRAIN_NORM_DATA ?= $(DATA_DIR)/train_norm.csv
TRAIN_UNDERSAMPLED_DATA ?= $(DATA_DIR)/train_undersampled.csv
TEST_DATA ?= $(DATA_DIR)/test.csv

# Undersampling parameters
UNDERSAMPLE_TARGET ?= loan_status
UNDERSAMPLE_RATIO ?= 1.0

# Model parameters
OUTPUT_DIR ?= model_outputs
RANDOM_SEARCH_ITER ?= 10

# Business data
BUSINESS_STATS ?= $(DATA_DIR)/business_statistics.csv

# Test parameters
TEST_DIR ?= tests

.PHONY: help install preprocess collect clean validate transform train evaluate test all

help:
	@echo Available targets:
	@echo   install      Install project dependencies with Poetry
	@echo   preprocess   Run data collection, cleaning, validation, and transformation
	@echo   collect      Build the merged raw dataset
	@echo   clean        Clean the merged dataset
	@echo   validate     Validate the merged dataset
	@echo   transform    Create transformed and train/test datasets
	@echo   train        Train and evaluate the model pipeline
	@echo   evaluate     Run the model pipeline for metrics and reports
	@echo   test         Run the test suite
	@echo   all          Run preprocessing, training, and tests
	@echo 
	@echo Environment variables to override defaults:
	@echo   DATA_DIR              Base data directory (default: data)
	@echo   LOAN_DATA             Loan dataset path
	@echo   INDICATORS_DATA       Indicators dataset path
	@echo   MERGED_DATA           Merged dataset path
	@echo   CLEANED_DATA          Cleaned dataset path
	@echo   TRANSFORMED_DATA      Transformed dataset path
	@echo   TRAIN_DATA            Train split path
	@echo   TRAIN_NORM_DATA       Normalized train split path
	@echo   TRAIN_UNDERSAMPLED_DATA Undersampled train split path
	@echo   TEST_DATA             Test split path
	@echo   OUTPUT_DIR            Model output directory (default: model_outputs)
	@echo   RANDOM_SEARCH_ITER    Hyperparameter search iterations (default: 10)
	@echo 
	@echo Example:
	@echo   make preprocess DATA_DIR=./custom_data
	@echo   make train OUTPUT_DIR=./my_models RANDOM_SEARCH_ITER=20

install:
	poetry install

preprocess: collect validate clean transform undersample

collect:
	$(PYTHON) src/DataCollection.py \
		--loan-output $(LOAN_DATA) \
		--indicators-output $(INDICATORS_DATA) \
		--merged-output $(MERGED_DATA)

clean:
	$(PYTHON) src/DataCleaning.py \
		--input $(MERGED_DATA) \
		--output $(CLEANED_DATA)

validate:
	$(PYTHON) src/DataValidation.py \
		--input $(MERGED_DATA)

transform:
	$(PYTHON) src/DataTransformation.py \
		--input $(CLEANED_DATA) \
		--output $(TRANSFORMED_DATA) \
		--train-output $(TRAIN_DATA) \
		--train-norm-output $(TRAIN_NORM_DATA) \
		--test-output $(TEST_DATA) \
		--business-stats $(BUSINESS_STATS)

undersample:
	$(PYTHON) src/DataUndersampling.py \
		--input $(TRAIN_NORM_DATA) \
		--output $(TRAIN_UNDERSAMPLED_DATA) \
		--target $(UNDERSAMPLE_TARGET) \
		--ratio $(UNDERSAMPLE_RATIO)

train:
	$(PYTHON) src/Model.py \
		--train-data $(TRAIN_UNDERSAMPLED_DATA) \
		--test-data $(TEST_DATA) \
		--business-stats $(BUSINESS_STATS) \
		--output-dir $(OUTPUT_DIR) \
		--random-search-iter $(RANDOM_SEARCH_ITER)

evaluate: train

test:
	$(PYTEST) --cov=src --cov-report=html $(TEST_DIR)

all: preprocess train test