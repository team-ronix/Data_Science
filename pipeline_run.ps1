poetry run python .\src\DataCollection.py
poetry run python .\src\DataCleaning.py
poetry run python .\src\DataTransformation.py
poetry run python .\src\DataUndersampling.py --input data/train_norm.csv --output data/train_undersampled.csv --target loan_status --ratio 1.0
poetry run python .\src\Model.py