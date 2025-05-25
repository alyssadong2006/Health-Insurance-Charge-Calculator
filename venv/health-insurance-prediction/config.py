from pathlib import Path

"""
DATA_PATH = 'data/raw/insurance.csv'
PROCESSED_DATA_PATH = 'data/processed/processed_insurance.csv'
MODEL_PATH = 'model/insurance_model.joblib'
ENCODERS_PATH = 'model/label_encoders.joblib'
SCALER_PATH = 'model/scaler.joblib'
TEST_SIZE = 0.2
RANDOM_STATE = 42
"""

# Base directory setup
BASE_DIR = Path(__file__).parent

# Data paths
DATA_PATH = BASE_DIR / 'data' / 'raw' / 'insurance.csv'
PROCESSED_DATA_PATH = BASE_DIR / 'data' / 'processed' / 'processed_insurance.csv'

# Model paths
MODEL_PATH = BASE_DIR / 'model' / 'insurance_model.joblib'
ENCODERS_PATH = BASE_DIR / 'model' / 'label_encoders.joblib'
SCALER_PATH = BASE_DIR / 'model' / 'scaler.joblib'

# Model parameters
TEST_SIZE = 0.2
RANDOM_STATE = 42