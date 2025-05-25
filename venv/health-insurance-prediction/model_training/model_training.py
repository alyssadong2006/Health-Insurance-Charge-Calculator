from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import pandas as pd
# from config import *
# Configuration constants remain the same
DATA_PATH = 'data/raw/insurance.csv'
PROCESSED_DATA_PATH = 'data/processed/processed_insurance.csv'
MODEL_PATH = 'model/insurance_model.joblib'
ENCODERS_PATH = 'model/label_encoders.joblib'
SCALER_PATH = 'model/scaler.joblib'

# Model parameters
TEST_SIZE = 0.2
RANDOM_STATE = 42

def load_processed_data():
    """Load the processed data."""
    return pd.read_csv(PROCESSED_DATA_PATH)

def train_model():
    """Train and save the insurance prediction model."""
    df = load_processed_data()
    
    # Split data
    X = df.drop('charges', axis=1)
    y = df['charges']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
    print(f"R2 Score: {r2_score(y_test, y_pred):.2f}")
    
    # Save model
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train_model()