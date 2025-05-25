import pandas as pd
import joblib
from app import preprocess_input  # Import your preprocessing function

# Load artifacts
MODEL_PATH = 'model/insurance_model.joblib'
ENCODERS_PATH = 'model/label_encoders.joblib'
SCALER_PATH = 'model/scaler.joblib'

model = joblib.load(MODEL_PATH)
encoders = joblib.load(ENCODERS_PATH)
scaler = joblib.load(SCALER_PATH)

# Manual test case (matches your form fields)
test_case = {
    'age': 35,
    'sex': 'male',
    'bmi': 28.5,
    'children': 2,
    'smoker': 'no',
    'region': 'southeast'
}

# Debug: Print expected features
print("\n=== Model Features ===")
print(model.feature_names_in_)

# Preprocess the test case
try:
    processed_data = preprocess_input(test_case, encoders, scaler, model)
    
    # Debug: Show processed data
    print("\n=== Processed Data ===")
    print(processed_data)
    print("\nFeature match check:")
    print("Expected:", model.feature_names_in_)
    print("Actual:", processed_data.columns.tolist())
    
    # Make prediction
    prediction = model.predict(processed_data)
    print(f"\nPredicted charges: ${prediction[0]:,.2f}")
    
except Exception as e:
    print(f"\nError: {str(e)}")
    print("\n=== Debug Info ===")
    print("Encoders available:", list(encoders.keys()))
    print("Scaler features:", scaler.feature_names_in_ if hasattr(scaler, 'feature_names_in_') else "No feature names in scaler")