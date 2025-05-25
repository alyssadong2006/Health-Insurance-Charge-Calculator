import pandas as pd
import joblib
from ..config import *

# Model parameters
TEST_SIZE = 0.2
RANDOM_STATE = 42

def load_artifacts():
    """Load the trained model and preprocessing artifacts."""
    model = joblib.load(MODEL_PATH)
    encoders = joblib.load(ENCODERS_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, encoders, scaler

def preprocess_input(data, encoders, scaler):
    """Preprocess new input data."""
    # Convert to DataFrame if not already
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame([data])
    
    # Create features
    data['age_group'] = pd.cut(data['age'], 
                             bins=[0, 18, 30, 45, 60, 100],
                             labels=['0-18', '19-30', '31-45', '46-60', '60+'])
    data['bmi_category'] = pd.cut(data['bmi'],
                                bins=[0, 18.5, 25, 30, 100],
                                labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
    data['has_children'] = data['children'].apply(lambda x: 1 if x > 0 else 0)
    
    # Encode categoricals
    categorical_cols = ['sex', 'smoker', 'region', 'age_group', 'bmi_category']
    for col in categorical_cols:
        data[col] = encoders[col].transform(data[col])
    
    # Scale numericals
    numerical_cols = ['age', 'bmi', 'children']
    data[numerical_cols] = scaler.transform(data[numerical_cols])
    
    return data

def predict_charges(input_data):
    """Predict insurance charges for new data."""
    model, encoders, scaler = load_artifacts()
    processed_data = preprocess_input(input_data, encoders, scaler)
    prediction = model.predict(processed_data)
    return prediction[0]

if __name__ == "__main__":
    # Example prediction
    sample_input = {
        'age': 35,
        'sex': 'male',
        'bmi': 28,
        'children': 2,
        'smoker': 'no',
        'region': 'southeast'
    }
    prediction = predict_charges(sample_input)
    print(f"Predicted insurance charges: ${prediction:.2f}")