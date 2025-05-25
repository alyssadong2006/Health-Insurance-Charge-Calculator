import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os
DATA_PATH = 'data/raw/insurance.csv'
PROCESSED_DATA_PATH = 'data/processed/processed_insurance.csv'
MODEL_PATH = 'model/insurance_model.joblib'
ENCODERS_PATH = 'model/label_encoders.joblib'
SCALER_PATH = 'model/scaler.joblib'
TEST_SIZE = 0.2
RANDOM_STATE = 42


def load_data():
    """Load and clean the insurance data."""
    df = pd.read_csv('data/raw/insurance.csv')
    return df

def clean_data(df):
    """Clean the dataset."""
    # Remove duplicates and missing values
    df = df.drop_duplicates().dropna()
    
    # Create new features
    df['age_group'] = pd.cut(df['age'], 
                           bins=[0, 18, 30, 45, 60, 100],
                           labels=['0-18', '19-30', '31-45', '46-60', '60+'])
    df['bmi_category'] = pd.cut(df['bmi'],
                               bins=[0, 18.5, 25, 30, 100],
                               labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
    df['has_children'] = df['children'].apply(lambda x: 1 if x > 0 else 0)
    return df

def transform_data(df):
    """Transform and scale the data."""
    # Encode categorical variables
    categorical_cols = ['sex', 'smoker', 'region', 'age_group', 'bmi_category']
    label_encoders = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    
    # Scale numerical features
    numerical_cols = ['age', 'bmi', 'children']  # NOT including 'charges'
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    # Save scaler
    joblib.dump(scaler, SCALER_PATH)
    
    return df

def save_processed_data(df):
    """Save the processed data."""
    df.to_csv('data/processed/processed_insurance.csv', index=False)

def process_data():
    """Main data processing function."""
    df = load_data()
    df = clean_data(df)
    df = transform_data(df)
    save_processed_data(df)
    return df

if __name__ == "__main__":
    process_data()