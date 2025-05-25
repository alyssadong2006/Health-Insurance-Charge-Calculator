import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
from ..config import *


def load_data():
    """Load Raw Insurance Data"""
    df = pd.read_csv('data/raw/insurance.csv')
    return df

def clean_data(df):
    """Clean and Improve the Dataset"""
    # Remove duplicates and missing values:
    df = df.drop_duplicates().dropna()
    
    # Create new features age group, bmi_category, has_children:
    # reads through 'age' values and sorts then to different labels using bin values
    df['age_group'] = pd.cut(df['age'], 
                           bins=[0, 18, 30, 45, 60, 100],
                           labels=['0-18', '19-30', '31-45', '46-60', '60+'])
    # reads through 'bmi' values and sorts them to different labels using bin values
    df['bmi_category'] = pd.cut(df['bmi'],
                               bins=[0, 18.5, 25, 30, 100],
                               labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
    # reads through 'children' values and gives them 1/0 (True/False) values
    df['has_children'] = df['children'].apply(lambda x: 1 if x > 0 else 0)

    return df

def transform_data(df):
    """Transform and Scale the Data"""
    # Makes a list of columns that contain words/categories
    categorical_cols = ['sex', 'smoker', 'region', 'age_group', 'bmi_category']
    label_encoders = {}
    
    # for each category column:
    #   - create a translator that turns words into numbers
    #   - replaces the words in the data with these numbers
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    
    # Scale numerical features:
    #   - makes sure all numbers are on the same scale
    numerical_cols = ['age', 'bmi', 'children']
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    # Save the scaler, the "conversion rules" that the program used
    # such that it can be used to perform translations on new data later
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