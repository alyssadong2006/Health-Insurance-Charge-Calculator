"""
model_training.py
Reads the processed data and trains/tests a model with it.
"""
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import pandas as pd
from ..config import *

def train_model():
    # Opens the cleaned-up date
    df = pd.read_csv(PROCESSED_DATA_PATH)
    
    # X = All the information we'll use to make predictions
    #   = independent variables
    X = df.drop('charges', axis=1)
    # Y = What we want to predict
    #   = dependent variables
    y = df['charges']
    
    # Splits data to two sections (train and test)
    # 80% goes to training the model, 20% goes to testing the model
    # random_state : ensures the split is the same every time
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    
    # Adjusting Number Scales
    numerical_cols = ['age', 'bmi', 'children']
    scaler = StandardScaler()
    # Learns the scaling from training data
    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    # Applies the same scaling to test data
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])
    
    # Saves scaler
    joblib.dump(scaler, SCALER_PATH)
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Model Evaluation:")
    print(f"- MAE: ${mae:,.2f}")
    print(f"- R² Score: {r2:.4f}")
    
    # Save model
    joblib.dump(model, MODEL_PATH)
    print("Model retrained with correct features")

if __name__ == "__main__":
    train_model()