
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import pandas as pd
from ..config import *

def train_model():
    df = pd.read_csv(PROCESSED_DATA_PATH)
    
    # CORRECTED: Only scale features, not target
    X = df.drop('charges', axis=1)
    y = df['charges']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    
    # Load scaler and fit ONLY to feature columns (excluding 'charges')
    numerical_cols = ['age', 'bmi', 'children']
    scaler = StandardScaler()
    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])
    
    # Save scaler that only knows about features
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