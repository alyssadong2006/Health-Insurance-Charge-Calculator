from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import os
#from config import MODEL_PATH, ENCODERS_PATH, SCALER_PATH
DATA_PATH = 'data/raw/insurance.csv'
PROCESSED_DATA_PATH = 'data/processed/processed_insurance.csv'
MODEL_PATH = 'model/insurance_model.joblib'
ENCODERS_PATH = 'model/label_encoders.joblib'
SCALER_PATH = 'model/scaler.joblib'

app = Flask(__name__)

def load_artifacts():
    """Load the trained model and preprocessing artifacts."""
    model = joblib.load(MODEL_PATH)
    encoders = joblib.load(ENCODERS_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, encoders, scaler

def preprocess_input(form_data, encoders, scaler):
    """Preprocess form input for prediction."""
    # Convert form data to DataFrame
    data = pd.DataFrame([form_data])
    
    # Create features
    data['age_group'] = pd.cut(data['age'].astype(int), 
                             bins=[0, 18, 30, 45, 60, 100],
                             labels=['0-18', '19-30', '31-45', '46-60', '60+'])
    data['bmi_category'] = pd.cut(data['bmi'].astype(float),
                                bins=[0, 18.5, 25, 30, 100],
                                labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
    data['has_children'] = data['children'].apply(lambda x: 1 if int(x) > 0 else 0)
    
    # Encode categoricals
    categorical_cols = ['sex', 'smoker', 'region', 'age_group', 'bmi_category']
    for col in categorical_cols:
        data[col] = encoders[col].transform(data[col])
    
    # Scale numericals
    numerical_cols = ['age', 'bmi', 'children']
    data[numerical_cols] = scaler.transform(data[numerical_cols].astype(float))
    
    return data

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # Get form data
            form_data = {
                'age': request.form['age'],
                'sex': request.form['sex'],
                'bmi': request.form['bmi'],
                'children': request.form['children'],
                'smoker': request.form['smoker'],
                'region': request.form['region']
            }
            
            # Load model and artifacts
            model, encoders, scaler = load_artifacts()
            
            # Preprocess and predict
            processed_data = preprocess_input(form_data, encoders, scaler)
            prediction = model.predict(processed_data)
            
            # Format result
            result = f"Predicted Insurance Premium: ${prediction[0]:,.2f}"
            return render_template('index.html', result=result, form_data=form_data)
        
        except Exception as e:
            error = f"Error processing request: {str(e)}"
            return render_template('index.html', error=error)
    
    return render_template('index.html')

if __name__ == '__main__':
    os.makedirs('app/static', exist_ok=True)
    os.makedirs('app/templates', exist_ok=True)
    app.run(debug=True)