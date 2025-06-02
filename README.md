# 👩‍⚕️🚑 Predictive Health Insurance Charge Calculator

A web-based application that intelligently predicts health insurance charges using machine learning. The calculator provides estimates based on user-inputted data, helping customers get an idea of their potential insurance costs.

## 💻 Local Flask Webapp Demo (R² Score: 0.8818)
![Demo](venv/assets/webDemo.png) 

## 🔎 Model Visualization
![Results](venv/assets/charges_distribution.png) 
![Results](venv/assets/correlation_matrix.png)
![Results](venv/assets/feature_relationships.png)

## ✨ Features
- **Machine Learning Model**: Trained model that predicts premiums based on various factors
- **User-Friendly Interface**: Simple web form for inputting vehicle and driver details
- **Instant Results**: Quick premium calculation after form submission

## 🛠️ Technologies Used
- **Frontend**: HTML, CSS, JavaScript
- **Backend**: Flask
- **Machine Learning**: XGBoost, scikit-learn
- **Data Processing**: Pandas, NumPy

## 📊 Dataset
The machine learning model was trained using the [Healthcare Insurance](https://www.kaggle.com/datasets/willianoliveiragibin/healthcare-insurance) by Willian Oliveira and Arun Jangir.

Dataset includes:
- Age
- Sex
- BMI
- Children
- Smoker
- Region
- Health Insurance Charges

## 🚀 Getting Started

### Prerequisites
- Python (version 3.x)
- pip install -r requirements.txt

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/alyssadong2006/Health-Insurance-Premium-Calculator
