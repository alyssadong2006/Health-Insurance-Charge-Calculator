import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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

# Update load_data() to:
def load_data():
    """Load the processed data."""
    return pd.read_csv(PROCESSED_DATA_PATH)

def plot_correlation_matrix(df):
    """Plot correlation matrix heatmap."""
    plt.figure(figsize=(12, 8))
    corr = df.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png')
    plt.close()

def plot_charges_distribution(df):
    """Plot distribution of insurance charges."""
    plt.figure(figsize=(10, 6))
    sns.histplot(df['charges'], kde=True)
    plt.title('Distribution of Insurance Charges')
    plt.xlabel('Charges')
    plt.ylabel('Frequency')
    plt.savefig('charges_distribution.png')
    plt.close()

def plot_feature_relationships(df):
    """Plot relationships between features and charges."""
    numerical_features = ['age', 'bmi', 'children']
    
    plt.figure(figsize=(15, 5))
    for i, feature in enumerate(numerical_features, 1):
        plt.subplot(1, 3, i)
        sns.scatterplot(x=df[feature], y=df['charges'])
        plt.title(f'{feature} vs Charges')
    
    plt.tight_layout()
    plt.savefig('feature_relationships.png')
    plt.close()

def create_visualizations():
    """Generate all visualizations."""
    df = load_data()
    plot_correlation_matrix(df)
    plot_charges_distribution(df)
    plot_feature_relationships(df)
    print("Visualizations created and saved as PNG files.")

if __name__ == "__main__":
    create_visualizations()