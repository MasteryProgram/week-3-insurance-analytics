import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

def load_data_for_modeling(path="data/cleaned_insurance_data.csv"):
    """Load and prepare data for modeling"""
    df = pd.read_csv(path, low_memory=False)
    
    # Filter only policies with claims for severity modeling
    severity_df = df[df['TotalClaims'] > 0].copy()
    
    print(f"Original data shape: {df.shape}")
    print(f"Severity modeling data shape (claims > 0): {severity_df.shape}")
    
    return df, severity_df


def feature_engineering(df):
    """Feature Engineering"""
    df = df.copy()
    
    # Basic features
    df['VehicleAge'] = 2015 - df['RegistrationYear']
    df['ClaimFrequency'] = (df['TotalClaims'] > 0).astype(int)
    df['LogTotalPremium'] = np.log1p(df['TotalPremium'])
    
    return df


def prepare_features(df, target_col='TotalClaims', is_severity=True):
    """Prepare features for modeling"""
    df = feature_engineering(df)
    
    # Select features
    categorical_cols = ['Province', 'Gender', 'VehicleType', 'make', 'CoverType']
    numerical_cols = ['VehicleAge', 'SumInsured', 'TotalPremium', 'LogTotalPremium']
    
    X = df[categorical_cols + numerical_cols]
    y = df[target_col]
    
    # One-hot encoding
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    
    return X, y


def train_and_evaluate_models(X, y, test_size=0.2, random_state=42):
    """Train multiple models and compare"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=random_state),
        'XGBoost': XGBRegressor(n_estimators=100, random_state=random_state, verbosity=0)
    }
    
    results = {}
    
    print("\n" + "="*60)
    print("MODEL TRAINING & EVALUATION (Claim Severity)")
    print("="*60)
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {'MAE': mae, 'RMSE': rmse, 'R2': r2}
        
        print(f"\n{name}:")
        print(f"   MAE  : {mae:,.2f}")
        print(f"   RMSE : {rmse:,.2f}")
        print(f"   R²   : {r2:.4f}")
    
    return models, results, X_train, X_test, y_train, y_test


def save_best_model(models, results, output_path="models/"):
    """Save the best model"""
    os.makedirs(output_path, exist_ok=True)
    best_model_name = max(results, key=lambda x: results[x]['R2'])
    best_model = models[best_model_name]
    
    joblib.dump(best_model, f"{output_path}best_severity_model.pkl")
    print(f"\n✅ Best model ({best_model_name}) saved!")
    
    return best_model_name
