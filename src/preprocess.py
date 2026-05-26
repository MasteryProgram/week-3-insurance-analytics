import pandas as pd
import numpy as np
from datetime import datetime

def load_raw_data(data_path="data/insurance_data.csv"):
    """Load the raw insurance data with proper handling of South African number format"""
    print("📂 Loading data with comma decimal separator handling...")
    
    # Read the data as string first to handle cleaning
    df = pd.read_csv(data_path, low_memory=False)
    
    # Columns that may have comma as decimal separator
    numeric_cols_with_comma = ['TotalPremium', 'TotalClaims', 'SumInsured', 
                               'CustomValueEstimate', 'CapitalOutstanding']
    
    for col in numeric_cols_with_comma:
        if col in df.columns:
            # Replace comma with dot and convert to float
            df[col] = df[col].astype(str).str.replace(',', '.').str.strip()
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    print(f"✅ Data loaded successfully. Shape: {df.shape}")
    return df

def clean_data(df):
    """Clean and preprocess the insurance dataset"""
    
    print("🔄 Starting data cleaning...")
    df_clean = df.copy()
    
    # 1. Basic info
    print(f"Original shape: {df_clean.shape}")
    
    # 2. Handle missing values
    missing = df_clean.isnull().sum()
    print(f"Missing values before cleaning:\n{missing[missing > 0]}")
    
    # Fill missing values with appropriate strategies
    df_clean['Gender'] = df_clean['Gender'].fillna('Unknown')
    df_clean['MaritalStatus'] = df_clean['MaritalStatus'].fillna('Unknown')
    df_clean['Province'] = df_clean['Province'].fillna('Unknown')
    
    # Numerical columns - fill with median
    num_cols = ['TotalPremium', 'TotalClaims', 'SumInsured', 'CustomValueEstimate']
    for col in num_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    
    # 3. Convert data types
    if 'TransactionMonth' in df_clean.columns:
        df_clean['TransactionMonth'] = pd.to_datetime(df_clean['TransactionMonth'])
    
    # 4. Create useful features
    df_clean['LossRatio'] = df_clean['TotalClaims'] / df_clean['TotalPremium'].replace(0, np.nan)
    df_clean['Margin'] = df_clean['TotalPremium'] - df_clean['TotalClaims']
    df_clean['ClaimFrequency'] = (df_clean['TotalClaims'] > 0).astype(int)
    
    # Vehicle Age
    if 'RegistrationYear' in df_clean.columns:
        df_clean['VehicleAge'] = 2015 - df_clean['RegistrationYear']  # Since data is up to 2015
    
    print(f"✅ Cleaning completed. New shape: {df_clean.shape}")
    
    return df_clean

def save_clean_data(df_clean, output_path="data/cleaned_insurance_data.csv"):
    """Save cleaned data"""
    df_clean.to_csv(output_path, index=False)
    print(f"✅ Cleaned data saved to: {output_path}")

if __name__ == "__main__":
    df = load_raw_data()
    df_clean = clean_data(df)
    save_clean_data(df_clean)