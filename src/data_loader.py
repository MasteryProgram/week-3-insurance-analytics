
import pandas as pd

def load_data(path='../data/insurance_data.csv'):
    """Load the insurance dataset"""
    df = pd.read_csv(path, low_memory=False)
    print(f"✅ Data Loaded: {df.shape[0]:,} rows and {df.shape[1]} columns")
    return df


def basic_info(df):
    """Print basic information"""
    print("\n=== Dataset Info ===")
    print(df.info())
    print("\n=== Missing Values (%) ===")
    print((df.isnull().sum() / len(df) * 100).round(2))
    print("\n=== First 3 Rows ===")
    print(df.head(3))