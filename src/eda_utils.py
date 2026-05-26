import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_clean_data(path="../data/cleaned_insurance_data.csv"):
    """Load the cleaned dataset"""
    df = pd.read_csv(path, low_memory=False)
    print(f"✅ Cleaned data loaded. Shape: {df.shape}")
    return df

def data_summary(df):
    """Data summarization and dtypes check"""
    print("=== Data Summary ===")
    print(f"Shape: {df.shape}")
    print("\nData Types Count:")
    print(df.dtypes.value_counts())
    print("\nNumerical Features Summary:")
    print(df.describe().round(2))
    return df.describe()

def check_missing_values(df):
    """Data Quality Assessment"""
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    print(f"\nMissing Values:\n{missing}")
    return missing

def univariate_analysis(df):
    """Univariate Analysis - Histograms and Bar Charts"""
    print("\n=== Univariate Analysis ===")
    
    # Numerical columns
    num_cols = ['TotalPremium', 'TotalClaims', 'SumInsured', 'CustomValueEstimate', 'LossRatio']
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.ravel()
    
    for i, col in enumerate(num_cols[:5]):
        if col in df.columns:
            sns.histplot(data=df, x=col, bins=50, ax=axes[i])
            axes[i].set_title(f'Distribution of {col}')
    
    plt.tight_layout()
    plt.show()
    
    # Categorical columns
    cat_cols = ['Province', 'Gender', 'VehicleType', 'make']
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.ravel()
    
    for i, col in enumerate(cat_cols):
        if col in df.columns:
            top_categories = df[col].value_counts().head(10)
            sns.barplot(x=top_categories.values, y=top_categories.index, ax=axes[i])
            axes[i].set_title(f'Top 10 {col}')
    
    plt.tight_layout()
    plt.show()

def outlier_detection(df):
    """Outlier Detection using Box Plots"""
    print("\n=== Outlier Detection ===")
    key_cols = ['TotalPremium', 'TotalClaims', 'SumInsured', 'CustomValueEstimate']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.ravel()
    
    for i, col in enumerate(key_cols):
        if col in df.columns:
            sns.boxplot(data=df, y=col, ax=axes[i])
            axes[i].set_title(f'Box Plot of {col}')
    
    plt.tight_layout()
    plt.show()

def bivariate_analysis(df):
    """Bivariate & Multivariate Analysis"""
    print("\n=== Bivariate Analysis ===")
    
    # Scatter plot: TotalPremium vs TotalClaims
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=df, x='TotalPremium', y='TotalClaims', alpha=0.5)
    plt.title('TotalPremium vs TotalClaims')
    plt.show()
    
    # Correlation Matrix (numerical features)
    num_features = ['TotalPremium', 'TotalClaims', 'SumInsured', 'CustomValueEstimate', 
                   'LossRatio', 'Margin', 'VehicleAge']
    corr = df[num_features].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Matrix')
    plt.show()

def geographic_trends(df):
    """Geographic Trends Analysis"""
    print("\n=== Geographic Trends ===")
    
    # Premium by Province
    province_premium = df.groupby('Province')['TotalPremium'].mean().sort_values(ascending=False)
    print("Average Premium by Province:")
    print(province_premium.head(10))
    
    # Loss Ratio by Province
    province_lr = df.groupby('Province').agg({
        'TotalPremium': 'sum',
        'TotalClaims': 'sum'
    }).assign(LossRatio=lambda x: (x['TotalClaims']/x['TotalPremium']*100).round(2))
    
    print("\nLoss Ratio by Province:")
    print(province_lr[['LossRatio']].sort_values('LossRatio', ascending=False))
    
    # Visualizations
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    
    sns.barplot(x=province_premium.head(8).index, y=province_premium.head(8).values, ax=axes[0])
    axes[0].set_title('Average Premium by Province')
    axes[0].tick_params(axis='x', rotation=45)
    
    sns.barplot(x=province_lr['LossRatio'].head(8).index, y=province_lr['LossRatio'].head(8).values, ax=axes[1])
    axes[1].set_title('Loss Ratio by Province (%)')
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()

def calculate_loss_ratio(df):
    """Overall Loss Ratio and Grouped Analysis"""
    overall_lr = (df['TotalClaims'].sum() / df['TotalPremium'].sum() * 100).round(2)
    print(f"\nOverall Portfolio Loss Ratio: {overall_lr}%")
    
    return overall_lr

def top_claims_analysis(df):
    """Enhanced analysis of vehicle makes with clear printed output"""
    print("\n" + "="*60)
    print("🚗 VEHICLE MAKES - CLAIMS ANALYSIS")
    print("="*60)
    
    make_claims = df.groupby('make').agg({
        'TotalClaims': ['count', 'sum', 'mean', 'max']
    }).round(2)
    
    make_claims.columns = ['Policy_Count', 'Total_Claims', 'Avg_Claim', 'Max_Claim']
    
    # Top 10 by Total Claims
    print("\n🔥 TOP 10 VEHICLE MAKES BY TOTAL CLAIMS:")
    top_total = make_claims.sort_values('Total_Claims', ascending=False).head(10)
    print(top_total)
    
    # Top 10 by Average Claim Amount
    print("\n💰 TOP 10 VEHICLE MAKES BY AVERAGE CLAIM AMOUNT:")
    top_avg = make_claims[make_claims['Policy_Count'] >= 100].sort_values('Avg_Claim', ascending=False).head(10)
    print(top_avg)
    
    # Lowest risk makes (by average claim)
    print("\n✅ LOWEST RISK MAKES (by Average Claim - min 100 policies):")
    lowest_risk = make_claims[make_claims['Policy_Count'] >= 100].sort_values('Avg_Claim').head(10)
    print(lowest_risk)
    
    # Summary insight
    print("\n" + "-"*50)
    print("📊 SUMMARY:")
    print(f"• Total unique vehicle makes: {len(make_claims)}")
    print(f"• Highest average claim: {make_claims['Avg_Claim'].max():,.2f} Rand")
    print(f"• Lowest average claim: {make_claims['Avg_Claim'].min():,.2f} Rand")
    
    return make_claims