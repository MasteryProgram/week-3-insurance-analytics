import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_prepare_data(path='../data/insurance_data.csv'):
    """Load and do initial cleaning"""
    df = pd.read_csv(path, low_memory=False)
    df['LossRatio'] = df['TotalClaims'] / df['TotalPremium'].replace(0, pd.NA)
    df['Margin'] = df['TotalPremium'] - df['TotalClaims']
    print(f"✅ Data Loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
    return df


def portfolio_summary(df):
    """Overall portfolio metrics"""
    total_premium = df['TotalPremium'].sum()
    total_claims = df['TotalClaims'].sum()
    overall_loss_ratio = (total_claims / total_premium * 100).round(2) if total_premium > 0 else 0
    
    print("=== PORTFOLIO SUMMARY ===")
    print(f"Total Premium      : {total_premium:,.2f} ZAR")
    print(f"Total Claims       : {total_claims:,.2f} ZAR")
    print(f"Overall Loss Ratio : {overall_loss_ratio}%")
    print(f"Total Margin       : {(total_premium - total_claims):,.2f} ZAR")
    return overall_loss_ratio


def analyze_by_group(df, group_col, top_n=10):
    """General function for group analysis"""
    result = df.groupby(group_col).agg(
        TotalPremium=('TotalPremium', 'sum'),
        TotalClaims=('TotalClaims', 'sum'),
        AvgLossRatio=('LossRatio', 'mean'),
        PolicyCount=('TotalPremium', 'count')
    ).round(4)
    
    result = result.sort_values('AvgLossRatio', ascending=False)
    print(f"\n=== Top {top_n} {group_col} by Loss Ratio ===")
    print(result.head(top_n))
    return result


def plot_loss_ratio_by_group(df, group_col, title=None, figsize=(12, 6)):
    """Plot loss ratio by any categorical variable"""
    group_data = df.groupby(group_col)['LossRatio'].mean().sort_values(ascending=False).head(10)
    
    plt.figure(figsize=figsize)
    sns.barplot(x=group_data.index, y=group_data.values, palette="viridis")
    plt.title(title or f'Average Loss Ratio by {group_col}', fontsize=14, fontweight='bold')
    plt.xlabel(group_col)
    plt.ylabel('Average Loss Ratio')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_distributions(df):
    """Plot key distributions"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    sns.histplot(df['TotalPremium'], bins=50, kde=True, ax=axes[0,0])
    axes[0,0].set_title('Distribution of TotalPremium')
    
    claims_only = df[df['TotalClaims'] > 0]
    sns.histplot(claims_only['TotalClaims'], bins=50, kde=True, ax=axes[0,1])
    axes[0,1].set_title('Distribution of TotalClaims (Claims > 0)')
    
    sns.boxplot(x=df['TotalPremium'], ax=axes[1,0])
    axes[1,0].set_title('Boxplot - TotalPremium')
    
    sns.boxplot(x=claims_only['TotalClaims'], ax=axes[1,1])
    axes[1,1].set_title('Boxplot - TotalClaims')
    
    plt.tight_layout()
    plt.show()


def correlation_analysis(df):
    """Correlation heatmap - robust version"""
    num_cols = ['TotalPremium', 'TotalClaims', 'LossRatio', 'Margin', 
                'CustomValueEstimate', 'SumInsured', 'CalculatedPremiumPerTerm']
    print("\n=== updated Correlation Analysis ===")
    # Filter existing columns
    num_cols = [col for col in num_cols if col in df.columns]
    
    corr_df = df[num_cols].copy()
    
    # Convert to numeric and handle missing values safely
    for col in corr_df.columns:
        corr_df[col] = pd.to_numeric(corr_df[col], errors='coerce')
    
    # Fill NaNs with 0 for correlation (common in insurance analysis)
    corr_matrix = corr_df.fillna(0).corr()
    
    # Plot
    plt.figure(figsize=(11, 9))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Optional: hide upper triangle
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', 
                linewidths=0.5, mask=mask, vmin=-1, vmax=1)
    plt.title('Correlation Matrix of Key Variables', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    return corr_matrix