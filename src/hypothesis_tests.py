
import pandas as pd
import numpy as np
from scipy import stats

def load_data(path="../data/cleaned_insurance_data.csv"):
    """Load cleaned data"""
    df = pd.read_csv(path, low_memory=False)
    # Ensure key derived columns exist
    if 'Margin' not in df.columns:
        df['Margin'] = df['TotalPremium'] - df['TotalClaims']
    if 'HasClaim' not in df.columns:
        df['HasClaim'] = (df['TotalClaims'] > 0).astype(int)
    return df


def chi2_claim_frequency_test(df, group_col, group_a, group_b):
    """Chi-squared test for Claim Frequency"""
    contingency = pd.crosstab(df[group_col], df['HasClaim'])
    contingency = contingency.loc[[group_a, group_b]]
    
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
    
    print(f"\n{'='*60}")
    print(f"CHI-SQUARED TEST: Claim Frequency - {group_col}")
    print(f"Comparing: {group_a} vs {group_b}")
    print(f"{'='*60}")
    print(f"Chi2 Statistic : {chi2:.4f}")
    print(f"p-value        : {p_value:.6f}")
    print("Decision       :", "Reject H0 (Significant Difference)" if p_value < 0.05 else "Fail to reject H0")
    print(f"{'='*60}")
    return p_value, chi2


def ttest_margin(df, group_col, group_a, group_b):
    """Independent T-test for Margin"""
    group1 = df[df[group_col] == group_a]['Margin']
    group2 = df[df[group_col] == group_b]['Margin']
    
    t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=False)
    
    print(f"\n{'='*60}")
    print(f"T-TEST: Margin Difference - {group_col}")
    print(f"Comparing: {group_a} vs {group_b}")
    print(f"{'='*60}")
    print(f"p-value  : {p_value:.6f}")
    print("Decision :", "Reject H0 (Significant Difference)" if p_value < 0.05 else "Fail to reject H0")
    print(f"{'='*60}")
    return p_value


def run_hypothesis_tests(df):
    """Run all main hypothesis tests"""
    print("Running Hypothesis Tests for Task 3...\n")
    
    results = {}
    
    # Test 1: Provinces (Claim Frequency)
    p1, _ = chi2_claim_frequency_test(df, 'Province', 'Gauteng', 'Western Cape')
    results['Province_ClaimFreq'] = p1
    
    # Test 4: Gender (Claim Frequency)
    p4, _ = chi2_claim_frequency_test(df, 'Gender', 'Male', 'Female')
    results['Gender_ClaimFreq'] = p4
    
    # Test 3: Margin difference (Provinces)
    p3 = ttest_margin(df, 'Province', 'Gauteng', 'Western Cape')
    results['Province_Margin'] = p3
    
    return results
