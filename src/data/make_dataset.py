"""
Generate synthetic credit application data for model training and testing.

This script creates a synthetic dataset with features relevant to credit 
underwriting decisions. No real customer data is used.
"""

import argparse
import os
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from faker import Faker

np.random.seed(42)
fake = Faker()
Faker.seed(42)


def generate_synthetic_data(n_samples: int = 10000) -> pd.DataFrame:
    """
    Generate synthetic credit application data.
    
    Args:
        n_samples: Number of samples to generate
        
    Returns:
        DataFrame with synthetic credit application data
    """
    # Generate base features
    data = {
        'application_id': [f'APP{str(i).zfill(8)}' for i in range(n_samples)],
        'application_date': [
            fake.date_between(start_date='-2y', end_date='today') 
            for _ in range(n_samples)
        ],
        'age': np.random.randint(18, 75, n_samples),
        'annual_income': np.random.lognormal(mean=10.8, sigma=0.6, size=n_samples),
        'employment_length_years': np.random.exponential(scale=5, size=n_samples).clip(0, 40),
        'debt_to_income_ratio': np.random.beta(2, 5, n_samples) * 100,
        'credit_history_length_years': np.random.exponential(scale=8, size=n_samples).clip(0, 50),
        'num_open_credit_lines': np.random.poisson(lam=4, size=n_samples),
        'num_derogatory_marks': np.random.poisson(lam=0.5, size=n_samples),
        'total_revolving_balance': np.random.lognormal(mean=8.5, sigma=1.2, size=n_samples),
        'revolving_utilization': np.random.beta(2, 3, n_samples) * 100,
        'num_recent_inquiries': np.random.poisson(lam=1.5, size=n_samples),
        'loan_amount_requested': np.random.choice([5000, 10000, 15000, 20000, 25000, 30000], n_samples),
        'loan_purpose': np.random.choice(
            ['debt_consolidation', 'home_improvement', 'major_purchase', 'medical', 'other'],
            n_samples,
            p=[0.4, 0.2, 0.2, 0.1, 0.1]
        )
    }
    
    df = pd.DataFrame(data)
    
    # Generate target based on realistic credit risk factors
    # Higher risk with: low income, high DTI, short credit history, derogatory marks
    risk_score = (
        -0.3 * (df['annual_income'] / df['annual_income'].max()) +
        0.4 * (df['debt_to_income_ratio'] / 100) +
        -0.2 * (df['credit_history_length_years'] / df['credit_history_length_years'].max()) +
        0.3 * (df['num_derogatory_marks'] / (df['num_derogatory_marks'].max() + 1)) +
        0.2 * (df['revolving_utilization'] / 100) +
        0.1 * (df['num_recent_inquiries'] / (df['num_recent_inquiries'].max() + 1)) +
        np.random.normal(0, 0.2, n_samples)  # Add noise
    )
    
    # Convert risk score to binary default outcome (1 = default, 0 = no default)
    # Use sigmoid transformation to get probabilities
    default_prob = 1 / (1 + np.exp(-3 * (risk_score - risk_score.mean())))
    df['default'] = (np.random.random(n_samples) < default_prob).astype(int)
    
    # Round numeric columns
    df['annual_income'] = df['annual_income'].round(2)
    df['employment_length_years'] = df['employment_length_years'].round(1)
    df['debt_to_income_ratio'] = df['debt_to_income_ratio'].round(2)
    df['credit_history_length_years'] = df['credit_history_length_years'].round(1)
    df['total_revolving_balance'] = df['total_revolving_balance'].round(2)
    df['revolving_utilization'] = df['revolving_utilization'].round(2)
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description='Generate synthetic credit application data'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/sample/credit_applications.csv',
        help='Output path for the synthetic dataset'
    )
    parser.add_argument(
        '--n-samples',
        type=int,
        default=10000,
        help='Number of samples to generate'
    )
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Generate data
    print(f"Generating {args.n_samples} synthetic credit applications...")
    df = generate_synthetic_data(n_samples=args.n_samples)
    
    # Save to CSV
    df.to_csv(args.output, index=False)
    print(f"Saved synthetic data to {args.output}")
    print(f"Shape: {df.shape}")
    print(f"Default rate: {df['default'].mean():.2%}")
    print(f"\nFirst few rows:")
    print(df.head())


if __name__ == '__main__':
    main()
