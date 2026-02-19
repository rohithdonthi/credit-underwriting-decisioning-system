"""
Synthetic credit dataset generator for underwriting decisioning system.
Generates realistic credit application data with features and default labels.
"""

import numpy as np
import pandas as pd
from typing import Optional


def generate_synthetic_credit_data(
    n_samples: int = 10000,
    default_rate: float = 0.15,
    random_state: Optional[int] = 42
) -> pd.DataFrame:
    """
    Generate synthetic credit application dataset.
    
    Args:
        n_samples: Number of samples to generate
        default_rate: Target default rate (approximate)
        random_state: Random seed for reproducibility
        
    Returns:
        DataFrame with features and target variable
    """
    np.random.seed(random_state)
    
    # Generate features
    data = {}
    
    # Credit history features
    data['credit_score'] = np.random.normal(680, 80, n_samples).clip(300, 850)
    data['num_credit_lines'] = np.random.poisson(5, n_samples).clip(0, 30)
    data['credit_utilization'] = np.random.beta(2, 5, n_samples)
    data['num_delinquencies'] = np.random.poisson(0.5, n_samples).clip(0, 10)
    data['months_since_last_delinq'] = np.where(
        data['num_delinquencies'] > 0,
        np.random.exponential(24, n_samples).clip(0, 120),
        -1
    )
    
    # Income and employment
    data['annual_income'] = np.random.lognormal(10.8, 0.6, n_samples).clip(15000, 500000)
    data['employment_length_years'] = np.random.exponential(5, n_samples).clip(0, 40)
    data['debt_to_income'] = np.random.beta(3, 5, n_samples).clip(0, 1)
    
    # Loan characteristics
    data['loan_amount'] = np.random.lognormal(9.5, 0.8, n_samples).clip(1000, 100000)
    data['loan_term_months'] = np.random.choice([12, 24, 36, 48, 60], n_samples)
    data['interest_rate'] = np.random.normal(12, 4, n_samples).clip(5, 25)
    
    # Additional features
    data['num_inquiries_6m'] = np.random.poisson(1, n_samples).clip(0, 10)
    data['revolving_balance'] = np.random.lognormal(8.5, 1.2, n_samples).clip(0, 100000)
    data['has_mortgage'] = np.random.binomial(1, 0.35, n_samples)
    data['has_car_loan'] = np.random.binomial(1, 0.40, n_samples)
    
    df = pd.DataFrame(data)
    
    # Generate target variable (default) with realistic relationships
    # Calculate default probability based on features
    default_logit = (
        -5.0
        + (850 - df['credit_score']) / 50  # Lower credit score increases risk
        + df['num_delinquencies'] * 0.3
        + df['credit_utilization'] * 2.0
        + df['debt_to_income'] * 3.0
        + df['num_inquiries_6m'] * 0.2
        - df['employment_length_years'] / 10
        + np.random.normal(0, 0.5, n_samples)  # Add noise
    )
    
    default_prob = 1 / (1 + np.exp(-default_logit))
    
    # Adjust probabilities to match target default rate
    prob_threshold = np.percentile(default_prob, (1 - default_rate) * 100)
    df['default'] = (default_prob > prob_threshold).astype(int)
    
    # Add application date for time-based splitting
    start_date = pd.Timestamp('2023-01-01')
    date_range = pd.date_range(start_date, periods=n_samples, freq='h')
    df['application_date'] = np.random.choice(date_range, n_samples, replace=True)
    df = df.sort_values('application_date').reset_index(drop=True)
    
    # Add application ID
    df.insert(0, 'application_id', [f'APP{str(i).zfill(8)}' for i in range(1, n_samples + 1)])
    
    return df


def main():
    """Generate and save sample dataset."""
    print("Generating synthetic credit dataset...")
    df = generate_synthetic_credit_data(n_samples=10000, default_rate=0.15)
    
    print(f"Generated {len(df)} samples")
    print(f"Default rate: {df['default'].mean():.2%}")
    print(f"\nFeature statistics:")
    print(df.describe())
    
    # Save to CSV
    output_path = "data/sample/credit_applications.csv"
    df.to_csv(output_path, index=False)
    print(f"\nDataset saved to {output_path}")


if __name__ == "__main__":
    main()
