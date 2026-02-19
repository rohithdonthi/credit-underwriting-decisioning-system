"""
Unit tests for synthetic data generator.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.synthetic_generator import generate_synthetic_credit_data


def test_generate_synthetic_data_basic():
    """Test basic data generation."""
    df = generate_synthetic_credit_data(n_samples=100, default_rate=0.15, random_state=42)
    
    assert len(df) == 100
    assert 'application_id' in df.columns
    assert 'default' in df.columns
    assert 'credit_score' in df.columns


def test_generate_synthetic_data_default_rate():
    """Test that default rate is approximately correct."""
    df = generate_synthetic_credit_data(n_samples=1000, default_rate=0.20, random_state=42)
    
    actual_rate = df['default'].mean()
    # Allow 5% tolerance
    assert 0.15 <= actual_rate <= 0.25


def test_generate_synthetic_data_features():
    """Test that all expected features are present."""
    df = generate_synthetic_credit_data(n_samples=100, random_state=42)
    
    expected_features = [
        'application_id',
        'credit_score',
        'num_credit_lines',
        'credit_utilization',
        'num_delinquencies',
        'months_since_last_delinq',
        'annual_income',
        'employment_length_years',
        'debt_to_income',
        'loan_amount',
        'loan_term_months',
        'interest_rate',
        'num_inquiries_6m',
        'revolving_balance',
        'has_mortgage',
        'has_car_loan',
        'application_date',
        'default'
    ]
    
    for feature in expected_features:
        assert feature in df.columns


def test_generate_synthetic_data_ranges():
    """Test that feature values are in expected ranges."""
    df = generate_synthetic_credit_data(n_samples=1000, random_state=42)
    
    # Credit score should be 300-850
    assert df['credit_score'].min() >= 300
    assert df['credit_score'].max() <= 850
    
    # Credit utilization should be 0-1
    assert df['credit_utilization'].min() >= 0
    assert df['credit_utilization'].max() <= 1
    
    # Default should be 0 or 1
    assert df['default'].isin([0, 1]).all()


def test_generate_synthetic_data_reproducibility():
    """Test that same seed produces same data."""
    df1 = generate_synthetic_credit_data(n_samples=100, random_state=42)
    df2 = generate_synthetic_credit_data(n_samples=100, random_state=42)
    
    pd.testing.assert_frame_equal(df1, df2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
