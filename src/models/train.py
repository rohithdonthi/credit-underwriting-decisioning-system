"""
Train a baseline credit risk model with time-based validation split.

This script trains a logistic regression model on synthetic credit data,
using a time-based split to simulate real-world deployment scenarios.
"""

import argparse
import os
import pickle
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, 
    classification_report, 
    confusion_matrix,
    precision_recall_curve,
    roc_curve
)
import joblib


def load_and_prepare_data(data_path: str) -> pd.DataFrame:
    """Load and prepare the dataset."""
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    df['application_date'] = pd.to_datetime(df['application_date'])
    print(f"Loaded {len(df)} records")
    return df


def time_based_split(df: pd.DataFrame, train_ratio: float = 0.7):
    """
    Split data based on time (application_date).
    
    Args:
        df: DataFrame with 'application_date' column
        train_ratio: Proportion of data to use for training
        
    Returns:
        train_df, test_df
    """
    df_sorted = df.sort_values('application_date').reset_index(drop=True)
    split_idx = int(len(df_sorted) * train_ratio)
    
    train_df = df_sorted.iloc[:split_idx].copy()
    test_df = df_sorted.iloc[split_idx:].copy()
    
    print(f"\nTime-based split:")
    print(f"  Training: {len(train_df)} records ({train_df['application_date'].min()} to {train_df['application_date'].max()})")
    print(f"  Test: {len(test_df)} records ({test_df['application_date'].min()} to {test_df['application_date'].max()})")
    print(f"  Train default rate: {train_df['default'].mean():.2%}")
    print(f"  Test default rate: {test_df['default'].mean():.2%}")
    
    return train_df, test_df


def prepare_features(df: pd.DataFrame):
    """
    Prepare features for modeling.
    
    Returns:
        X (features), y (target), feature_names
    """
    # Numeric features
    numeric_features = [
        'age',
        'annual_income',
        'employment_length_years',
        'debt_to_income_ratio',
        'credit_history_length_years',
        'num_open_credit_lines',
        'num_derogatory_marks',
        'total_revolving_balance',
        'revolving_utilization',
        'num_recent_inquiries',
        'loan_amount_requested'
    ]
    
    # One-hot encode categorical features
    df_encoded = pd.get_dummies(df, columns=['loan_purpose'], prefix='purpose')
    
    # Get all feature columns (numeric + encoded categorical)
    purpose_cols = [col for col in df_encoded.columns if col.startswith('purpose_')]
    feature_cols = numeric_features + purpose_cols
    
    X = df_encoded[feature_cols].values
    y = df_encoded['default'].values
    
    return X, y, feature_cols


def train_model(X_train, y_train):
    """Train a baseline logistic regression model."""
    print("\nTraining logistic regression model...")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Train model with L2 regularization
    model = LogisticRegression(
        random_state=42,
        max_iter=1000,
        class_weight='balanced',  # Handle class imbalance
        C=1.0
    )
    model.fit(X_train_scaled, y_train)
    
    return model, scaler


def evaluate_model(model, scaler, X, y, dataset_name='Test'):
    """Evaluate model performance."""
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)
    y_pred_proba = model.predict_proba(X_scaled)[:, 1]
    
    print(f"\n{dataset_name} Set Performance:")
    print(f"  ROC-AUC Score: {roc_auc_score(y, y_pred_proba):.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y, y_pred, target_names=['No Default', 'Default']))
    print(f"\nConfusion Matrix:")
    print(confusion_matrix(y, y_pred))
    
    return y_pred_proba


def save_model(model, scaler, feature_names, output_dir: str):
    """Save the trained model and artifacts."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(output_dir, 'model.pkl')
    joblib.dump(model, model_path)
    print(f"\nSaved model to {model_path}")
    
    # Save scaler
    scaler_path = os.path.join(output_dir, 'scaler.pkl')
    joblib.dump(scaler, scaler_path)
    print(f"Saved scaler to {scaler_path}")
    
    # Save feature names
    feature_path = os.path.join(output_dir, 'features.pkl')
    joblib.dump(feature_names, feature_path)
    print(f"Saved feature names to {feature_path}")
    
    # Save metadata
    metadata = {
        'training_date': datetime.now().isoformat(),
        'model_type': 'LogisticRegression',
        'n_features': len(feature_names),
        'feature_names': feature_names
    }
    metadata_path = os.path.join(output_dir, 'metadata.pkl')
    joblib.dump(metadata, metadata_path)
    print(f"Saved metadata to {metadata_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Train credit risk model with time-based split'
    )
    parser.add_argument(
        '--data',
        type=str,
        default='data/sample/credit_applications.csv',
        help='Path to input data CSV'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='models',
        help='Directory to save trained model artifacts'
    )
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.7,
        help='Proportion of data to use for training (time-based split)'
    )
    
    args = parser.parse_args()
    
    # Load data
    df = load_and_prepare_data(args.data)
    
    # Time-based split
    train_df, test_df = time_based_split(df, train_ratio=args.train_ratio)
    
    # Prepare features
    X_train, y_train, feature_names = prepare_features(train_df)
    X_test, y_test, _ = prepare_features(test_df)
    
    print(f"\nFeature dimensions:")
    print(f"  Training: {X_train.shape}")
    print(f"  Test: {X_test.shape}")
    
    # Train model
    model, scaler = train_model(X_train, y_train)
    
    # Evaluate on both sets
    _ = evaluate_model(model, scaler, X_train, y_train, dataset_name='Training')
    _ = evaluate_model(model, scaler, X_test, y_test, dataset_name='Test')
    
    # Save model
    save_model(model, scaler, feature_names, args.output_dir)
    
    print("\n✓ Training complete!")


if __name__ == '__main__':
    main()
