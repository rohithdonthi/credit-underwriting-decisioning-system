"""
Batch scoring script for credit risk predictions.

This script loads a trained model and generates predictions on new data.
"""

import argparse
import os
import pandas as pd
import joblib
from datetime import datetime


def load_model_artifacts(model_dir: str):
    """Load model, scaler, and feature configuration."""
    print(f"Loading model artifacts from {model_dir}...")
    
    model_path = os.path.join(model_dir, 'model.pkl')
    scaler_path = os.path.join(model_dir, 'scaler.pkl')
    features_path = os.path.join(model_dir, 'features.pkl')
    metadata_path = os.path.join(model_dir, 'metadata.pkl')
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    feature_names = joblib.load(features_path)
    metadata = joblib.load(metadata_path)
    
    print(f"Loaded model: {metadata['model_type']}")
    print(f"Training date: {metadata['training_date']}")
    print(f"Number of features: {metadata['n_features']}")
    
    return model, scaler, feature_names, metadata


def prepare_input_features(df: pd.DataFrame, feature_names: list):
    """
    Prepare input features to match training format.
    
    Args:
        df: Input DataFrame
        feature_names: List of feature names expected by the model
        
    Returns:
        X (feature matrix matching training format)
    """
    # Numeric features that should exist
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
    
    # One-hot encode loan_purpose if it exists
    if 'loan_purpose' in df.columns:
        df_encoded = pd.get_dummies(df, columns=['loan_purpose'], prefix='purpose')
    else:
        df_encoded = df.copy()
    
    # Ensure all required features exist (add missing ones with 0)
    for feature in feature_names:
        if feature not in df_encoded.columns:
            df_encoded[feature] = 0
    
    # Select features in the correct order
    X = df_encoded[feature_names].values
    
    return X


def score_batch(model, scaler, X):
    """
    Generate predictions for a batch of data.
    
    Returns:
        predictions (0/1), probabilities (0-1), risk scores (scaled)
    """
    # Scale features
    X_scaled = scaler.transform(X)
    
    # Generate predictions
    predictions = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)[:, 1]
    
    # Convert probabilities to risk scores (scaled 300-850 like FICO)
    # Higher probability of default = lower score
    risk_scores = (850 - (probabilities * 550)).astype(int)
    
    return predictions, probabilities, risk_scores


def main():
    parser = argparse.ArgumentParser(
        description='Generate credit risk predictions for batch data'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to input data CSV for scoring'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='predictions.csv',
        help='Path to output predictions CSV'
    )
    parser.add_argument(
        '--model-dir',
        type=str,
        default='models',
        help='Directory containing trained model artifacts'
    )
    
    args = parser.parse_args()
    
    # Load model
    model, scaler, feature_names, metadata = load_model_artifacts(args.model_dir)
    
    # Load input data
    print(f"\nLoading input data from {args.input}...")
    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} records")
    
    # Prepare features
    print("\nPreparing features...")
    X = prepare_input_features(df, feature_names)
    print(f"Feature matrix shape: {X.shape}")
    
    # Generate predictions
    print("\nGenerating predictions...")
    predictions, probabilities, risk_scores = score_batch(model, scaler, X)
    
    # Create output DataFrame
    output_df = df[['application_id']].copy() if 'application_id' in df.columns else df.copy()
    output_df['predicted_default'] = predictions
    output_df['default_probability'] = probabilities.round(4)
    output_df['risk_score'] = risk_scores
    output_df['scoring_date'] = datetime.now().isoformat()
    
    # Add risk category
    output_df['risk_category'] = pd.cut(
        output_df['default_probability'],
        bins=[0, 0.05, 0.15, 0.30, 1.0],
        labels=['Low', 'Medium', 'High', 'Very High']
    )
    
    # Save predictions
    output_df.to_csv(args.output, index=False)
    print(f"\nSaved predictions to {args.output}")
    
    # Print summary statistics
    print(f"\nPrediction Summary:")
    print(f"  Total applications: {len(output_df)}")
    print(f"  Predicted defaults: {predictions.sum()} ({predictions.mean():.2%})")
    print(f"  Average default probability: {probabilities.mean():.2%}")
    print(f"  Average risk score: {risk_scores.mean():.0f}")
    print(f"\nRisk Category Distribution:")
    print(output_df['risk_category'].value_counts().sort_index())
    print(f"\nFirst few predictions:")
    print(output_df.head(10))
    
    print("\n✓ Batch scoring complete!")


if __name__ == '__main__':
    main()
