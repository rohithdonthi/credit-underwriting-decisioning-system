"""
Batch scoring script for credit applications.
Loads trained model and scores new applications.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import argparse
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.train import CreditModel
from scoring.scorecard import ScorecardMapper


def load_and_prepare_data(filepath: str) -> pd.DataFrame:
    """Load and prepare data for scoring."""
    df = pd.read_csv(filepath)
    if 'application_date' in df.columns:
        df['application_date'] = pd.to_datetime(df['application_date'])
    return df


def batch_score(input_file: str, model_file: str, output_file: str,
                scorecard_config: dict = None):
    """
    Score applications in batch.
    
    Args:
        input_file: Path to input CSV file
        model_file: Path to trained model pickle file
        output_file: Path to output CSV file
        scorecard_config: Scorecard configuration (score0, odds0, pdo)
    """
    print(f"Loading data from {input_file}...")
    df = load_and_prepare_data(input_file)
    print(f"Loaded {len(df)} applications")
    
    print(f"\nLoading model from {model_file}...")
    model = CreditModel.load(model_file)
    print(f"Model type: {model.model_type}")
    print(f"Training date: {model.training_date}")
    
    # Prepare features
    exclude_cols = ['application_id', 'application_date', 'default']
    feature_cols = [col for col in df.columns 
                   if col not in exclude_cols and col in model.feature_names]
    
    print(f"\nUsing {len(feature_cols)} features for scoring")
    
    X = df[feature_cols]
    
    # Score
    print("Scoring applications...")
    probabilities = model.predict_proba(X)
    
    # Convert to scores
    if scorecard_config is None:
        scorecard_config = {'score0': 600, 'odds0': 50, 'pdo': 20}
    
    mapper = ScorecardMapper(**scorecard_config)
    scores = mapper.prob_to_score(probabilities)
    
    # Create output dataframe
    output_df = df[['application_id']].copy()
    if 'application_date' in df.columns:
        output_df['application_date'] = df['application_date']
    
    output_df['default_probability'] = probabilities
    output_df['credit_score'] = scores.round(0).astype(int)
    output_df['scoring_timestamp'] = datetime.now().isoformat()
    
    # Add risk band
    output_df['risk_band'] = pd.cut(
        scores,
        bins=[-np.inf, 500, 550, 600, 650, np.inf],
        labels=['Very High', 'High', 'Medium', 'Low', 'Very Low']
    )
    
    # Save output
    output_df.to_csv(output_file, index=False)
    print(f"\nScored results saved to {output_file}")
    
    # Print summary
    print("\nScoring Summary:")
    print(f"  Total applications: {len(output_df)}")
    print(f"  Mean probability: {probabilities.mean():.4f}")
    print(f"  Mean score: {scores.mean():.1f}")
    print(f"  Score range: [{scores.min():.0f}, {scores.max():.0f}]")
    print("\nRisk Band Distribution:")
    print(output_df['risk_band'].value_counts().sort_index())
    
    return output_df


def main():
    """Main entry point for batch scoring."""
    parser = argparse.ArgumentParser(
        description='Batch score credit applications'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input CSV file with applications'
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained model pickle file'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output CSV file for scores'
    )
    parser.add_argument(
        '--score0',
        type=float,
        default=600,
        help='Reference score (default: 600)'
    )
    parser.add_argument(
        '--odds0',
        type=float,
        default=50,
        help='Reference odds (default: 50)'
    )
    parser.add_argument(
        '--pdo',
        type=float,
        default=20,
        help='Points to double odds (default: 20)'
    )
    
    args = parser.parse_args()
    
    scorecard_config = {
        'score0': args.score0,
        'odds0': args.odds0,
        'pdo': args.pdo
    }
    
    batch_score(
        args.input,
        args.model,
        args.output,
        scorecard_config
    )


if __name__ == "__main__":
    main()
