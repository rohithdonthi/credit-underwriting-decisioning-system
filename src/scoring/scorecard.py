"""
Scorecard mapping module for converting probabilities to credit scores.
Implements standard Points to Double the Odds (PDO) methodology.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional


class ScorecardMapper:
    """Maps default probabilities to credit scores using PDO methodology."""
    
    def __init__(self, score0: float = 600, odds0: float = 50, pdo: float = 20):
        """
        Initialize scorecard mapper.
        
        Args:
            score0: Reference score at reference odds
            odds0: Reference odds (good:bad ratio)
            pdo: Points to double the odds
            
        The relationship is: Score = score0 + (pdo / ln(2)) * ln(odds / odds0)
        where odds = (1 - prob) / prob
        Higher odds (lower default probability) = higher score
        """
        self.score0 = score0
        self.odds0 = odds0
        self.pdo = pdo
        
        # Calculate scaling factors
        self.factor = pdo / np.log(2)
        self.offset = score0 - self.factor * np.log(odds0)
    
    def prob_to_score(self, prob: np.ndarray) -> np.ndarray:
        """
        Convert default probabilities to credit scores.
        
        Args:
            prob: Default probabilities (0-1)
            
        Returns:
            Credit scores
        """
        # Clip probabilities to avoid division by zero
        prob = np.clip(prob, 1e-10, 1 - 1e-10)
        
        # Calculate odds (good:bad ratio)
        odds = (1 - prob) / prob
        
        # Calculate score - higher odds means higher score
        score = self.offset + self.factor * np.log(odds)
        
        return score
    
    def score_to_prob(self, score: np.ndarray) -> np.ndarray:
        """
        Convert credit scores back to default probabilities.
        
        Args:
            score: Credit scores
            
        Returns:
            Default probabilities
        """
        # Calculate odds from score
        odds = np.exp((score - self.offset) / self.factor)
        
        # Calculate probability
        prob = 1 / (1 + odds)
        
        return prob
    
    def get_config(self) -> Dict[str, float]:
        """Get scorecard configuration."""
        return {
            'score0': self.score0,
            'odds0': self.odds0,
            'pdo': self.pdo
        }
    
    def describe(self):
        """Print scorecard configuration and examples."""
        print("Scorecard Configuration:")
        print(f"  Reference Score (SCORE0): {self.score0}")
        print(f"  Reference Odds (ODDS0): {self.odds0}")
        print(f"  Points to Double Odds (PDO): {self.pdo}")
        print(f"\nExamples:")
        
        example_probs = [0.01, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]
        for prob in example_probs:
            score = self.prob_to_score(np.array([prob]))[0]
            odds = (1 - prob) / prob
            print(f"  Probability: {prob:.2%} (Odds: {odds:.1f}:1) -> Score: {score:.0f}")


def create_score_bins(scores: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """
    Create score bins and calculate statistics.
    
    Args:
        scores: Array of credit scores
        n_bins: Number of bins
        
    Returns:
        DataFrame with bin statistics
    """
    bins = pd.qcut(scores, q=n_bins, duplicates='drop')
    
    df = pd.DataFrame({
        'score': scores,
        'bin': bins
    })
    
    bin_stats = df.groupby('bin').agg(
        count=('score', 'size'),
        min_score=('score', 'min'),
        max_score=('score', 'max'),
        mean_score=('score', 'mean')
    ).reset_index()
    
    return bin_stats


def main():
    """Demonstrate scorecard mapping."""
    print("Credit Scorecard Mapping Example\n")
    print("=" * 60)
    
    # Create scorecard mapper
    mapper = ScorecardMapper(score0=600, odds0=50, pdo=20)
    mapper.describe()
    
    print("\n" + "=" * 60)
    print("\nGenerating example scores from probabilities...")
    
    # Generate example data
    np.random.seed(42)
    probs = np.random.beta(2, 8, 1000)  # Skewed towards lower probabilities
    scores = mapper.prob_to_score(probs)
    
    print(f"\nScore Statistics:")
    print(f"  Mean: {scores.mean():.1f}")
    print(f"  Median: {np.median(scores):.1f}")
    print(f"  Std Dev: {scores.std():.1f}")
    print(f"  Min: {scores.min():.1f}")
    print(f"  Max: {scores.max():.1f}")
    
    # Create bins
    print("\nScore Distribution by Decile:")
    bin_stats = create_score_bins(scores, n_bins=10)
    print(bin_stats.to_string(index=False))
    
    print("\n" + "=" * 60)
    print("\nVerification (score back to probability):")
    test_scores = np.array([550, 600, 650, 700])
    recovered_probs = mapper.score_to_prob(test_scores)
    for score, prob in zip(test_scores, recovered_probs):
        print(f"  Score {score:.0f} -> Probability: {prob:.4f}")


if __name__ == "__main__":
    main()
