"""
Unit tests for scorecard mapping.
"""

import pytest
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from scoring.scorecard import ScorecardMapper


def test_scorecard_initialization():
    """Test scorecard mapper initialization."""
    mapper = ScorecardMapper(score0=600, odds0=50, pdo=20)
    
    assert mapper.score0 == 600
    assert mapper.odds0 == 50
    assert mapper.pdo == 20


def test_prob_to_score_basic():
    """Test basic probability to score conversion."""
    mapper = ScorecardMapper(score0=600, odds0=50, pdo=20)
    
    # Test with single probability
    prob = np.array([0.10])
    score = mapper.prob_to_score(prob)
    
    assert len(score) == 1
    assert isinstance(score[0], (int, float, np.number))
    assert 500 < score[0] < 700  # Reasonable range


def test_prob_to_score_multiple():
    """Test multiple probability conversions."""
    mapper = ScorecardMapper(score0=600, odds0=50, pdo=20)
    
    probs = np.array([0.01, 0.05, 0.10, 0.20, 0.50])
    scores = mapper.prob_to_score(probs)
    
    assert len(scores) == len(probs)
    # Lower probabilities should give higher scores
    assert scores[0] > scores[-1]


def test_prob_to_score_monotonic():
    """Test that higher probabilities give lower scores."""
    mapper = ScorecardMapper(score0=600, odds0=50, pdo=20)
    
    probs = np.array([0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30])
    scores = mapper.prob_to_score(probs)
    
    # Scores should be monotonically decreasing
    assert np.all(np.diff(scores) < 0)


def test_score_to_prob_roundtrip():
    """Test that prob->score->prob is consistent."""
    mapper = ScorecardMapper(score0=600, odds0=50, pdo=20)
    
    original_probs = np.array([0.05, 0.10, 0.15, 0.20])
    scores = mapper.prob_to_score(original_probs)
    recovered_probs = mapper.score_to_prob(scores)
    
    np.testing.assert_array_almost_equal(original_probs, recovered_probs, decimal=6)


def test_prob_edge_cases():
    """Test edge case probabilities."""
    mapper = ScorecardMapper(score0=600, odds0=50, pdo=20)
    
    # Very low probability
    low_prob = np.array([0.001])
    score_low = mapper.prob_to_score(low_prob)
    assert score_low[0] > 600  # Should be higher than reference
    
    # High probability
    high_prob = np.array([0.90])
    score_high = mapper.prob_to_score(high_prob)
    assert score_high[0] < 600  # Should be lower than reference


def test_get_config():
    """Test getting scorecard configuration."""
    mapper = ScorecardMapper(score0=600, odds0=50, pdo=20)
    config = mapper.get_config()
    
    assert config['score0'] == 600
    assert config['odds0'] == 50
    assert config['pdo'] == 20


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
