# src/models/scorecard.py
import numpy as np

def prob_to_score(p_good: np.ndarray, score0=600, odds0=50, pdo=20) -> np.ndarray:
    """
    Map probability of 'good' to a scorecard-like score.
    Higher p_good => higher score.
    """
    p_good = np.clip(np.asarray(p_good), 1e-6, 1 - 1e-6)
    odds = p_good / (1 - p_good)  # good:bad
    factor = pdo / np.log(2)
    offset = score0 - factor * np.log(odds0)
    score = offset + factor * np.log(odds)
    return np.round(score).astype(int)

def decision_from_prob(p_good: float, thresh_approve=0.70, thresh_review=0.50) -> str:
    if p_good >= thresh_approve:
        return "APPROVE"
    if p_good >= thresh_review:
        return "REVIEW"
    return "DECLINE"
