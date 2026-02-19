# src/config.py
TIME_COL = "ym"
TARGET_COL = "good"

# Decision thresholds (tune later)
THRESH_APPROVE = 0.70
THRESH_REVIEW = 0.50

# Scorecard mapping params (classic PDO scaling)
SCORE0 = 600
ODDS0 = 50     # odds good:bad at SCORE0
PDO = 20       # points to double the odds

RANDOM_SEED = 42
