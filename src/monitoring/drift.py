"""
Drift monitoring module for credit models.
Calculates Population Stability Index (PSI) for features and score distributions.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional
import json
from datetime import datetime


def calculate_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    bins: int = 10,
    bin_edges: Optional[np.ndarray] = None
) -> Tuple[float, pd.DataFrame]:
    """
    Calculate Population Stability Index (PSI).
    
    PSI = sum((actual_pct - expected_pct) * ln(actual_pct / expected_pct))
    
    Args:
        expected: Expected (baseline) distribution
        actual: Actual (current) distribution
        bins: Number of bins (if bin_edges not provided)
        bin_edges: Custom bin edges (optional)
        
    Returns:
        PSI value and bin-level details
    """
    # Create bins
    if bin_edges is None:
        _, bin_edges = np.histogram(expected, bins=bins)
    
    # Add small value to avoid zero bins
    epsilon = 1e-5
    
    # Calculate distributions
    expected_counts, _ = np.histogram(expected, bins=bin_edges)
    actual_counts, _ = np.histogram(actual, bins=bin_edges)
    
    expected_pct = expected_counts / len(expected)
    actual_pct = actual_counts / len(actual)
    
    # Add epsilon to avoid log(0)
    expected_pct = expected_pct + epsilon
    actual_pct = actual_pct + epsilon
    
    # Normalize
    expected_pct = expected_pct / expected_pct.sum()
    actual_pct = actual_pct / actual_pct.sum()
    
    # Calculate PSI
    psi_values = (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)
    psi = psi_values.sum()
    
    # Create detail dataframe
    psi_details = pd.DataFrame({
        'bin_start': bin_edges[:-1],
        'bin_end': bin_edges[1:],
        'expected_pct': expected_pct,
        'actual_pct': actual_pct,
        'psi_contribution': psi_values
    })
    
    return psi, psi_details


def calculate_feature_psi(
    baseline_df: pd.DataFrame,
    current_df: pd.DataFrame,
    feature_cols: List[str],
    bins: int = 10
) -> Dict[str, Dict]:
    """
    Calculate PSI for multiple features.
    
    Args:
        baseline_df: Baseline/reference dataset
        current_df: Current dataset to monitor
        feature_cols: List of feature columns to monitor
        bins: Number of bins for PSI calculation
        
    Returns:
        Dictionary with PSI values and details for each feature
    """
    psi_results = {}
    
    for feature in feature_cols:
        if feature not in baseline_df.columns or feature not in current_df.columns:
            continue
        
        baseline_values = baseline_df[feature].dropna().values
        current_values = current_df[feature].dropna().values
        
        if len(baseline_values) == 0 or len(current_values) == 0:
            continue
        
        # Calculate PSI
        psi_value, psi_details = calculate_psi(
            baseline_values,
            current_values,
            bins=bins
        )
        
        psi_results[feature] = {
            'psi': float(psi_value),
            'baseline_mean': float(baseline_values.mean()),
            'current_mean': float(current_values.mean()),
            'baseline_std': float(baseline_values.std()),
            'current_std': float(current_values.std()),
            'details': psi_details.to_dict('records')
        }
    
    return psi_results


def interpret_psi(psi: float) -> str:
    """
    Interpret PSI value.
    
    Args:
        psi: PSI value
        
    Returns:
        Interpretation string
    """
    if psi < 0.1:
        return "No significant change"
    elif psi < 0.2:
        return "Moderate change - investigate"
    else:
        return "Significant change - action required"


class DriftMonitor:
    """Monitor drift in features and predictions."""
    
    def __init__(self, baseline_data: pd.DataFrame = None):
        """
        Initialize drift monitor.
        
        Args:
            baseline_data: Baseline/reference dataset
        """
        self.baseline_data = baseline_data
        self.baseline_stats = None
        
        if baseline_data is not None:
            self._calculate_baseline_stats()
    
    def _calculate_baseline_stats(self):
        """Calculate baseline statistics."""
        numeric_cols = self.baseline_data.select_dtypes(
            include=[np.number]
        ).columns
        
        self.baseline_stats = {}
        for col in numeric_cols:
            self.baseline_stats[col] = {
                'mean': float(self.baseline_data[col].mean()),
                'std': float(self.baseline_data[col].std()),
                'min': float(self.baseline_data[col].min()),
                'max': float(self.baseline_data[col].max()),
                'q25': float(self.baseline_data[col].quantile(0.25)),
                'q50': float(self.baseline_data[col].quantile(0.50)),
                'q75': float(self.baseline_data[col].quantile(0.75))
            }
    
    def monitor(
        self,
        current_data: pd.DataFrame,
        feature_cols: List[str] = None,
        score_col: str = None
    ) -> Dict:
        """
        Monitor drift in current data.
        
        Args:
            current_data: Current dataset
            feature_cols: Features to monitor (if None, use all numeric)
            score_col: Score column to monitor (optional)
            
        Returns:
            Dictionary with drift metrics
        """
        if self.baseline_data is None:
            raise ValueError("Baseline data not set")
        
        if feature_cols is None:
            feature_cols = list(self.baseline_stats.keys())
        
        # Calculate feature PSI
        feature_psi = calculate_feature_psi(
            self.baseline_data,
            current_data,
            feature_cols
        )
        
        # Calculate score PSI if provided
        score_psi = None
        if score_col and score_col in self.baseline_data.columns and score_col in current_data.columns:
            psi_value, _ = calculate_psi(
                self.baseline_data[score_col].values,
                current_data[score_col].values
            )
            score_psi = {
                'psi': float(psi_value),
                'interpretation': interpret_psi(psi_value)
            }
        
        # Create monitoring report
        report = {
            'monitoring_timestamp': datetime.now().isoformat(),
            'baseline_samples': len(self.baseline_data),
            'current_samples': len(current_data),
            'feature_psi': feature_psi,
            'score_psi': score_psi,
            'alerts': []
        }
        
        # Add alerts for high PSI values
        for feature, metrics in feature_psi.items():
            if metrics['psi'] > 0.2:
                report['alerts'].append({
                    'feature': feature,
                    'psi': metrics['psi'],
                    'severity': 'high',
                    'message': f"Significant drift detected in {feature}"
                })
            elif metrics['psi'] > 0.1:
                report['alerts'].append({
                    'feature': feature,
                    'psi': metrics['psi'],
                    'severity': 'medium',
                    'message': f"Moderate drift detected in {feature}"
                })
        
        return report
    
    def save_report(self, report: Dict, filepath: str):
        """Save monitoring report to file."""
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
    
    @classmethod
    def load_baseline(cls, filepath: str):
        """Load baseline data from file."""
        df = pd.read_csv(filepath)
        return cls(baseline_data=df)


def main():
    """Demonstrate drift monitoring."""
    print("Credit Model Drift Monitoring Example\n")
    
    # Generate synthetic baseline and current data
    np.random.seed(42)
    n_baseline = 1000
    n_current = 1000
    
    # Baseline data
    baseline = pd.DataFrame({
        'credit_score': np.random.normal(680, 80, n_baseline),
        'annual_income': np.random.lognormal(10.8, 0.6, n_baseline),
        'debt_to_income': np.random.beta(3, 5, n_baseline),
        'score': np.random.normal(650, 50, n_baseline)
    })
    
    # Current data with some drift
    current = pd.DataFrame({
        'credit_score': np.random.normal(660, 80, n_current),  # Mean shifted
        'annual_income': np.random.lognormal(10.9, 0.6, n_current),  # Slight shift
        'debt_to_income': np.random.beta(3.5, 5, n_current),  # Distribution changed
        'score': np.random.normal(640, 55, n_current)  # Mean and std changed
    })
    
    # Initialize monitor
    monitor = DriftMonitor(baseline_data=baseline)
    
    # Monitor current data
    feature_cols = ['credit_score', 'annual_income', 'debt_to_income']
    report = monitor.monitor(current, feature_cols=feature_cols, score_col='score')
    
    print("Drift Monitoring Report")
    print("=" * 60)
    print(f"Baseline samples: {report['baseline_samples']}")
    print(f"Current samples: {report['current_samples']}")
    print(f"\nFeature PSI Values:")
    
    for feature, metrics in report['feature_psi'].items():
        psi = metrics['psi']
        interpretation = interpret_psi(psi)
        print(f"\n  {feature}:")
        print(f"    PSI: {psi:.4f} ({interpretation})")
        print(f"    Baseline mean: {metrics['baseline_mean']:.2f}")
        print(f"    Current mean: {metrics['current_mean']:.2f}")
    
    if report['score_psi']:
        print(f"\nScore Distribution PSI:")
        print(f"  PSI: {report['score_psi']['psi']:.4f}")
        print(f"  Interpretation: {report['score_psi']['interpretation']}")
    
    if report['alerts']:
        print(f"\nAlerts ({len(report['alerts'])}):")
        for alert in report['alerts']:
            print(f"  [{alert['severity'].upper()}] {alert['message']} (PSI: {alert['psi']:.4f})")
    else:
        print("\nNo alerts")


if __name__ == "__main__":
    main()
