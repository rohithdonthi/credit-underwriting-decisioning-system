"""
Training module for credit underwriting models.
Implements logistic regression and gradient boosting with time-based split,
calibration, and comprehensive metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score,
    brier_score_loss,
    roc_curve,
    precision_recall_curve
)
import pickle
import json
from datetime import datetime


class CreditModel:
    """Credit underwriting model with training and evaluation capabilities."""
    
    def __init__(self, model_type: str = 'logistic', calibrate: bool = True):
        """
        Initialize credit model.
        
        Args:
            model_type: 'logistic' or 'gradient_boosting'
            calibrate: Whether to calibrate probabilities
        """
        self.model_type = model_type
        self.calibrate = calibrate
        self.model = None
        self.feature_names = None
        self.training_date = None
        
    def _create_base_model(self):
        """Create base model based on model type."""
        if self.model_type == 'logistic':
            return LogisticRegression(
                max_iter=2000,
                random_state=42,
                solver='lbfgs'
            )
        elif self.model_type == 'gradient_boosting':
            return GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_cal: pd.DataFrame = None, y_cal: pd.Series = None):
        """
        Train the model with optional calibration.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_cal: Calibration features (optional)
            y_cal: Calibration labels (optional)
        """
        self.feature_names = list(X_train.columns)
        self.training_date = datetime.now().isoformat()
        
        base_model = self._create_base_model()
        
        if self.calibrate and X_cal is not None and y_cal is not None:
            # Train base model on training set
            base_model.fit(X_train, y_train)
            # Create calibrator separately and fit on calibration set
            from sklearn.isotonic import IsotonicRegression
            # Get predictions on calibration set
            y_pred_cal = base_model.predict_proba(X_cal)[:, 1]
            # Fit isotonic regression
            iso_reg = IsotonicRegression(out_of_bounds='clip')
            iso_reg.fit(y_pred_cal, y_cal)
            # Store both models
            self.base_model = base_model
            self.calibrator = iso_reg
            self.model = base_model  # For compatibility
        elif self.calibrate:
            # Use cross-validation for calibration
            self.model = CalibratedClassifierCV(
                self._create_base_model(),
                method='isotonic',
                cv=5,
                ensemble=False
            )
            self.model.fit(X_train, y_train)
            self.base_model = None
            self.calibrator = None
        else:
            # No calibration
            base_model.fit(X_train, y_train)
            self.model = base_model
            self.base_model = None
            self.calibrator = None
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict default probabilities."""
        if hasattr(self, 'base_model') and self.base_model is not None and hasattr(self, 'calibrator') and self.calibrator is not None:
            # Use manually calibrated model
            raw_probs = self.base_model.predict_proba(X)[:, 1]
            calibrated_probs = self.calibrator.predict(raw_probs)
            return calibrated_probs
        else:
            # Use standard model
            return self.model.predict_proba(X)[:, 1]
    
    def save(self, filepath: str):
        """Save model to file."""
        model_data = {
            'model': self.model,
            'base_model': getattr(self, 'base_model', None),
            'calibrator': getattr(self, 'calibrator', None),
            'model_type': self.model_type,
            'calibrate': self.calibrate,
            'feature_names': self.feature_names,
            'training_date': self.training_date
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    @classmethod
    def load(cls, filepath: str):
        """Load model from file."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        instance = cls(
            model_type=model_data['model_type'],
            calibrate=model_data['calibrate']
        )
        instance.model = model_data['model']
        instance.base_model = model_data.get('base_model')
        instance.calibrator = model_data.get('calibrator')
        instance.feature_names = model_data['feature_names']
        instance.training_date = model_data['training_date']
        return instance


def time_based_split(df: pd.DataFrame, date_column: str, 
                     train_ratio: float = 0.6, cal_ratio: float = 0.2
                     ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data based on time.
    
    Args:
        df: Input dataframe
        date_column: Name of date column
        train_ratio: Proportion for training
        cal_ratio: Proportion for calibration
        
    Returns:
        train_df, cal_df, test_df
    """
    df_sorted = df.sort_values(date_column).reset_index(drop=True)
    n = len(df_sorted)
    
    train_end = int(n * train_ratio)
    cal_end = int(n * (train_ratio + cal_ratio))
    
    train_df = df_sorted.iloc[:train_end]
    cal_df = df_sorted.iloc[train_end:cal_end]
    test_df = df_sorted.iloc[cal_end:]
    
    return train_df, cal_df, test_df


def calculate_ks_statistic(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
    """
    Calculate Kolmogorov-Smirnov statistic.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        
    Returns:
        KS statistic
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    ks = np.max(tpr - fpr)
    return ks


def calculate_lift_at_percentile(y_true: np.ndarray, y_pred_proba: np.ndarray,
                                  percentile: float = 10) -> float:
    """
    Calculate lift at given percentile.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        percentile: Percentile to calculate lift at
        
    Returns:
        Lift value
    """
    df = pd.DataFrame({'y_true': y_true, 'y_pred_proba': y_pred_proba})
    df = df.sort_values('y_pred_proba', ascending=False)
    
    n_top = int(len(df) * percentile / 100)
    if n_top == 0:
        return 0.0
    
    top_default_rate = df.iloc[:n_top]['y_true'].mean()
    overall_default_rate = df['y_true'].mean()
    
    if overall_default_rate == 0:
        return 0.0
    
    lift = top_default_rate / overall_default_rate
    return lift


def evaluate_model(y_true: np.ndarray, y_pred_proba: np.ndarray,
                   dataset_name: str = 'test') -> Dict[str, float]:
    """
    Calculate comprehensive model metrics.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        dataset_name: Name of dataset being evaluated
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        f'{dataset_name}_auc': roc_auc_score(y_true, y_pred_proba),
        f'{dataset_name}_ks': calculate_ks_statistic(y_true, y_pred_proba),
        f'{dataset_name}_brier': brier_score_loss(y_true, y_pred_proba),
        f'{dataset_name}_lift_10pct': calculate_lift_at_percentile(y_true, y_pred_proba, 10),
        f'{dataset_name}_default_rate': float(y_true.mean())
    }
    return metrics


def prepare_features(df: pd.DataFrame, target_col: str = 'default',
                    exclude_cols: list = None) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare features and target from dataframe.
    
    Args:
        df: Input dataframe
        target_col: Name of target column
        exclude_cols: Columns to exclude from features
        
    Returns:
        X (features), y (target)
    """
    if exclude_cols is None:
        exclude_cols = ['application_id', 'application_date']
    
    exclude_cols = exclude_cols + [target_col]
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols]
    y = df[target_col]
    
    return X, y


def main():
    """Train and evaluate credit models."""
    print("Loading data...")
    df = pd.read_csv('data/sample/credit_applications.csv')
    df['application_date'] = pd.to_datetime(df['application_date'])
    
    print("Splitting data by time...")
    train_df, cal_df, test_df = time_based_split(df, 'application_date')
    print(f"Train: {len(train_df)}, Cal: {len(cal_df)}, Test: {len(test_df)}")
    
    X_train, y_train = prepare_features(train_df)
    X_cal, y_cal = prepare_features(cal_df)
    X_test, y_test = prepare_features(test_df)
    
    results = {}
    
    # Train Logistic Regression
    print("\nTraining Logistic Regression...")
    lr_model = CreditModel(model_type='logistic', calibrate=True)
    lr_model.train(X_train, y_train, X_cal, y_cal)
    lr_model.save('models/logistic_model.pkl')
    
    y_pred_lr = lr_model.predict_proba(X_test)
    lr_metrics = evaluate_model(y_test, y_pred_lr, 'logistic')
    results.update(lr_metrics)
    
    print("\nLogistic Regression Metrics:")
    for metric, value in lr_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Train Gradient Boosting
    print("\nTraining Gradient Boosting...")
    gb_model = CreditModel(model_type='gradient_boosting', calibrate=True)
    gb_model.train(X_train, y_train, X_cal, y_cal)
    gb_model.save('models/gradient_boosting_model.pkl')
    
    y_pred_gb = gb_model.predict_proba(X_test)
    gb_metrics = evaluate_model(y_test, y_pred_gb, 'gradient_boosting')
    results.update(gb_metrics)
    
    print("\nGradient Boosting Metrics:")
    for metric, value in gb_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Save metrics
    with open('models/training_metrics.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nModels and metrics saved to models/")


if __name__ == "__main__":
    main()
