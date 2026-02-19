# Credit Underwriting Decisioning System

End-to-end underwriting decisioning pipeline with time-based validation, calibrated probability outputs, scorecard mapping, and monitoring-ready artifacts.

## Overview

This system provides a complete credit underwriting decisioning pipeline that includes:

- **Synthetic data generation** for testing and development
- **Multiple ML models** (Logistic Regression and Gradient Boosting)
- **Time-based data splitting** to simulate realistic production scenarios
- **Probability calibration** for reliable risk estimates
- **Scorecard mapping** using industry-standard Points-to-Double-the-Odds (PDO) methodology
- **Comprehensive metrics** (AUC, KS, Brier, Lift@10%)
- **Batch scoring** capabilities
- **Drift monitoring** using Population Stability Index (PSI)
- **REST API** for real-time scoring (optional)

## Architecture

```
credit-underwriting-decisioning-system/
├── src/
│   ├── data/              # Data generation modules
│   │   └── synthetic_generator.py
│   ├── models/            # Training and model management
│   │   └── train.py
│   ├── scoring/           # Scoring and scorecard mapping
│   │   ├── scorecard.py
│   │   └── batch_score.py
│   ├── monitoring/        # Drift monitoring
│   │   └── drift.py
│   └── api/              # FastAPI application
│       └── app.py
├── data/
│   └── sample/           # Sample datasets
├── models/               # Trained model artifacts
├── notebooks/            # Jupyter notebooks
├── docs/                 # Documentation
├── tests/               # Unit tests
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

### System Components

1. **Data Layer**: Generates synthetic credit applications with realistic feature distributions
2. **Model Layer**: Trains and calibrates ML models (Logistic Regression, Gradient Boosting)
3. **Scoring Layer**: Converts probabilities to credit scores using PDO methodology
4. **Monitoring Layer**: Tracks feature and score drift using PSI
5. **API Layer**: Optional REST API for real-time scoring

## Getting Started

### Prerequisites

- Python 3.8 or higher
- pip or conda for package management

### Installation

1. Clone the repository:
```bash
git clone https://github.com/rohithdonthi/credit-underwriting-decisioning-system.git
cd credit-underwriting-decisioning-system
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create necessary directories:
```bash
mkdir -p models data/sample
```

## How to Run

### 1. Generate Synthetic Data

Generate a synthetic credit application dataset:

```bash
python src/data/synthetic_generator.py
```

This creates `data/sample/credit_applications.csv` with 10,000 synthetic credit applications.

### 2. Train Models

Train and evaluate credit models:

```bash
python src/models/train.py
```

This will:
- Split data using time-based approach (60% train, 20% calibration, 20% test)
- Train Logistic Regression and Gradient Boosting models
- Calibrate probabilities using isotonic regression
- Calculate comprehensive metrics (AUC, KS, Brier, Lift@10%)
- Save trained models to `models/` directory

**Expected Output:**
```
Train: 6000, Cal: 2000, Test: 2000

Logistic Regression Metrics:
  logistic_auc: 0.8542
  logistic_ks: 0.5234
  logistic_brier: 0.0987
  logistic_lift_10pct: 2.45

Gradient Boosting Metrics:
  gradient_boosting_auc: 0.8756
  gradient_boosting_ks: 0.5678
  gradient_boosting_brier: 0.0876
  gradient_boosting_lift_10pct: 2.67
```

### 3. Score Applications (Batch)

Score applications in batch mode:

```bash
python src/scoring/batch_score.py \
  --input data/sample/credit_applications.csv \
  --model models/gradient_boosting_model.pkl \
  --output data/sample/scored_applications.csv
```

Optional scorecard parameters:
```bash
python src/scoring/batch_score.py \
  --input data/sample/credit_applications.csv \
  --model models/gradient_boosting_model.pkl \
  --output data/sample/scored_applications.csv \
  --score0 600 \
  --odds0 50 \
  --pdo 20
```

**Output includes:**
- Application ID
- Default probability
- Credit score (mapped from probability)
- Risk band (Very Low, Low, Medium, High, Very High)

### 4. Monitor Drift

Monitor feature and score drift:

```bash
python src/monitoring/drift.py
```

This demonstrates:
- Calculation of Population Stability Index (PSI)
- Feature-level drift detection
- Score distribution drift
- Automated alerting for significant drift

**PSI Interpretation:**
- PSI < 0.1: No significant change
- 0.1 ≤ PSI < 0.2: Moderate change - investigate
- PSI ≥ 0.2: Significant change - action required

### 5. Run API Server (Optional)

Start the FastAPI server:

```bash
# Set model path (optional, defaults to models/gradient_boosting_model.pkl)
export MODEL_PATH=models/gradient_boosting_model.pkl

# Run server
python src/api/app.py
```

Or using uvicorn directly:
```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

Access the interactive API documentation at: `http://localhost:8000/docs`

#### API Endpoints

**Health Check:**
```bash
curl http://localhost:8000/health
```

**Score Single Application:**
```bash
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{
    "application_id": "APP00001234",
    "credit_score": 720,
    "num_credit_lines": 5,
    "credit_utilization": 0.3,
    "num_delinquencies": 0,
    "months_since_last_delinq": -1,
    "annual_income": 75000,
    "employment_length_years": 5,
    "debt_to_income": 0.25,
    "loan_amount": 15000,
    "loan_term_months": 36,
    "interest_rate": 8.5,
    "num_inquiries_6m": 1,
    "revolving_balance": 5000,
    "has_mortgage": 1,
    "has_car_loan": 0
  }'
```

**Response:**
```json
{
  "application_id": "APP00001234",
  "default_probability": 0.0456,
  "credit_score": 678,
  "risk_band": "Low",
  "decision": "Approve"
}
```

## Scorecard Methodology

The system uses industry-standard Points-to-Double-the-Odds (PDO) methodology to convert default probabilities to credit scores.

### Configuration

- **SCORE0 (Reference Score)**: 600 - The score at the reference odds
- **ODDS0 (Reference Odds)**: 50 - The good:bad odds ratio at the reference score (50:1)
- **PDO (Points to Double Odds)**: 20 - Points required to double the odds

### Formula

```
Score = SCORE0 - (PDO / ln(2)) × ln(Odds / ODDS0)
where Odds = (1 - Probability) / Probability
```

### Example Mappings

| Probability | Odds | Score |
|------------|------|-------|
| 1% | 99:1 | 679 |
| 5% | 19:1 | 640 |
| 10% | 9:1 | 620 |
| 15% | 5.67:1 | 605 |
| 20% | 4:1 | 591 |

## Monitoring

The system includes drift monitoring using Population Stability Index (PSI):

### Feature Monitoring

Tracks changes in feature distributions between baseline and current data:

```python
from src.monitoring.drift import DriftMonitor

# Load baseline data
monitor = DriftMonitor.load_baseline('data/sample/baseline.csv')

# Monitor current data
current_df = pd.read_csv('data/sample/current.csv')
report = monitor.monitor(current_df, feature_cols=['credit_score', 'annual_income'])

# Save report
monitor.save_report(report, 'monitoring/drift_report.json')
```

### Score Distribution Monitoring

Tracks changes in the score distribution to detect model degradation or population shifts.

## Model Metrics

### Training Metrics

The system calculates the following metrics during training:

1. **AUC (Area Under ROC Curve)**: Measures model's ability to discriminate between classes
   - Good: > 0.8
   - Acceptable: 0.7 - 0.8
   - Poor: < 0.7

2. **KS Statistic (Kolmogorov-Smirnov)**: Maximum separation between good and bad distributions
   - Good: > 0.4
   - Acceptable: 0.3 - 0.4
   - Poor: < 0.3

3. **Brier Score**: Measures accuracy of probability predictions (lower is better)
   - Good: < 0.10
   - Acceptable: 0.10 - 0.15
   - Poor: > 0.15

4. **Lift@10%**: How much better the model performs on the top 10% compared to random
   - Good: > 2.5
   - Acceptable: 2.0 - 2.5
   - Poor: < 2.0

### Calibration

Models are calibrated using isotonic regression on a held-out calibration set to ensure:
- Predicted probabilities match observed frequencies
- Reliable risk estimates across the probability range
- Better decision-making capabilities

## Limitations

### Current Limitations

1. **Synthetic Data Only**: The system uses synthetic data and should not be used for real credit decisions without proper validation

2. **Feature Engineering**: Limited feature engineering is implemented. Real-world applications would benefit from:
   - Interaction features
   - Non-linear transformations
   - Domain-specific features

3. **Model Complexity**: Current models are relatively simple. Consider:
   - Deep learning approaches for larger datasets
   - Ensemble methods beyond gradient boosting
   - AutoML for hyperparameter optimization

4. **Fairness**: No explicit fairness constraints or bias mitigation is implemented. In production:
   - Conduct fairness audits
   - Implement bias mitigation techniques
   - Ensure compliance with fair lending regulations

5. **Explainability**: Limited model interpretability features. Consider adding:
   - SHAP values for individual predictions
   - Feature importance analysis
   - Reason codes for decisions

6. **Scalability**: Batch processing may be slow for very large datasets. Consider:
   - Distributed computing (Spark, Dask)
   - Database integration
   - Streaming capabilities

7. **Regulatory Compliance**: The system does not include:
   - Audit trails
   - Model governance workflows
   - Regulatory reporting
   - Documentation for SR 11-7 compliance

### Production Considerations

Before deploying to production:

1. **Validation**: Validate on real historical data with known outcomes
2. **Backtesting**: Perform comprehensive backtesting over multiple time periods
3. **Champion-Challenger**: Implement A/B testing framework
4. **Monitoring**: Set up comprehensive monitoring and alerting
5. **Model Governance**: Establish model review and approval processes
6. **Documentation**: Create detailed model documentation (model cards)
7. **Security**: Implement authentication, authorization, and data encryption
8. **Performance**: Optimize for latency and throughput requirements
9. **Disaster Recovery**: Implement backup and failover strategies
10. **Compliance**: Ensure regulatory compliance (FCRA, ECOA, etc.)

## Testing

Run tests (when implemented):

```bash
pytest tests/ -v --cov=src
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions or feedback, please open an issue on GitHub.

## Acknowledgments

- Scikit-learn for machine learning models
- FastAPI for API framework
- Standard credit risk modeling methodologies from industry best practices
