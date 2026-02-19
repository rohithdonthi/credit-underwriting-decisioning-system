# Credit Underwriting Decisioning System

End-to-end underwriting decisioning pipeline with time-based validation, calibrated probability outputs, scorecard mapping, and monitoring-ready artifacts.

## Overview

This repository provides a minimal, production-ready framework for building and deploying credit risk models. The system uses synthetic data to demonstrate best practices for:

- **Time-based model validation** to simulate real-world deployment
- **Baseline modeling** with interpretable logistic regression
- **Batch scoring infrastructure** for credit decisioning
- **Risk categorization** and scorecard outputs

**Note:** This system uses only synthetic data generated via Faker and statistical distributions. No real customer data is included.

## Architecture

```
credit-underwriting-decisioning-system/
├── src/
│   ├── data/
│   │   └── make_dataset.py       # Generate synthetic credit data
│   └── models/
│       ├── train.py               # Train model with time-based split
│       └── predict.py             # Batch scoring pipeline
├── notebooks/                     # Jupyter notebooks for exploration
├── docs/                          # Additional documentation
├── tests/                         # Unit and integration tests
├── data/
│   └── sample/                    # Synthetic sample data
├── models/                        # Trained model artifacts (created after training)
├── requirements.txt               # Python dependencies
└── README.md
```

### Pipeline Flow

1. **Data Generation**: Create synthetic credit applications with realistic features
2. **Model Training**: Train on historical data (time-based split)
3. **Model Evaluation**: Validate on future time period
4. **Batch Scoring**: Score new applications with risk probabilities

## How to Run

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/rohithdonthi/credit-underwriting-decisioning-system.git
cd credit-underwriting-decisioning-system
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Generate Synthetic Data

Create a synthetic dataset of credit applications:

```bash
python src/data/make_dataset.py --output data/sample/credit_applications.csv --n-samples 10000
```

**Parameters:**
- `--output`: Path to save the CSV file (default: `data/sample/credit_applications.csv`)
- `--n-samples`: Number of synthetic applications to generate (default: 10000)

**Output:** CSV file with features including:
- Applicant demographics (age, income, employment)
- Credit history (length, derogatory marks, inquiries)
- Current debt metrics (DTI ratio, revolving utilization)
- Loan details (amount, purpose)
- Target variable (default: 0/1)

### Train Model

Train a baseline logistic regression model using time-based validation:

```bash
python src/models/train.py --data data/sample/credit_applications.csv --output-dir models --train-ratio 0.7
```

**Parameters:**
- `--data`: Path to input CSV file
- `--output-dir`: Directory to save model artifacts (default: `models`)
- `--train-ratio`: Proportion for training set (default: 0.7, remaining is test set)

**Output:** Saves trained model artifacts to `models/`:
- `model.pkl`: Trained logistic regression model
- `scaler.pkl`: Feature scaler (StandardScaler)
- `features.pkl`: Feature names and order
- `metadata.pkl`: Training metadata and model info

**Validation Approach:**
- Time-based split ensures training data precedes test data chronologically
- Simulates real-world scenario where model predicts future outcomes
- Reports ROC-AUC, precision, recall, and confusion matrix

### Batch Scoring

Score new credit applications using the trained model:

```bash
python src/models/predict.py --input data/sample/credit_applications.csv --output predictions.csv --model-dir models
```

**Parameters:**
- `--input`: Path to input data for scoring
- `--output`: Path to save predictions (default: `predictions.csv`)
- `--model-dir`: Directory containing trained model (default: `models`)

**Output:** CSV file with predictions including:
- `predicted_default`: Binary prediction (0 = approve, 1 = decline)
- `default_probability`: Probability of default (0.0 to 1.0)
- `risk_score`: FICO-style score (300-850, higher is lower risk)
- `risk_category`: Low/Medium/High/Very High risk classification

### End-to-End Example

Run the complete pipeline:

```bash
# 1. Generate synthetic data
python src/data/make_dataset.py

# 2. Train the model
python src/models/train.py

# 3. Generate predictions
python src/models/predict.py --input data/sample/credit_applications.csv
```

## Model Details

### Baseline Model

- **Algorithm**: Logistic Regression with L2 regularization
- **Features**: 
  - Numeric: age, income, employment length, DTI ratio, credit history, revolving balances, inquiries
  - Categorical: loan purpose (one-hot encoded)
- **Class balancing**: Weighted by inverse class frequency
- **Preprocessing**: StandardScaler for feature normalization

### Feature Engineering

The model uses interpretable features aligned with traditional credit underwriting:

1. **Capacity**: Income and employment stability
2. **Capital**: Existing debt and revolving utilization
3. **Credit History**: Length, derogatory marks, open lines
4. **Conditions**: Loan amount and purpose

### Risk Scoring

Predicted probabilities are converted to FICO-style scores:
- Score range: 300-850
- Higher score = Lower risk
- Formula: `score = 850 - (probability × 550)`

Risk categories:
- **Low Risk**: < 5% default probability (scores 575-850)
- **Medium Risk**: 5-15% default probability (scores 468-575)
- **High Risk**: 15-30% default probability (scores 350-468)
- **Very High Risk**: > 30% default probability (scores 300-350)

## Monitoring

### Model Performance Tracking

Monitor these key metrics over time:

1. **Discrimination Metrics**:
   - ROC-AUC: Overall ability to rank risky applicants
   - Precision/Recall: Trade-off at chosen threshold
   
2. **Calibration Metrics**:
   - Brier Score: Accuracy of probability estimates
   - Calibration plots: Predicted vs observed default rates
   
3. **Business Metrics**:
   - Approval rate
   - Default rate in approved applications
   - Revenue impact from decisions

### Data Drift Detection

Monitor for distribution shifts in:
- Feature distributions (e.g., average income, DTI ratios)
- Application volume patterns
- Geographic or demographic mix changes

### Retraining Strategy

Consider retraining when:
- Model performance degrades (ROC-AUC drops > 5%)
- Significant data drift detected
- Major economic changes occur
- New features become available

## Limitations

### Model Limitations

1. **Synthetic Data**: Model trained on artificial data may not generalize to real credit data
2. **Simple Baseline**: Logistic regression captures linear relationships only
3. **Limited Features**: Real systems use hundreds of features and alternative data
4. **No Explainability**: Basic model lacks detailed reason codes for decisions
5. **Static Model**: No online learning or adaptation to changing patterns

### Data Limitations

1. **No Real-World Validation**: Synthetic data doesn't capture true credit risk complexity
2. **Simplified Correlations**: Real credit data has more nuanced feature relationships
3. **Missing External Factors**: Macroeconomic conditions, industry trends not included
4. **No Time-Series Features**: Doesn't capture trajectory of financial behavior

### Production Readiness

This is a **demonstration system** and would require significant enhancements for production:

1. **Data Pipeline**: Integration with credit bureaus, fraud detection, data validation
2. **Model Improvements**: Ensemble methods, calibration, fairness constraints
3. **Infrastructure**: API endpoints, model versioning, A/B testing framework
4. **Compliance**: Fair lending requirements, adverse action notices, audit trails
5. **Monitoring**: Real-time dashboards, alerting, automated retraining
6. **Security**: Data encryption, access controls, PII protection
7. **Governance**: Model documentation, validation reports, regulatory approval

### Regulatory Considerations

Credit models must comply with:
- Fair Credit Reporting Act (FCRA)
- Equal Credit Opportunity Act (ECOA)
- Fair lending regulations
- Model risk management guidelines (SR 11-7)

This demonstration does not include fairness testing, bias mitigation, or adverse action reason codes required for production deployment.

## Contributing

This is a demonstration repository. For production implementations, consult with:
- Credit risk experts
- Compliance and legal teams
- Data scientists and ML engineers
- Model validation teams

## License

MIT License - See LICENSE file for details.

## Disclaimer

This system is for educational and demonstration purposes only. It uses entirely synthetic data and should not be used for actual credit decisions without proper validation, compliance review, and regulatory approval.
