# Credit Underwriting Decisioning System - Architecture

## System Architecture

### Overview

The Credit Underwriting Decisioning System is designed as a modular, production-ready pipeline for credit risk assessment. The architecture follows industry best practices for ML model deployment and monitoring.

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         Data Sources                             │
│  (CSV Files / Database / API / Real-time Streams)               │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Data Ingestion Layer                          │
│  - Synthetic Data Generator (Development)                        │
│  - Data Validation                                               │
│  - Feature Engineering                                           │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Model Training Layer                          │
│  ┌──────────────┐  ┌──────────────┐                            │
│  │  Logistic    │  │  Gradient    │                            │
│  │  Regression  │  │  Boosting    │                            │
│  └──────────────┘  └──────────────┘                            │
│           │               │                                      │
│           ▼               ▼                                      │
│  ┌─────────────────────────────┐                               │
│  │  Isotonic Calibration       │                               │
│  └─────────────────────────────┘                               │
│           │                                                      │
│           ▼                                                      │
│  ┌─────────────────────────────┐                               │
│  │  Model Artifacts (Pickle)   │                               │
│  └─────────────────────────────┘                               │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Scoring Layer                                 │
│  ┌──────────────────────────┐  ┌──────────────────────────┐   │
│  │  Probability Prediction  │  │  Scorecard Mapping       │   │
│  │  (0.0 - 1.0)            │→ │  (PDO Methodology)       │   │
│  └──────────────────────────┘  └──────────────────────────┘   │
│                     │                                            │
│                     ▼                                            │
│  ┌─────────────────────────────────────────────────────┐       │
│  │  Credit Scores + Risk Bands + Decisions             │       │
│  └─────────────────────────────────────────────────────┘       │
└────────────────────┬────────────────────────────────────────────┘
                     │
         ┌───────────┴───────────┐
         ▼                       ▼
┌──────────────────┐    ┌──────────────────┐
│   Batch Scoring  │    │   Real-time API  │
│   (CLI Script)   │    │   (FastAPI)      │
└──────────────────┘    └──────────────────┘
         │                       │
         └───────────┬───────────┘
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Monitoring Layer                              │
│  ┌──────────────────┐  ┌──────────────────┐                    │
│  │  Feature Drift   │  │  Score Drift     │                    │
│  │  (PSI Analysis)  │  │  (PSI Analysis)  │                    │
│  └──────────────────┘  └──────────────────┘                    │
│           │                     │                                │
│           ▼                     ▼                                │
│  ┌─────────────────────────────────────┐                       │
│  │  Alerts & Monitoring Reports        │                       │
│  └─────────────────────────────────────┘                       │
└─────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Data Ingestion Layer

**Purpose**: Ingest, validate, and prepare data for modeling

**Components**:
- `synthetic_generator.py`: Generates realistic credit application data
- Data validation routines
- Feature engineering pipeline

**Key Features**:
- Generates 16 credit-relevant features
- Configurable default rate
- Time-stamped applications for temporal analysis

### 2. Model Training Layer

**Purpose**: Train, calibrate, and persist ML models

**Components**:
- `train.py`: Model training orchestration
- Time-based data splitting (60/20/20)
- Model calibration using isotonic regression
- Comprehensive metric calculation

**Models Supported**:
1. **Logistic Regression**: Interpretable linear model
2. **Gradient Boosting**: High-performance ensemble model

**Metrics**:
- AUC-ROC: Discrimination ability
- KS Statistic: Population separation
- Brier Score: Probability calibration
- Lift@10%: Business value capture

### 3. Scoring Layer

**Purpose**: Convert model predictions to actionable credit scores

**Components**:
- `scorecard.py`: PDO-based score mapping
- `batch_score.py`: High-throughput batch scoring

**Scorecard Configuration**:
- SCORE0: 600 (reference score)
- ODDS0: 50 (good:bad ratio at reference)
- PDO: 20 (points to double odds)

**Formula**:
```
Score = SCORE0 - (PDO / ln(2)) × ln(Odds / ODDS0)
Odds = (1 - P) / P
```

### 4. Deployment Layers

#### Batch Scoring (CLI)
- High-throughput processing
- CSV input/output
- Configurable scorecard parameters
- Risk band assignment

#### Real-time API (FastAPI)
- RESTful endpoints
- Interactive documentation (Swagger)
- Health monitoring
- Single and batch scoring

**Endpoints**:
- `GET /health`: Service health check
- `POST /score`: Score single application
- `POST /score/batch`: Score multiple applications
- `GET /docs`: Interactive API documentation

### 5. Monitoring Layer

**Purpose**: Detect model degradation and data drift

**Components**:
- `drift.py`: PSI-based drift detection
- Feature monitoring
- Score distribution monitoring
- Automated alerting

**PSI Thresholds**:
- < 0.1: No significant change
- 0.1-0.2: Moderate change (investigate)
- ≥ 0.2: Significant change (action required)

## Data Flow

### Training Flow

1. Generate or load data
2. Time-based split (train/cal/test)
3. Train base models
4. Calibrate on calibration set
5. Evaluate on test set
6. Persist models and metrics

### Scoring Flow

1. Load trained model
2. Receive application data
3. Extract and validate features
4. Predict default probability
5. Map to credit score
6. Assign risk band and decision
7. Return results

### Monitoring Flow

1. Establish baseline statistics
2. Collect current data
3. Calculate PSI for features and scores
4. Generate alerts for significant drift
5. Produce monitoring reports

## Technology Stack

- **Language**: Python 3.8+
- **ML Framework**: scikit-learn
- **API Framework**: FastAPI
- **Data Processing**: pandas, numpy
- **Serialization**: pickle (models), JSON (configs)

## Design Principles

1. **Modularity**: Each component is independent and testable
2. **Reproducibility**: Random seeds and versioning
3. **Monitoring**: Built-in drift detection
4. **Scalability**: Batch and real-time modes
5. **Maintainability**: Clear separation of concerns
6. **Documentation**: Comprehensive docstrings and examples

## Production Considerations

### Scalability
- Implement distributed scoring (Spark, Dask)
- Database integration for persistence
- Caching for frequently accessed data

### Reliability
- Model versioning and rollback
- A/B testing framework
- Circuit breakers for dependencies
- Graceful degradation

### Security
- API authentication (OAuth2, JWT)
- Input validation and sanitization
- Rate limiting
- Audit logging

### Compliance
- Fair lending compliance (FCRA, ECOA)
- Model governance documentation
- Explainability features (SHAP, LIME)
- Bias detection and mitigation

## Future Enhancements

1. **Advanced Models**: XGBoost, LightGBM, neural networks
2. **AutoML**: Automated hyperparameter tuning
3. **Feature Store**: Centralized feature management
4. **Real-time Features**: Streaming feature computation
5. **A/B Testing**: Champion-challenger framework
6. **Model Explainability**: SHAP values, reason codes
7. **Advanced Monitoring**: Performance tracking, fairness metrics
8. **Database Integration**: PostgreSQL, MongoDB support
