Production-style underwriting decisioning pipeline that generates a calibrated creditworthiness probability, maps it to a scorecard score, produces decision + reason codes, and logs monitoring-ready artifacts (drift/PSI, metrics, score distribution) using time-based validation.
Built as a public recreation of production credit risk work across 50,000+ accounts — anonymised and reproduced with synthetic data.

📊 Model Performance
Metric      Value           Benchmark 
AUC (ROC)   0.87            Industry standard: 0.75+
KS Statistic0.42            Strong separation
RMSE Reduction(LGD)22%      vs prior baseline
Accounts scored50,000+      —


Business impact demonstrated: $180K annualised savings · 20+ loan officers onboarded

🏗️ What This System Does (End-to-End)
Raw Credit Data
Behavioral · Bureau · Transactional
              │
              ▼
     Data Pipeline & Feature Engineering
     (APR standardisation, QC validation,
      multi-source integration)
              │
         ┌────┴────┐
         ▼          ▼
  Logistic         Gradient Boosting
  Regression       Loss Model
  (PD Model)       (LGD Model)
  AUC: 0.87        RMSE: -22%
  KS: 0.42
         └────┬────┘
              ▼
     Probability Calibration
     (sigmoid — ensures P(good) is reliable)
              │
              ▼
     Scorecard Mapping
     (PDO/ODDS scaling → interpretable score)
              │
              ▼
     Decision Engine
     APPROVE / REVIEW / DECLINE + Reason Codes
              │
              ▼
     Monitoring Artifacts
     PSI Drift Detection · Score Distribution · Metrics Log

🚀 Quickstart (3 commands)
bashpython -m src.data.make_dataset --out data/sample/credit_sample.csv
python -m src.models.train --data data/sample/credit_sample.csv
streamlit run app/streamlit_app.py
Or try the live hosted demo — no setup needed.

📁 Repository Structure
credit-underwriting-decisioning-system/
├── src/
│   ├── data/
│   │   └── make_dataset.py        # Synthetic data generation & preprocessing
│   ├── models/
│   │   └── train.py               # Model training with time-based validation
│   └── monitoring/
│       └── drift_detection.py     # PSI-based drift detection
├── app/
│   └── streamlit_app.py           # Interactive demo UI
├── Credit_score.ipynb             # Full analysis notebook
├── requirements.txt
├── .gitignore
└── README.md

🔑 Key Design Decisions
Time-based validation (not random split)
Prevents data leakage — model always trained on past data, evaluated on future data. Mirrors real production deployment.
Probability calibration
Raw model scores are not probabilities. Sigmoid calibration ensures P(good) = 0.8 actually means 80% of applicants with that score are creditworthy — critical for setting reliable approval thresholds.
PSI-based drift detection
Population Stability Index monitors input feature distributions over time. Flags when the applicant population shifts before it degrades model performance. Drift alert triggers at PSI > 0.2 (industry standard).
Scorecard mapping
Decision-makers need interpretable scores, not probability decimals. PDO/ODDS scaling maps probabilities to a familiar scorecard range with reason codes explaining each decision.

🛠️ Tech Stack
Category          Tools
ML Models         scikit-learn, XGBoost, LightGBM
Data Processing   pandas, NumPy
Calibration       scikit-learn CalibratedClassifierCV
Monitoring        Custom PSI implementation
UI / Demo         Streamlit
Visualisation     Matplotlib, Seaborn
Environment       Python 3.10

📄 License
MIT License — see LICENSE for details.

