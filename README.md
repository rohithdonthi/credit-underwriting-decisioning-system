# credit-underwriting-decisioning-system
Production-style underwriting decisioning system: generates a calibrated creditworthiness probability, maps it to a scorecard score, produces decision + reason codes, and logs monitoring-ready artifacts (metrics, score distribution, drift/PSI) using time-based validation.

## What this system does (end-to-end)

This repository implements a production-style **underwriting decisioning pipeline** that:
1) **Trains** a creditworthiness model using **time-based validation** (to reduce leakage)
2) **Calibrates** probabilities (sigmoid calibration) for decision reliability
3) **Scores** applicants to produce:
   - `p_creditworthy` (P(good))
   - a **scorecard score** (PDO/ODDS scaling)
   - a **decision** (APPROVE / REVIEW / DECLINE)
4) Writes **monitoring-ready artifacts** (metrics + drift/PSI checks)

## UI / Demo

- **Streamlit UI (local):**
  ```bash
  streamlit run app/streamlit_app.py

