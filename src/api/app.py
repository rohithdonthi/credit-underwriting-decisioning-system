"""
FastAPI application for credit scoring service.
Provides /score and /health endpoints.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.train import CreditModel
from scoring.scorecard import ScorecardMapper

app = FastAPI(
    title="Credit Underwriting API",
    description="API for credit underwriting decisioning",
    version="1.0.0"
)

# Global variables for model and scorecard
model = None
scorecard_mapper = None


class CreditApplication(BaseModel):
    """Credit application data model."""
    application_id: Optional[str] = Field(None, description="Application ID")
    credit_score: float = Field(..., ge=300, le=850, description="Credit score (300-850)")
    num_credit_lines: int = Field(..., ge=0, description="Number of credit lines")
    credit_utilization: float = Field(..., ge=0, le=1, description="Credit utilization ratio")
    num_delinquencies: int = Field(..., ge=0, description="Number of delinquencies")
    months_since_last_delinq: float = Field(..., description="Months since last delinquency (-1 if none)")
    annual_income: float = Field(..., gt=0, description="Annual income")
    employment_length_years: float = Field(..., ge=0, description="Employment length in years")
    debt_to_income: float = Field(..., ge=0, le=1, description="Debt to income ratio")
    loan_amount: float = Field(..., gt=0, description="Requested loan amount")
    loan_term_months: int = Field(..., gt=0, description="Loan term in months")
    interest_rate: float = Field(..., gt=0, description="Interest rate")
    num_inquiries_6m: int = Field(..., ge=0, description="Number of inquiries in last 6 months")
    revolving_balance: float = Field(..., ge=0, description="Revolving balance")
    has_mortgage: int = Field(..., ge=0, le=1, description="Has mortgage (0/1)")
    has_car_loan: int = Field(..., ge=0, le=1, description="Has car loan (0/1)")
    
    class Config:
        json_schema_extra = {
            "example": {
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
            }
        }


class ScoringResponse(BaseModel):
    """Scoring response model."""
    application_id: Optional[str]
    default_probability: float
    credit_score: int
    risk_band: str
    decision: str


class BatchScoringRequest(BaseModel):
    """Batch scoring request model."""
    applications: List[CreditApplication]


class BatchScoringResponse(BaseModel):
    """Batch scoring response model."""
    results: List[ScoringResponse]
    summary: Dict


@app.on_event("startup")
async def load_model():
    """Load model and scorecard mapper on startup."""
    global model, scorecard_mapper
    
    model_path = os.getenv('MODEL_PATH', 'models/gradient_boosting_model.pkl')
    
    try:
        model = CreditModel.load(model_path)
        scorecard_mapper = ScorecardMapper(score0=600, odds0=50, pdo=20)
        print(f"Model loaded successfully from {model_path}")
    except Exception as e:
        print(f"Warning: Could not load model: {e}")
        print("API will start but scoring will not work until model is loaded")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    model_loaded = model is not None
    
    return {
        "status": "healthy" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "model_type": model.model_type if model_loaded else None,
        "training_date": model.training_date if model_loaded else None
    }


@app.post("/score", response_model=ScoringResponse)
async def score_application(application: CreditApplication):
    """
    Score a single credit application.
    
    Args:
        application: Credit application data
        
    Returns:
        Scoring result with probability, score, and decision
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please check model path and restart service."
        )
    
    try:
        # Convert to features
        features = {
            'credit_score': application.credit_score,
            'num_credit_lines': application.num_credit_lines,
            'credit_utilization': application.credit_utilization,
            'num_delinquencies': application.num_delinquencies,
            'months_since_last_delinq': application.months_since_last_delinq,
            'annual_income': application.annual_income,
            'employment_length_years': application.employment_length_years,
            'debt_to_income': application.debt_to_income,
            'loan_amount': application.loan_amount,
            'loan_term_months': application.loan_term_months,
            'interest_rate': application.interest_rate,
            'num_inquiries_6m': application.num_inquiries_6m,
            'revolving_balance': application.revolving_balance,
            'has_mortgage': application.has_mortgage,
            'has_car_loan': application.has_car_loan
        }
        
        # Ensure features are in correct order
        X = np.array([[features[col] for col in model.feature_names]])
        
        # Score
        prob = float(model.predict_proba(X)[0])
        score = int(scorecard_mapper.prob_to_score(np.array([prob]))[0])
        
        # Determine risk band
        if score >= 700:
            risk_band = "Very Low"
        elif score >= 650:
            risk_band = "Low"
        elif score >= 600:
            risk_band = "Medium"
        elif score >= 550:
            risk_band = "High"
        else:
            risk_band = "Very High"
        
        # Make decision
        if prob < 0.10:
            decision = "Approve"
        elif prob < 0.20:
            decision = "Review"
        else:
            decision = "Decline"
        
        return ScoringResponse(
            application_id=application.application_id,
            default_probability=round(prob, 4),
            credit_score=score,
            risk_band=risk_band,
            decision=decision
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error scoring application: {str(e)}"
        )


@app.post("/score/batch", response_model=BatchScoringResponse)
async def score_batch(request: BatchScoringRequest):
    """
    Score multiple credit applications.
    
    Args:
        request: Batch of credit applications
        
    Returns:
        Batch scoring results with summary
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please check model path and restart service."
        )
    
    results = []
    for application in request.applications:
        result = await score_application(application)
        results.append(result)
    
    # Calculate summary
    summary = {
        "total_applications": len(results),
        "approved": sum(1 for r in results if r.decision == "Approve"),
        "declined": sum(1 for r in results if r.decision == "Decline"),
        "review": sum(1 for r in results if r.decision == "Review"),
        "mean_probability": np.mean([r.default_probability for r in results]),
        "mean_score": np.mean([r.credit_score for r in results])
    }
    
    return BatchScoringResponse(
        results=results,
        summary=summary
    )


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Credit Underwriting API",
        "version": "1.0.0",
        "endpoints": {
            "/health": "Health check",
            "/score": "Score single application (POST)",
            "/score/batch": "Score multiple applications (POST)",
            "/docs": "API documentation"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
