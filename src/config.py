"""Centralized project paths and application configuration."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

CUSTOMERS_PATH = RAW_DATA_DIR / "customers.csv"
LOAN_ACCOUNTS_PATH = RAW_DATA_DIR / "loan_accounts.csv"
BUREAU_CREDIT_PATH = RAW_DATA_DIR / "bureau_credit.csv"
REPAYMENT_HISTORY_PATH = RAW_DATA_DIR / "repayment_history.csv"
COLLECTIONS_EVENTS_PATH = RAW_DATA_DIR / "collections_events.csv"

ACCOUNT_DATA_PATH = PROCESSED_DATA_DIR / "account_data.csv"
MODEL_PATH = MODELS_DIR / "best_model.joblib"
MODEL_FEATURES_PATH = MODELS_DIR / "model_features.json"
FEATURE_IMPORTANCE_PATH = MODELS_DIR / "feature_importance.csv"

for directory in (PROCESSED_DATA_DIR, MODELS_DIR):
    directory.mkdir(parents=True, exist_ok=True)
# Authoritative exclusions for the scripted model-training workflow.
# Keep exploratory EDA/validation in the notebook; production-style model
# training and inference consume this same feature contract.
SELECTED_FEATURES = [
    "annual_income",
    "loan_type",
    "years_employed",
    "repayment_avg_dpd",
    "late_payment_count",
    "missed_payment_count",
    "loan_amount",
    "loan_to_income_ratio",
    "bureau_avg_dpd",
    "bureau_total_outstanding",
    "employment_type",
    "age",
    "bureau_active_account_ratio",
    "bureau_outstanding_per_active_account",
    "emi_payment_ratio",
]
