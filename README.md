# Synthetic Credit Risk Analytics Dashboard

A Streamlit application for exploring a synthetic loan portfolio, reviewing account-level risk, and evaluating a default prediction model. The dashboard includes portfolio analytics, an LLM-powered **Ask the Portfolio** feature, and an anomaly-review workflow.

> **Disclaimer:** The dataset is synthetic. Findings are associative, not causal, and should not be interpreted as evidence of real-world lending performance.

## Contents

- [Overview](#overview)
- [Features](#features)
- [Repository structure](#repository-structure)
- [Requirements](#requirements)
- [Setup and run](#setup-and-run)
- [LLM API key](#llm-api-key)
- [Data and modelling workflow](#data-and-modelling-workflow)
- [Model results](#model-results)
- [Dashboard workflows](#dashboard-workflows)
- [Assumptions and limitations](#assumptions-and-limitations)
- [Production considerations](#production-considerations)
- [Where the assignment requirements live](#where-the-assignment-requirements-live)

## Overview

The project combines synthetic loan, customer, bureau, repayment, and collection data for portfolio analysis and account-level risk review.

The workflow:
1. Loads and validates source tables.
2. Aggregates source records and creates account-level features.
3. Produces portfolio analytics and trains a default prediction model.
4. Presents the results through a Streamlit dashboard.

Logistic Regression and XGBoost are compared. Logistic Regression is selected based on stronger grouped cross-validation performance.

## Features

### Portfolio analytics
- Portfolio-level summaries and KPIs
- Exposure and default-rate breakdowns by segment
- Account exploration and model risk scores
- Feature-importance views

### Ask the Portfolio
Users can ask natural-language questions about the portfolio. The feature uses an LLM to interpret the question and answer using portfolio information made available by the application. It is intended for portfolio analysis, not for making lending decisions or supporting conclusions beyond the available data.

### Anomaly review
The dashboard surfaces accounts for analyst review and presents review information. An anomaly flag is a review signal—not proof of fraud or default.

## Repository structure

```text
.
├── app/
│   └── app.py                 # Streamlit dashboard entry point
├── data/
│   ├── raw/                   # Source CSV files
│   └── processed/             # Generated data for analytics/modeling
├── models/                    # Saved model and related artifacts
├── src/
│   ├── config.py              # Data, model and application paths/configuration
│   └── ...                    # Ingestion, feature engineering, modeling and analytics
├── docs/
│   └── architecture.pdf       # High-level architecture diagram
├── requirements.txt           # Python dependencies
├── .env.example               # Example environment configuration, if included
└── README.md
```

Generated datasets and model artifacts may be created during the build step. The exact filenames depend on the implementation.

## Requirements

- Python **3.13**
- `pip`
- Source CSV files in `data/raw/`
- An LLM API key for Ask the Portfolio; other dashboard features should remain usable without it

Use a virtual environment to isolate dependencies.

## Setup and run

Run these commands from the repository root. These examples use PowerShell on Windows.

```powershell
python -m pip install -r requirements.txt
python -m src.run_pipeline
python -m streamlit run app/app.py
```

Open the local URL printed by Streamlit.

## LLM API key

Ask the Portfolio requires an API key for the LLM provider configured in the application.

1. Create a `.env` file in the repository root.
2. Add the key using the exact environment-variable name read by the application.
3. Keep `.env` out of version control; never commit a real key.

For example, if the application uses Gemini:

```dotenv
GEMINI_API_KEY=your_api_key_here
```

Use the variable name expected by the code; the example above must be changed if the application reads a different name.

### If no key is configured

The LLM-powered Ask the Portfolio feature should be unavailable and show a clear message that an API key is required. Portfolio analytics, account exploration, model scores, and anomaly review should continue to work.

An API key should not be required to build the dataset or train the model unless the build code explicitly calls an external LLM.

## Data and modelling workflow

### Data preparation

The pipeline profiles the source tables, handles missing values and duplicates, aggregates bureau, repayment, and collection records, and joins the data at account level.

The modelling dataset contains:
- **14,685 accounts**
- **10,000 customers**
- **15 features**
- **16.1% default rate**

### Feature engineering

Features include:
- **Affordability:** annual income, loan amount, loan-to-income ratio, and EMI payment ratio
- **Repayment behavior:** average days past due (DPD), late-payment count, and missed-payment count
- **Bureau risk:** average DPD, total outstanding, active-account ratio, and outstanding balance per active account
- **Borrower attributes:** age, employment type, years employed, and loan type

### Train/test split and validation

The split is grouped by `customer_id` to prevent the same customer appearing in both training and test sets.

- Training: 11,737 accounts from 8,000 customers
- Test: 2,948 accounts from 2,000 customers
- Customer overlap: 0
- Grouped cross-validation: 5 folds

Preprocessing is fitted within each training fold. Random oversampling is applied only to training folds, not validation or test data.

### Model selection

Logistic Regression and XGBoost were compared using grouped cross-validation. Logistic Regression achieved higher mean ROC-AUC and PR-AUC and was selected as the final model.

The classification threshold was selected using out-of-fold training predictions and then held fixed for test evaluation.

### Held-out results

**Selected model: Logistic Regression**

| Metric | Held-out result |
|---|---:|
| ROC-AUC | 0.6881 |
| KS | 0.2728 |
| Gini | 0.3763 |
| PR-AUC | 0.3269 |
| Classification threshold | 0.1952 |
| Precision | 0.2842 |
| Recall | 0.5064 |
| F1 | 0.3641 |

The threshold reflects a precision–recall trade-off and should be reconsidered for real deployment based on business costs and review capacity.

## Dashboard workflows

### Portfolio exploration
Use portfolio views to review summary metrics and compare risk across segments. Interpret segment-level patterns alongside account counts and exposure.

### Account-level review
Use the account explorer to inspect available account attributes, predicted risk score, and model drivers. A score is an estimate, not a certainty about an account's future outcome.

### Ask the Portfolio
Ask portfolio-level questions using information supplied to the LLM by the application. Answers may be limited by available data and should be checked against the dashboard's underlying metrics.

### Anomaly review
Review flagged accounts, inspect the reason and priority information, and record an analyst decision. A flagged account requires review; it is not automatically classified as fraudulent or defaulted.

## Assumptions and limitations

- The dataset and labels are synthetic, so model results may not generalize to real lending.
- Findings are associative, not causal.
- Source data and target labels are assumed to be accurate and consistently defined.
- All model inputs must be available at the intended scoring time. Post-origination repayment or collection data must not leak into an origination model.
- The selected threshold is specific to this dataset and is not a production decision threshold.
- Model scores and anomaly flags support review; they should not replace governed lending decisions.

## Production considerations

Before production use:
- Validate bank-specific data definitions and point-in-time feature availability.
- Establish governed data access and secure deployment/API integration.
- Implement role-based access controls, secrets management, and audit logs.
- Monitor data quality, missing values, feature drift, and score drift.
- Evaluate model performance when outcomes mature, including ROC-AUC, PR-AUC, KS, Gini, precision, recall, and calibration.
- Review performance and error rates across relevant customer groups for fairness.
- Define alert thresholds, owners, and procedures for investigation, rollback, retraining, and revalidation.

## Where the assignment requirements live

| Requirement | Repository location |
|---|---|
| Source data | `data/raw/` |
| Data ingestion and preparation | `src/` — ingestion/build modules |
| Feature engineering | `src/` — feature engineering modules |
| Model training and evaluation | `src/` — modelling/training modules |
| Saved model artifacts | `models/` |
| Dashboard | `app/app.py` |
| Portfolio analytics and account views | `app/app.py` and supporting modules under `src/` |
| Ask the Portfolio LLM integration | `app/app.py` and the supporting LLM module under `src/` |
| Anomaly detection and review workflow | Supporting anomaly/review modules under `src/` and dashboard in `app/app.py` |
| Architecture diagram | `docs/architecture.pdf` |
| Dependencies | `requirements.txt` |
| Environment-variable example | `.env.example`, if included |
| Setup and run instructions | This `README.md` |

