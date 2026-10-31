"""Model transformers, model loading, prediction, and artifact export.

IMPORTANT: import the custom transformers from this module in the training
notebook before fitting the final pipeline. This makes serialized pipelines
loadable by the Streamlit app.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.config import MODEL_PATH, MODEL_FEATURES_PATH, FEATURE_IMPORTANCE_PATH


class LoanTypeMedianImputer(BaseEstimator, TransformerMixin):
    """Impute interest_rate_pct using training-fold loan_type medians."""
    def __init__(self, group_col="loan_type", value_col="interest_rate_pct"):
        self.group_col = group_col
        self.value_col = value_col

    def fit(self, X, y=None):
        self.group_medians_ = X.groupby(self.group_col)[self.value_col].median()
        self.overall_median_ = X[self.value_col].median()
        if pd.isna(self.overall_median_):
            raise ValueError(f"No observed training values for {self.value_col}.")
        return self

    def transform(self, X):
        X = X.copy()

        # Make sure this column can safely hold numeric values.
        X[self.value_col] = pd.to_numeric(
            X[self.value_col], errors="coerce"
        ).astype("float64")

        missing = X[self.value_col].isna()

        if missing.any():
            mapped = X.loc[missing, self.group_col].map(
                self.group_medians_
            )

            # Fill unknown groups with the overall training median.
            fill_values = pd.to_numeric(
                mapped, errors="coerce"
            ).fillna(self.overall_median_)

            # Assign a NumPy float array to avoid pandas string-array
            # dtype conflicts during indexed assignment.
            X.loc[missing, self.value_col] = fill_values.to_numpy(
                dtype="float64"
            )

        return X


class HierarchicalDPDImputer(BaseEstimator, TransformerMixin):
    """Impute days_past_due using customer, account_status, then overall medians."""

    def __init__(
        self,
        customer_col="customer_id",
        group_col="account_status",
        value_col="days_past_due",
        min_customer_values=2,
        add_indicator=True,
    ):
        self.customer_col = customer_col
        self.group_col = group_col
        self.value_col = value_col
        self.min_customer_values = min_customer_values
        self.add_indicator = add_indicator

    def fit(self, X, y=None):
        X = X.copy()

        X[self.value_col] = pd.to_numeric(
            X[self.value_col], errors="coerce"
        ).astype("float64")

        observed = X.dropna(subset=[self.value_col])

        if observed.empty:
            raise ValueError(f"No observed training values for {self.value_col}.")

        self.overall_median_ = observed[self.value_col].median()

        # Account-status medians, learned from observed training values.
        self.group_medians_ = observed.groupby(
            self.group_col
        )[self.value_col].median()

        # Customer medians only when enough observed values exist.
        customer_stats = observed.groupby(
            self.customer_col
        )[self.value_col].agg(["median", "count"])

        self.customer_medians_ = customer_stats.loc[
            customer_stats["count"] >= self.min_customer_values,
            "median",
        ]

        return self

    def transform(self, X):
        X = X.copy()

        X[self.value_col] = pd.to_numeric(
            X[self.value_col], errors="coerce"
        ).astype("float64")

        missing = X[self.value_col].isna()

        if self.add_indicator:
            X[f"{self.value_col}_missing"] = missing.astype("int8")

        if missing.any():
            # First fallback: customer's median
            customer_fill = X.loc[missing, self.customer_col].map(
                self.customer_medians_
            )

            # Second fallback: account-status median
            group_fill = X.loc[missing, self.group_col].map(
                self.group_medians_
            )

            # Final fallback: overall training median
            fill_values = (
                customer_fill
                .fillna(group_fill)
                .fillna(self.overall_median_)
            )

            X.loc[missing, self.value_col] = fill_values.to_numpy(
                dtype="float64"
            )

        return X

class EmploymentYearsMedianImputer(BaseEstimator, TransformerMixin):
    """Impute years_employed using training-fold employment_type medians."""
    def __init__(self, group_col="employment_type", value_col="years_employed"):
        self.group_col = group_col
        self.value_col = value_col

    def fit(self, X, y=None):
        self.group_medians_ = X.groupby(self.group_col)[self.value_col].median()
        self.overall_median_ = X[self.value_col].median()
        if pd.isna(self.overall_median_):
            raise ValueError(f"No observed training values for {self.value_col}.")
        return self

    def transform(self, X):
        X = X.copy()

        X[self.value_col] = pd.to_numeric(
            X[self.value_col], errors="coerce"
        ).astype("float64")

        missing = X[self.value_col].isna()

        if missing.any():
            mapped = X.loc[missing, self.group_col].map(
                self.group_medians_
            )

            fill_values = pd.to_numeric(
                mapped, errors="coerce"
            ).fillna(self.overall_median_)

            X.loc[missing, self.value_col] = fill_values.to_numpy(
                dtype="float64"
            )

        return X


def save_model_artifacts(
    model,
    feature_names: list[str],
    model_path: Path = MODEL_PATH,
    features_path: Path = MODEL_FEATURES_PATH,
    importance: pd.DataFrame | None = None,
    importance_path: Path = FEATURE_IMPORTANCE_PATH,
) -> None:
    """Save a fitted pipeline, its exact input columns, and optional importance."""
    model_path.parent.mkdir(parents=True, exist_ok=True)
    features_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    with open(features_path, "w", encoding="utf-8") as f:
        json.dump(list(feature_names), f, indent=2)

    if importance is not None:
        importance_path.parent.mkdir(parents=True, exist_ok=True)
        importance.to_csv(importance_path, index=False)


def load_model_artifacts(
    model_path: Path = MODEL_PATH,
    features_path: Path = MODEL_FEATURES_PATH,
):
    """Return (fitted_pipeline, expected_feature_names)."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not features_path.exists():
        raise FileNotFoundError(f"Feature list not found: {features_path}")
    model = joblib.load(model_path)
    with open(features_path, "r", encoding="utf-8") as f:
        features = json.load(f)
    return model, features


def predict_default_probability(
    model,
    account_rows: pd.DataFrame,
    feature_names: list[str],
) -> np.ndarray:
    """Predict probability of class 1, preserving the training feature order."""
    missing = [col for col in feature_names if col not in account_rows.columns]
    if missing:
        raise ValueError(f"Input rows are missing model features: {missing}")

    X = account_rows.loc[:, feature_names].copy()
    X = X.replace([np.inf, -np.inf], np.nan)

    if not hasattr(model, "predict_proba"):
        raise TypeError("The fitted model does not implement predict_proba.")

    classes = list(getattr(model, "classes_", []))
    if classes and 1 not in classes:
        raise ValueError("The fitted model has no positive class (1).")
    positive_index = classes.index(1) if classes else 1

    probabilities = model.predict_proba(X)[:, positive_index]
    return np.asarray(probabilities, dtype=float)