
"""Train and evaluate credit-risk default models.

Run from the project root:
    python -m src.train

Uses customer-grouped train/test splitting and grouped cross-validation
to prevent accounts belonging to the same customer from crossing splits.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    GroupShuffleSplit,
    RandomizedSearchCV,
    StratifiedGroupKFold,
    cross_validate,
    cross_val_predict,
)
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from xgboost import XGBClassifier

from src.config import (
    ACCOUNT_DATA_PATH,
    MODELS_DIR,
    MODEL_PATH,
    MODEL_FEATURES_PATH,
    FEATURE_IMPORTANCE_PATH,
    PROJECT_ROOT,
    SELECTED_FEATURES
)
from src.ingestion import build_and_save_account_data
from src.model import (
    LoanTypeMedianImputer,
    EmploymentYearsMedianImputer,
    save_model_artifacts,
)


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

RANDOM_STATE = 42
TEST_SIZE = 0.20
N_SPLITS = 5

# Set to 30 to reproduce the notebook's randomized search.
XGB_SEARCH_ITERATIONS = 30

# Permutation importance can be time-consuming.
PERMUTATION_REPEATS = 10

METRICS_PATH = MODELS_DIR / "model_training_metrics.csv"
METADATA_PATH = MODELS_DIR / "model_training_metadata.json"


# Shared feature contract; see src.config.DROP_COLUMNS.

# ---------------------------------------------------------
# Data loading and feature preparation
# ---------------------------------------------------------

def load_account_data() -> pd.DataFrame:
    """Load processed account data, creating it if it does not exist."""

    if not ACCOUNT_DATA_PATH.exists():
        print("Processed account data not found.")
        print("Building account-level dataset from raw CSV files...")
        return build_and_save_account_data()

    return pd.read_csv(ACCOUNT_DATA_PATH)


def prepare_features(
    account_data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Create model inputs, target and customer grouping variable."""

    required = {
        "account_id",
        "customer_id",
        "target_default",
    }

    missing = required - set(account_data.columns)
    if missing:
        raise ValueError(
            f"Account data is missing required columns: {sorted(missing)}"
        )

    if account_data["account_id"].duplicated().any():
        raise ValueError("Expected one row per loan account.")

    if account_data["target_default"].isna().any():
        raise ValueError("Target contains missing values.")

    if not account_data["target_default"].isin([0, 1]).all():
        raise ValueError("target_default must contain only 0 and 1.")

    if account_data["customer_id"].isna().any():
        raise ValueError("customer_id contains missing values.")

    y = account_data["target_default"].astype(int)
    groups = account_data["customer_id"].copy()

    
    # Fail clearly if a selected feature is missing from the processed data
    missing_features = [
        col for col in SELECTED_FEATURES
        if col not in account_data.columns
    ]
    if missing_features:
        raise ValueError(f"Missing selected features: {missing_features}")

    X = account_data.loc[:, SELECTED_FEATURES].copy()

    X = X.replace([np.inf, -np.inf], np.nan)

    # Remove columns that contain no usable values.
    all_missing = X.columns[X.isna().all()].tolist()
    if all_missing:
        print("Dropping entirely missing columns:", all_missing)
        X = X.drop(columns=all_missing)

    if X.empty:
        raise ValueError("No model features remain after exclusions.")

    if X.select_dtypes(
        include=["datetime", "datetimetz"]
    ).shape[1]:
        datetime_cols = X.select_dtypes(
            include=["datetime", "datetimetz"]
        ).columns.tolist()
        raise ValueError(
            f"Datetime columns need explicit handling: {datetime_cols}"
        )

    if not X.index.equals(y.index) or not X.index.equals(groups.index):
        raise ValueError("Feature, target and group indexes are misaligned.")

    print("\nModel dataset")
    print("Accounts:", len(X))
    print("Customers:", groups.nunique())
    print("Features:", X.shape[1])
    print("Default rate:", round(y.mean() * 100, 2), "%")
    print("Feature columns:", X.columns.tolist())

    return X, y, groups


# ---------------------------------------------------------
# Train/test split
# ---------------------------------------------------------

def grouped_train_test_split(X, y, groups):
    """Split by customer so a customer's accounts stay together."""

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    train_idx, test_idx = next(
        splitter.split(X, y, groups=groups)
    )

    X_train = X.iloc[train_idx].copy()
    X_test = X.iloc[test_idx].copy()

    y_train = y.iloc[train_idx].copy()
    y_test = y.iloc[test_idx].copy()

    groups_train = groups.iloc[train_idx].copy()
    groups_test = groups.iloc[test_idx].copy()

    if not set(groups_train).isdisjoint(set(groups_test)):
        raise AssertionError("Customer leakage detected across train/test.")

    print("\nGrouped train/test split")
    print("Train accounts:", len(X_train))
    print("Test accounts:", len(X_test))
    print("Train customers:", groups_train.nunique())
    print("Test customers:", groups_test.nunique())
    print("Customer overlap:", len(set(groups_train) & set(groups_test)))
    print("Train default rate:", round(y_train.mean() * 100, 2), "%")
    print("Test default rate:", round(y_test.mean() * 100, 2), "%")

    return X_train, X_test, y_train, y_test, groups_train, groups_test


# ---------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------

def build_preprocessor(X_train: pd.DataFrame):
    """Build preprocessing using training feature types only."""

    numeric_cols = X_train.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_cols = X_train.select_dtypes(
        include=["object", "category", "bool", "string"]
    ).columns.tolist()

    if not numeric_cols and not categorical_cols:
        raise ValueError("No supported numeric or categorical features.")

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    transformers = []

    if numeric_cols:
        transformers.append(
            ("num", numeric_pipeline, numeric_cols)
        )

    if categorical_cols:
        transformers.append(
            ("cat", categorical_pipeline, categorical_cols)
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

    print("\nPreprocessing")
    print("Numeric features:", len(numeric_cols))
    print("Categorical features:", len(categorical_cols))

    return preprocessor


def build_logistic_pipeline(preprocessor):
    """Create Logistic Regression pipeline."""

    return Pipeline([
        # ("loan_type_imputer", LoanTypeMedianImputer()),
        ("employment_years_imputer", EmploymentYearsMedianImputer()),
        ("preprocessor", clone(preprocessor)),
        ("sampler", "passthrough"),
        ("model", LogisticRegression(
            max_iter=3000,
            class_weight=None,
            random_state=RANDOM_STATE,
        )),
    ])


def build_xgboost_pipeline(preprocessor):
    """Create XGBoost pipeline."""

    return Pipeline([
        # ("loan_type_imputer", LoanTypeMedianImputer()),
        ("employment_years_imputer", EmploymentYearsMedianImputer()),
        ("preprocessor", clone(preprocessor)),
        ("sampler", "passthrough"),
        ("model", XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=1,
            tree_method="hist",
        )),
    ])


# ---------------------------------------------------------
# Model evaluation helpers
# ---------------------------------------------------------

def evaluate_probabilities(
    model_name: str,
    y_true,
    probabilities,
) -> dict:
    """Calculate probability-ranking metrics."""

    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)

    if len(y_true) != len(probabilities):
        raise ValueError("Target and probability lengths do not match.")

    if len(np.unique(y_true)) != 2:
        raise ValueError("Evaluation requires both target classes.")

    if not np.isfinite(probabilities).all():
        raise ValueError("Predictions contain NaN or infinity.")

    auc = roc_auc_score(y_true, probabilities)
    fpr, tpr, _ = roc_curve(y_true, probabilities)

    return {
        "Model": model_name,
        "ROC_AUC": auc,
        "PR_AUC": average_precision_score(y_true, probabilities),
        "KS": float(np.max(tpr - fpr)),
        "Gini": float(2 * auc - 1),
    }


# ---------------------------------------------------------
# Logistic Regression tuning
# ---------------------------------------------------------

def tune_logistic_regression(
    X_train, y_train, groups_train, preprocessor, cv,
):
    """Tune LR with and without random oversampling, selecting by CV ROC-AUC."""
    pipeline = build_logistic_pipeline(preprocessor)
    param_grid = {
        "sampler": [
            "passthrough",
            RandomOverSampler(
                sampling_strategy=0.2,
                random_state=RANDOM_STATE,
            ),
        ],
        "model__C": [0.01, 0.1, 1, 10, 100],
        "model__class_weight": [None],
        "model__solver": ["lbfgs"],
    }
    search = GridSearchCV(
        pipeline, param_grid, scoring="roc_auc", cv=cv, refit=True,
        n_jobs=-1, verbose=1, return_train_score=True,
    )
    print("\\nTuning Logistic Regression (no sampling and random oversampling)...")
    search.fit(X_train, y_train, groups=groups_train)
    print("Best LR parameters:", search.best_params_)
    print("Best grouped CV ROC-AUC:", round(search.best_score_, 4))
    return search


def tune_xgboost(
    X_train, y_train, groups_train, preprocessor, cv,
):
    """Tune XGBoost with and without random oversampling."""
    pipeline = build_xgboost_pipeline(preprocessor)
    param_distributions = {
        "sampler": [
            "passthrough",
            RandomOverSampler(
                sampling_strategy=0.2,
                random_state=RANDOM_STATE,
            ),
        ],
        "model__n_estimators": [200, 300, 400, 600],
        "model__max_depth": [2, 3, 4],
        "model__learning_rate": [0.02, 0.03, 0.05, 0.1],
        "model__min_child_weight": [3, 5, 8],
        "model__subsample": [0.7, 0.8, 0.9],
        "model__colsample_bytree": [0.7, 0.8, 0.9],
        "model__reg_alpha": [0, 0.01, 0.1, 1],
        "model__reg_lambda": [1, 3, 5, 10],
        "model__gamma": [0, 0.1, 0.3, 0.5],
    }
    search = RandomizedSearchCV(
        pipeline, param_distributions,
        n_iter=XGB_SEARCH_ITERATIONS, scoring="roc_auc", cv=cv,
        random_state=RANDOM_STATE, n_jobs=-1, refit=True,
        verbose=1, return_train_score=True,
    )
    print("\\nTuning XGBoost (no sampling and random oversampling)...")
    search.fit(X_train, y_train, groups=groups_train)
    print("Best XGBoost parameters:", search.best_params_)
    print("Best grouped CV ROC-AUC:", round(search.best_score_, 4))
    return search


# ---------------------------------------------------------
# Grouped CV comparison
# ---------------------------------------------------------

def cross_validate_selected_configuration(
    model,
    X_train,
    y_train,
    groups_train,
    cv,
    model_name,
):
    """Calculate grouped CV ROC-AUC and PR-AUC for a tuned pipeline."""

    results = cross_validate(
        model,
        X_train,
        y_train,
        groups=groups_train,
        cv=cv,
        scoring={
            "roc_auc": "roc_auc",
            "pr_auc": "average_precision",
        },
        n_jobs=-1,
        return_train_score=False,
    )

    return {
        "Model": model_name,
        "CV_ROC_AUC_mean": float(results["test_roc_auc"].mean()),
        "CV_ROC_AUC_std": float(results["test_roc_auc"].std()),
        "CV_PR_AUC_mean": float(results["test_pr_auc"].mean()),
        "CV_PR_AUC_std": float(results["test_pr_auc"].std()),
    }


# ---------------------------------------------------------
# Threshold selection using out-of-fold training predictions
# ---------------------------------------------------------

def select_threshold(
    selected_model,
    X_train,
    y_train,
    groups_train,
    cv,
) -> dict:
    """Select an F1 threshold from grouped out-of-fold train predictions."""

    print("\nGenerating out-of-fold training probabilities...")

    oof_probabilities = cross_val_predict(
        selected_model,
        X_train,
        y_train,
        groups=groups_train,
        cv=cv,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]

    
    from sklearn.metrics import (
        precision_score,
        recall_score,
        f1_score,
        confusion_matrix,
    )

    thresholds = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

    threshold_results = []

    for threshold in thresholds:
        predictions = (oof_probabilities >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_train, predictions
        ).ravel()

        threshold_results.append({
            "Threshold": threshold,
            "Precision": precision_score(y_train, predictions),
            "Recall": recall_score(y_train, predictions),
            "F1": f1_score(y_train, predictions),
            "False Positives": fp,
            "False Negatives": fn,
            "Accounts Flagged": fp + tp,
        })

    threshold_df = pd.DataFrame(threshold_results)

    print(threshold_df.round(4).to_string(index=False))

    precision, recall, thresholds = precision_recall_curve(
        y_train,
        oof_probabilities,
    )

    # Precision and recall contain one extra final value.
    precision = precision[:-1]
    recall = recall[:-1]

    f1_values = (
        2 * precision * recall
        / np.maximum(precision + recall, 1e-12)
    )

    if len(thresholds) == 0:
        raise ValueError("Unable to calculate a classification threshold.")

    best_idx = int(np.argmax(f1_values))

    result = {
        "threshold": float(thresholds[best_idx]),
        "oof_precision": float(precision[best_idx]),
        "oof_recall": float(recall[best_idx]),
        "oof_f1": float(f1_values[best_idx]),
    }

    print("OOF-selected threshold:", round(result["threshold"], 4))
    print("OOF precision:", round(result["oof_precision"], 4))
    print("OOF recall:", round(result["oof_recall"], 4))
    print("OOF F1:", round(result["oof_f1"], 4))

    return result


# ---------------------------------------------------------
# Final model evaluation
# ---------------------------------------------------------

def evaluate_final_model(
    model,
    model_name,
    X_train,
    X_test,
    y_train,
    y_test,
    threshold,
):
    """Evaluate the selected model on the untouched test set."""

    train_probabilities = model.predict_proba(X_train)[:, 1]
    test_probabilities = model.predict_proba(X_test)[:, 1]

    test_metrics = evaluate_probabilities(
        model_name,
        y_test,
        test_probabilities,
    )

    predictions = (
        test_probabilities >= threshold
    ).astype(int)

    classification_metrics = {
        "Accuracy": float(accuracy_score(y_test, predictions)),
        "Precision": float(
            precision_score(y_test, predictions, zero_division=0)
        ),
        "Recall": float(
            recall_score(y_test, predictions, zero_division=0)
        ),
        "F1": float(
            f1_score(y_test, predictions, zero_division=0)
        ),
        "Threshold": float(threshold),
        "Train_ROC_AUC": float(
            roc_auc_score(y_train, train_probabilities)
        ),
        "Test_ROC_AUC": float(test_metrics["ROC_AUC"]),
        "Test_PR_AUC": float(test_metrics["PR_AUC"]),
    }

    print("\nFinal test-set evaluation")
    print(pd.DataFrame([test_metrics]).round(4).to_string(index=False))

    print("\nClassification metrics")
    for metric, value in classification_metrics.items():
        print(f"{metric}: {value:.4f}")

    print("\nConfusion matrix [[TN, FP], [FN, TP]]:")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=["Non-default", "Default"],
            zero_division=0,
        )
    )

    return test_metrics, classification_metrics


# ---------------------------------------------------------
# Feature importance
# ---------------------------------------------------------

def calculate_permutation_importance(
    model,
    X_test,
    y_test,
) -> pd.DataFrame:
    """Calculate importance on original, unprocessed test features."""

    print("\nCalculating permutation feature importance...")

    result = permutation_importance(
        model,
        X_test,
        y_test,
        scoring="roc_auc",
        n_repeats=PERMUTATION_REPEATS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    importance = pd.DataFrame({
        "feature": X_test.columns,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values(
        "importance_mean",
        ascending=False,
    ).reset_index(drop=True)

    print("\nTop 20 features:")
    print(importance.head(20).round(4).to_string(index=False))

    return importance


# ---------------------------------------------------------
# Main training workflow
# ---------------------------------------------------------

def train_model() -> dict:
    """Run the complete training, selection and artifact export workflow."""

    # 1. Load data and prepare features.
    account_data = load_account_data()
    X, y, groups = prepare_features(account_data)

    # 2. Grouped train/test split.
    (
        X_train,
        X_test,
        y_train,
        y_test,
        groups_train,
        groups_test,
    ) = grouped_train_test_split(X, y, groups)

    # 3. Grouped cross-validation.
    cv = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    # 4. Verify group separation in each fold.
    print("\nValidating grouped CV folds...")

    for fold, (train_idx, val_idx) in enumerate(
        cv.split(X_train, y_train, groups=groups_train),
        start=1,
    ):
        fold_train_groups = set(groups_train.iloc[train_idx])
        fold_val_groups = set(groups_train.iloc[val_idx])

        if not fold_train_groups.isdisjoint(fold_val_groups):
            raise AssertionError(
                f"Customer leakage detected in fold {fold}."
            )

        print(
            f"Fold {fold}: "
            f"train={len(train_idx)}, "
            f"validation={len(val_idx)}, "
            f"validation default rate="
            f"{y_train.iloc[val_idx].mean():.3%}"
        )

    # 5. Build preprocessing and tune both models.
    preprocessor = build_preprocessor(X_train)

    lr_search = tune_logistic_regression(
        X_train,
        y_train,
        groups_train,
        preprocessor,
        cv,
    )

    xgb_search = tune_xgboost(
        X_train,
        y_train,
        groups_train,
        preprocessor,
        cv,
    )

    best_lr = lr_search.best_estimator_
    best_xgb = xgb_search.best_estimator_

    # 6. Compare the selected configurations using grouped CV.
    print("\nCalculating grouped CV comparison...")

    lr_cv_summary = cross_validate_selected_configuration(
        best_lr,
        X_train,
        y_train,
        groups_train,
        cv,
        "Logistic Regression",
    )

    xgb_cv_summary = cross_validate_selected_configuration(
        best_xgb,
        X_train,
        y_train,
        groups_train,
        cv,
        "XGBoost",
    )

    cv_comparison = pd.DataFrame([
        lr_cv_summary,
        xgb_cv_summary,
    ])

    print("\nGrouped CV comparison:")
    print(cv_comparison.round(4).to_string(index=False))

    # 7. Select the best of the four experiment families/configurations.
    # Each search compares oversampling vs no oversampling internally;
    # the final family choice uses grouped-CV ROC-AUC.
    if (
        xgb_search.best_score_
        > lr_search.best_score_
    ):
        final_model = best_xgb
        final_model_name = "XGBoost"
    else:
        final_model = best_lr
        final_model_name = "Logistic Regression"

    print("\nSelected final model:", final_model_name)

    # 8. Select classification threshold from OOF training predictions.
    threshold_results = select_threshold(
        final_model,
        X_train,
        y_train,
        groups_train,
        cv,
    )

    # 9. Evaluate once on the untouched test set.
    test_metrics, classification_metrics = evaluate_final_model(
        final_model,
        final_model_name,
        X_train,
        X_test,
        y_train,
        y_test,
        threshold_results["threshold"],
    )

    # 10. Calculate permutation importance on original input columns.
    feature_importance = calculate_permutation_importance(
        final_model,
        X_test,
        y_test,
    )

    # 11. Save model and exact feature order.
    model_features = X_train.columns.tolist()

    assert list(X_test.columns) == model_features

    save_model_artifacts(
        model=final_model,
        feature_names=model_features,
        importance=feature_importance,
    )

    # 12. Save training comparison and metadata.
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    cv_comparison.to_csv(METRICS_PATH, index=False)

    metadata = {
        "selected_model": final_model_name,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "cv_splits": N_SPLITS,
        "xgb_random_search_iterations": XGB_SEARCH_ITERATIONS,
        "oversampling": "RandomOverSampler(sampling_strategy=0.2) inside each CV training fold; compared against passthrough",
        "train_accounts": int(len(X_train)),
        "test_accounts": int(len(X_test)),
        "train_customers": int(groups_train.nunique()),
        "test_customers": int(groups_test.nunique()),
        "customer_overlap": int(
            len(set(groups_train) & set(groups_test))
        ),
        "feature_count": int(len(model_features)),
        "model_features": model_features,
        "threshold_selection": threshold_results,
        "test_probability_metrics": test_metrics,
        "test_classification_metrics": classification_metrics,
        "logistic_regression_cv": lr_cv_summary,
        "xgboost_cv": xgb_cv_summary,
        "limitations": [
            "The dataset does not define a prediction snapshot or horizon.",
            "Full-history repayment and bureau aggregates may contain information "
            "not available at loan origination.",
            "Collections features are excluded from model inputs due to "
            "unresolved event timing.",
            "The model estimates patterns in the supplied synthetic dataset "
            "and is not a validated production default model.",
        ],
    }

    with open(METADATA_PATH, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print("\nTraining completed.")
    print("Selected model:", final_model_name)
    print("Model saved to:", MODEL_PATH)
    print("Feature list saved to:", MODEL_FEATURES_PATH)
    print("Feature importance saved to:", FEATURE_IMPORTANCE_PATH)
    print("CV comparison saved to:", METRICS_PATH)
    print("Training metadata saved to:", METADATA_PATH)

    return {
        "model": final_model,
        "model_name": final_model_name,
        "cv_comparison": cv_comparison,
        "test_metrics": test_metrics,
        "classification_metrics": classification_metrics,
        "feature_importance": feature_importance,
        "metadata": metadata,
    }


if __name__ == "__main__":
    train_model()
