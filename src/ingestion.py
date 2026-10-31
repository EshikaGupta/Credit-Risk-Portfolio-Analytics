"""Load, clean, aggregate, and join the five synthetic portfolio tables.

The transformations here follow the notebook's current aggregation logic.
Important: repayment and bureau aggregates are full-history/current fields.
Their availability at a defined prediction timestamp has not been established.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import (
    CUSTOMERS_PATH,
    LOAN_ACCOUNTS_PATH,
    BUREAU_CREDIT_PATH,
    REPAYMENT_HISTORY_PATH,
    COLLECTIONS_EVENTS_PATH,
    ACCOUNT_DATA_PATH,
)


def load_raw_tables(
    customers_path=CUSTOMERS_PATH,
    loan_accounts_path=LOAN_ACCOUNTS_PATH,
    bureau_credit_path=BUREAU_CREDIT_PATH,
    repayment_history_path=REPAYMENT_HISTORY_PATH,
    collections_events_path=COLLECTIONS_EVENTS_PATH,
) -> dict[str, pd.DataFrame]:
    """Load all five CSVs, allowing paths to be overridden for tests."""
    tables = {
        "customers": pd.read_csv(customers_path),
        "loan_accounts": pd.read_csv(loan_accounts_path),
        "bureau_credit": pd.read_csv(bureau_credit_path),
        "repayment_history": pd.read_csv(repayment_history_path),
        "collections_events": pd.read_csv(collections_events_path),
    }
    return tables


def _require_columns(df: pd.DataFrame, columns: list[str], table: str) -> None:
    missing = sorted(set(columns) - set(df.columns))
    if missing:
        raise ValueError(f"{table} is missing required columns: {missing}")


def clean_customer_data(customers: pd.DataFrame) -> pd.DataFrame:
    """Normalize customer fields and create missingness indicators."""
    customers = customers.copy()
    _require_columns(
        customers,
        [
            "customer_id", "gender", "employment_type", "years_employed",
            "num_dependents", "years_at_residence",
        ],
        "customers",
    )

    # Capture missingness flags before filling the known unemployed rule.
    customers["years_employed_missing"] = (
        customers["years_employed"].isna().astype("int8")
    )
    customers["num_dependents_missing"] = (
        customers["num_dependents"].isna().astype("int8")
    )
    customers["residence_missing"] = (
        customers["years_at_residence"].isna().astype("int8")
    )

    # In this dataset, unemployed customers with unknown tenure are set to 0.
    unemployed = customers["employment_type"].eq("Unemployed")
    customers.loc[unemployed, "years_employed"] = (
        customers.loc[unemployed, "years_employed"].fillna(0)
    )

    gender_map = {"M": "Male", "F": "Female"}
    customers["gender"] = customers["gender"].replace(gender_map)
    return customers


def aggregate_bureau(bureau_credit: pd.DataFrame) -> pd.DataFrame:
    """Aggregate external credit lines to one row per customer."""
    bureau = bureau_credit.copy()
    _require_columns(
        bureau,
        [
            "bureau_id", "customer_id", "sanctioned_amount",
            "outstanding_amount", "days_past_due", "account_status",
        ],
        "bureau_credit",
    )
    result = (
        bureau.groupby("customer_id")
        .agg(
            bureau_record_count=("bureau_id", "size"),
            bureau_total=("sanctioned_amount", lambda s: s.sum(min_count=1)),
            bureau_total_outstanding=(
                "outstanding_amount", lambda s: s.sum(min_count=1)
            ),
            bureau_avg_dpd=("days_past_due", "mean"),
            bureau_max_dpd=("days_past_due", "max"),
            bureau_dpd_missing_count=("days_past_due", lambda s: s.isna().sum()),
            bureau_writeoff_count=(
                "account_status", lambda s: s.eq("Written-Off").sum()
            ),
            bureau_active_count=(
                "account_status", lambda s: s.eq("Active").sum()
            ),
        )
        .reset_index()
    )
    if result["customer_id"].duplicated().any():
        raise ValueError("Bureau aggregation did not produce unique customers.")
    return result

def aggregate_repayments(repayment_history: pd.DataFrame) -> pd.DataFrame:
    """Aggregate monthly repayment history to one row per account."""
    repayment = repayment_history.copy()
    _require_columns(
        repayment,
        [
            "account_id", "payment_status", "days_past_due",
            "emi_due", "emi_paid",
        ],
        "repayment_history",
    )
    result = (
        repayment.groupby("account_id")
        .agg(
            missed_payment_count=(
                "payment_status", lambda s: s.eq("Missed").sum()
            ),
            late_payment_count=(
                "payment_status", lambda s: s.eq("Late").sum()
            ),
            repayment_avg_dpd=("days_past_due", "mean"),
            repayment_max_dpd=("days_past_due", "max"),
            total_emi_due=("emi_due", lambda s: s.sum(min_count=1)),
            total_emi_paid=("emi_paid", lambda s: s.sum(min_count=1)),
        )
        .reset_index()
    )
    if result["account_id"].duplicated().any():
        raise ValueError("Repayment aggregation did not produce unique accounts.")
    return result

def aggregate_collections(collections_events: pd.DataFrame) -> pd.DataFrame:
    """Aggregate collections events to one row per account."""
    collections = collections_events.copy()
    _require_columns(
        collections,
        [
            "collection_event_id", "account_id", "event_date",
            "action_type", "outcome", "amount_recovered",
        ],
        "collections_events",
    )
    collections["event_date"] = pd.to_datetime(
        collections["event_date"], errors="coerce"
    )
    result = (
        collections.groupby("account_id")
        .agg(
            collection_event_count=("collection_event_id", "size"),
            payment_received_count=(
                "outcome", lambda s: s.eq("Payment Received").sum()
            ),
            event_max_date=("event_date", "max"),
            legal_notice_count=(
                "action_type", lambda s: s.eq("Legal Notice").sum()
            ),
            total_amount_recovered=(
                "amount_recovered", lambda s: s.sum(min_count=1)
            ),
        )
        .reset_index()
    )
    if result["account_id"].duplicated().any():
        raise ValueError("Collections aggregation did not produce unique accounts.")
    return result

def build_account_data(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build one row per loan account from already-loaded raw tables."""
    required = {
        "customers", "loan_accounts", "bureau_credit",
        "repayment_history", "collections_events",
    }
    missing_tables = required - set(tables)
    if missing_tables:
        raise ValueError(f"Missing input tables: {sorted(missing_tables)}")

    customers = clean_customer_data(tables["customers"])
    loans = tables["loan_accounts"].copy()
    bureau = tables["bureau_credit"].copy()
    repayments = tables["repayment_history"].copy()
    collections = tables["collections_events"].copy()

    _require_columns(
        loans,
        [
            "account_id", "customer_id", "target_default", "loan_amount",
            "origination_date", "interest_rate_pct",
        ],
        "loan_accounts",
    )

    if loans["account_id"].duplicated().any():
        raise ValueError("loan_accounts has duplicate account_id values.")

    loans["origination_date"] = pd.to_datetime(
        loans["origination_date"], errors="coerce"
    )
    loans["interest_rate_missing"] = (
        loans["interest_rate_pct"].isna().astype("int8")
    )

    customer_features = aggregate_bureau(bureau)
    repayment_features = aggregate_repayments(repayments)
    collection_features = aggregate_collections(collections)

    original_ids = set(loans["account_id"])
    original_rows = len(loans)

    data = loans.merge(
        customers, on="customer_id", how="left",
        validate="many_to_one", indicator="_customer_join"
    )
    if not data["_customer_join"].eq("both").all():
        raise ValueError("Some loan accounts did not match a customer.")
    data = data.drop(columns="_customer_join")

    data = data.merge(
        customer_features, on="customer_id", how="left",
        validate="many_to_one"
    )
    data = data.merge(
        repayment_features, on="account_id", how="left",
        validate="one_to_one"
    )
    data = data.merge(
        collection_features, on="account_id", how="left",
        validate="one_to_one"
    )

    # No bureau record means no recorded credit line totals/counts.
    no_bureau = data["bureau_record_count"].isna()
    for col in [
        "bureau_total", "bureau_total_outstanding",
        "bureau_record_count", "bureau_writeoff_count",
        "bureau_active_count", "bureau_dpd_missing_count",
    ]:
        data.loc[no_bureau, col] = data.loc[no_bureau, col].fillna(0)

    # No collections rows means zero recorded collection activity.
    no_collections = data["collection_event_count"].isna()
    for col in [
        "collection_event_count", "payment_received_count",
        "legal_notice_count", "total_amount_recovered",
    ]:
        data.loc[no_collections, col] = data.loc[no_collections, col].fillna(0)

    # Engineered ratios, matching the notebook.
    def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
        return numerator / denominator.replace(0, np.nan)

    data["emi_payment_ratio"] = safe_divide(
        data["total_emi_paid"], data["total_emi_due"]
    )
    data["loan_to_income_ratio"] = safe_divide(
        data["loan_amount"], data["annual_income"]
    )
    data["bureau_outstanding_ratio"] = safe_divide(
        data["bureau_total_outstanding"], data["annual_income"]
    )
    data["bureau_active_account_ratio"] = safe_divide(
        data["bureau_active_count"],
        data["bureau_record_count"],
    )

    data["bureau_outstanding_per_active_account"] = safe_divide(
        data["bureau_total_outstanding"],
        data["bureau_active_count"],
    )
    data["dpd_gap"] = data["repayment_avg_dpd"] - data["bureau_avg_dpd"]

    engineered = [
        "emi_payment_ratio", "loan_to_income_ratio",
        "bureau_outstanding_ratio", "dpd_gap",
    ]
    data[engineered] = data[engineered].replace([np.inf, -np.inf], np.nan)

    # Segments are used for dashboard profiling, not model input.
    data["income_band"] = pd.cut(
        data["annual_income"],
        bins=[-np.inf, 25_000, 50_000, 75_000, 100_000, np.inf],
        labels=[
            "Below $25k", "$25k–$50k", "$50k–$75k",
            "$75k–$100k", "$100k+",
        ],
        right=False,
    )
    data["age_band"] = pd.cut(
        data["age"],
        bins=[-np.inf, 25, 35, 45, 55, 65, np.inf],
        labels=["Under 25", "25–34", "35–44", "45–54", "55–64", "65+"],
        right=False,
    )

    if len(data) != original_rows or data["account_id"].duplicated().any():
        raise ValueError("Final account data has row loss or duplicate accounts.")
    if set(data["account_id"]) != original_ids:
        raise ValueError("Final account IDs do not match source loan accounts.")

    return data


def build_and_save_account_data(output_path=ACCOUNT_DATA_PATH) -> pd.DataFrame:
    """Convenience function: load raw CSVs, build dataset, save processed CSV."""
    tables = load_raw_tables()
    account_data = build_account_data(tables)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    account_data.to_csv(output_path, index=False)
    return account_data


if __name__ == "__main__":
    df = build_and_save_account_data()