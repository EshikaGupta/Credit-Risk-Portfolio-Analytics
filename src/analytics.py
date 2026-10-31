"""Reusable portfolio analytics for notebooks and Streamlit."""
from __future__ import annotations

import numpy as np
import pandas as pd


def validate_account_data(df: pd.DataFrame) -> None:
    required = {"account_id", "customer_id", "target_default", "loan_amount"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if df["account_id"].duplicated().any():
        raise ValueError("Expected one row per account.")
    valid_targets = df["target_default"].dropna()
    if not valid_targets.isin([0, 1]).all():
        raise ValueError("target_default must contain only 0, 1 or missing.")


def portfolio_kpis(df: pd.DataFrame) -> dict:
    """Calculate headline portfolio metrics using explicit denominators."""
    validate_account_data(df)
    valid = df.loc[df["target_default"].isin([0, 1])].copy()
    valid_exposure = valid.loc[
        valid["loan_amount"].notna()
        & np.isfinite(valid["loan_amount"])
        & valid["loan_amount"].ge(0)
    ]
    return {
        "accounts": int(valid["account_id"].nunique()),
        "customers": int(valid["customer_id"].nunique()),
        "defaulted_accounts": int(valid["target_default"].sum()),
        "default_rate_pct": (
            float(valid["target_default"].mean() * 100)
            if len(valid) else np.nan
        ),
        "original_exposure": float(valid_exposure["loan_amount"].sum()),
        "accounts_with_valid_exposure": int(
            valid_exposure["account_id"].nunique()
        ),
    }


def build_segment_summary(
    df: pd.DataFrame,
    group_cols: str | list[str],
) -> pd.DataFrame:
    """Default and exposure metrics by segment, with separate denominators."""
    if isinstance(group_cols, str):
        group_cols = [group_cols]

    validate_account_data(df)
    valid_targets = df.loc[df["target_default"].isin([0, 1])].copy()
    if valid_targets.empty:
        raise ValueError("No records with valid binary targets.")

    valid_exposure = valid_targets.loc[
        valid_targets["loan_amount"].notna()
        & np.isfinite(valid_targets["loan_amount"])
        & valid_targets["loan_amount"].ge(0)
    ].copy()

    default_summary = (
        valid_targets.groupby(
            group_cols, dropna=False, observed=True
        )
        .agg(
            accounts=("account_id", "nunique"),
            customers=("customer_id", "nunique"),
            defaulted_accounts=("target_default", "sum"),
            default_rate_pct=("target_default", lambda s: s.mean() * 100),
        )
        .reset_index()
    )

    exposure_summary = (
        valid_exposure.groupby(
            group_cols, dropna=False, observed=True
        )
        .agg(exposure=("loan_amount", "sum"))
        .reset_index()
    )

    total_exposure = valid_exposure["loan_amount"].sum()
    if total_exposure > 0:
        exposure_summary["exposure_share_pct"] = (
            exposure_summary["exposure"] / total_exposure * 100
        )
    else:
        exposure_summary["exposure_share_pct"] = np.nan

    result = default_summary.merge(
        exposure_summary,
        on=group_cols,
        how="left",
        validate="one_to_one",
    )
    overall_rate = valid_targets["target_default"].mean() * 100
    result["default_rate_diff_pp"] = (
        result["default_rate_pct"] - overall_rate
    )
    return result.sort_values(
        "exposure_share_pct", ascending=False, na_position="last"
    ).reset_index(drop=True)


def income_employment_crosstab(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-dimensional default analysis with sample-size flags."""
    required = {"income_band", "employment_type"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing columns: {sorted(required - set(df.columns))}")

    valid = df.loc[df["target_default"].isin([0, 1])].copy()
    result = (
        valid.groupby(
            ["income_band", "employment_type"],
            observed=True,
            dropna=False,
        )
        .agg(
            accounts=("account_id", "nunique"),
            customers=("customer_id", "nunique"),
            defaulted_accounts=("target_default", "sum"),
            default_rate_pct=("target_default", lambda s: s.mean() * 100),
        )
        .reset_index()
    )
    result["sample_flag"] = np.where(
        result["accounts"] < 30,
        "Small sample",
        "Sufficient for descriptive reporting",
    )
    return result



def filter_portfolio(
    df: pd.DataFrame,
    loan_types: list[str] | None = None,
    regions: list[str] | None = None,
    employment_types: list[str] | None = None,
    default_status: str = "All",
) -> pd.DataFrame:
    """Apply dashboard filters and return the filtered DataFrame."""
    result = df.loc[df["target_default"].isin([0, 1])].copy()

    if loan_types:
        result = result[
            result["loan_type"].astype(str).isin(loan_types)
        ]

    if regions:
        result = result[
            result["region"].astype(str).isin(regions)
        ]

    if employment_types:
        result = result[
            result["employment_type"].astype(str).isin(employment_types)
        ]

    if default_status == "Defaulted":
        result = result[result["target_default"] == 1]
    elif default_status == "Performing":
        result = result[result["target_default"] == 0]
    elif default_status != "All":
        raise ValueError(
            "default_status must be All, Defaulted or Performing"
        )

    return result
