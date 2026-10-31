"""Feature engineering for repayment-based anomaly detection.

These features are intended for anomaly detection and analyst review.
Check temporal availability before reusing them in a default prediction model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "account_id", "payment_month", "emi_due", "emi_paid",
    "days_past_due", "payment_status", "outstanding_balance",
}


def build_repayment_anomaly_features(
    repayment_history: pd.DataFrame,
) -> pd.DataFrame:
    """Return one row per account with repayment behaviour indicators.

    Uses payment_month as the supplied sequence, not reconstructed dates.

    Missed = status contains 'miss' OR due > 0 and paid <= 0.
    Late = not missed AND (status contains 'late' OR days_past_due > 0).
    Repayment ratios are capped at 100% so overpayments do not inflate them.
    """
    missing = REQUIRED_COLUMNS - set(repayment_history.columns)
    if missing:
        raise ValueError(
            f"Repayment history is missing required columns: {sorted(missing)}"
        )

    df = repayment_history.copy()
    df["payment_month"] = pd.to_numeric(df["payment_month"], errors="coerce")

    for col in ["emi_due", "emi_paid", "days_past_due", "outstanding_balance"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["payment_status_norm"] = (
        df["payment_status"].astype("string").str.strip().str.lower()
    )
    df = df.dropna(subset=["account_id", "payment_month"])
    df = df.sort_values(["account_id", "payment_month"])

    due = df["emi_due"].clip(lower=0)
    paid = df["emi_paid"].clip(lower=0)

    df["repayment_ratio"] = np.where(
        due > 0, (paid / due).clip(0, 1), np.nan
    )

    status = df["payment_status_norm"].fillna("")
    df["is_missed"] = (
        status.str.contains("miss", regex=False)
        | ((due > 0) & (paid <= 0))
    )
    df["is_late"] = (
        ~df["is_missed"]
        & (
            status.str.contains("late", regex=False)
            | (df["days_past_due"].fillna(0) > 0)
        )
    )
    df["is_on_time"] = (
        ~df["is_missed"]
        & ~df["is_late"]
        & status.str.contains(r"on[- ]?time", regex=True)
    )

    rows = []
    for account_id, group in df.groupby("account_id", sort=False):
        group = group.sort_values("payment_month")
        n = len(group)

        first_period = group[group["payment_month"] <= 6]
        last_period = group[group["payment_month"] > 6]

        # Fallback for nonstandard payment_month numbering.
        if first_period.empty or last_period.empty:
            first_period = group.head(6)
            last_period = group.tail(6)

        valid_due = group["emi_due"].notna() & (group["emi_due"] > 0)
        total_due = group.loc[valid_due, "emi_due"].sum()
        total_paid = group.loc[valid_due, "emi_paid"].clip(lower=0).sum()

        overall_rate = (
            min(total_paid / total_due, 1.0) if total_due > 0 else np.nan
        )
        first_rate = first_period["repayment_ratio"].mean()
        last_rate = last_period["repayment_ratio"].mean()
        rate_drop = (
            first_rate - last_rate
            if pd.notna(first_rate) and pd.notna(last_rate)
            else np.nan
        )

        balances = group["outstanding_balance"].dropna()

        rows.append({
            "account_id": account_id,
            "payment_records": n,
            "missed_payment_count": int(group["is_missed"].sum()),
            "late_payment_count": int(group["is_late"].sum()),
            "on_time_payment_count": int(group["is_on_time"].sum()),
            "missed_payment_rate_pct": group["is_missed"].mean() * 100,
            "late_payment_rate_pct": group["is_late"].mean() * 100,
            "average_days_past_due": group["days_past_due"].mean(),
            "maximum_days_past_due": group["days_past_due"].max(),
            "repayment_rate_pct": (
                overall_rate * 100 if pd.notna(overall_rate) else np.nan
            ),
            "average_repayment_ratio_pct": (
                group["repayment_ratio"].mean() * 100
            ),
            "first_period_repayment_rate_pct": (
                first_rate * 100 if pd.notna(first_rate) else np.nan
            ),
            "last_period_repayment_rate_pct": (
                last_rate * 100 if pd.notna(last_rate) else np.nan
            ),
            "repayment_rate_drop_pp": (
                rate_drop * 100 if pd.notna(rate_drop) else np.nan
            ),
            "first_outstanding_balance": (
                balances.iloc[0] if len(balances) else np.nan
            ),
            "last_outstanding_balance": (
                balances.iloc[-1] if len(balances) else np.nan
            ),
            "outstanding_balance_change": (
                balances.iloc[-1] - balances.iloc[0]
                if len(balances) >= 2 else np.nan
            ),
            "history_sequence_note": (
                "payment_month sequence; reconstructed dates not used"
            ),
        })

    return pd.DataFrame(rows)
