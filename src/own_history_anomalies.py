"""Own-history repayment anomaly detection.

Flags accounts whose repayment rate declined materially between the first
and last periods in the supplied repayment sequence.

Important: payment_month is treated as a supplied sequence. The feature
does not establish verified calendar chronology or predict future default.
"""

from __future__ import annotations

import pandas as pd


REQUIRED_COLUMNS = {
    "account_id",
    "first_period_repayment_rate_pct",
    "last_period_repayment_rate_pct",
    "repayment_rate_drop_pp",
}

def detect_own_history_anomalies(
    repayment_features: pd.DataFrame,
    threshold: float = 30.0,
) -> pd.DataFrame:
    """Return one row per account with own-history anomaly indicators.

    Parameters
    ----------
    repayment_features:
        Output of build_repayment_anomaly_features().

    threshold:
        Minimum decline in percentage points required to flag an account.
        A positive repayment_rate_drop_pp represents deterioration.
    """
    missing = REQUIRED_COLUMNS - set(repayment_features.columns)
    if missing:
        raise ValueError(
            f"repayment_features is missing required columns: {sorted(missing)}"
        )

    if repayment_features["account_id"].duplicated().any():
        raise ValueError("repayment_features must contain unique account_id values.")

    if threshold < 0:
        raise ValueError("threshold must be non-negative.")

    result = repayment_features[
        [
            "account_id",
            "first_period_repayment_rate_pct",
            "last_period_repayment_rate_pct",
            "repayment_rate_drop_pp",
        ]
    ].copy()

    result["is_own_history_anomaly"] = (
        pd.to_numeric(result["repayment_rate_drop_pp"], errors="coerce")
        >= threshold
    )

    result["own_history_anomaly_reason"] = ""

    flagged = result["is_own_history_anomaly"]
    result.loc[flagged, "own_history_anomaly_reason"] = (
        "Repayment rate declined by "
        + result.loc[flagged, "repayment_rate_drop_pp"]
            .round(2).astype(str)
        + " percentage points (from "
        + result.loc[flagged, "first_period_repayment_rate_pct"]
            .round(2).astype(str)
        + "% to "
        + result.loc[flagged, "last_period_repayment_rate_pct"]
            .round(2).astype(str)
        + "%)"
    )

    result["own_history_status"] = result["is_own_history_anomaly"].map(
        {True: "Flagged", False: "Normal"}
    )

    return result
