
"""Peer-group anomaly detection for repayment behaviour."""

import numpy as np
import pandas as pd


HIGH_SIDE = {
    "missed_payment_rate_pct": "High missed-payment rate",
    "late_payment_rate_pct": "High late-payment rate",
    "average_days_past_due": "High average days past due",
    "maximum_days_past_due": "High maximum days past due",
}


def _modified_z(values):
    """Calculate modified Z-scores using MAD, with IQR fallback."""

    x = pd.to_numeric(values, errors="coerce").astype(float)

    med = x.median()
    mad = (x - med).abs().median()

    if pd.isna(med):
        return pd.Series(np.nan, index=x.index)

    if pd.notna(mad) and mad > 0:
        return 0.6745 * (x - med) / mad

    iqr = x.quantile(0.75) - x.quantile(0.25)

    if pd.notna(iqr) and iqr > 0:
        return (x - med) / (iqr / 1.349)

    return pd.Series(0.0, index=x.index).where(x.notna())


def detect_peer_anomalies(
    repayment_features: pd.DataFrame,
    account_data: pd.DataFrame,
    min_peer_accounts: int = 30,
    threshold: float = 3.5,
) -> pd.DataFrame:
    """
    Compare each account against similar borrowers.

    Primary peer group:
        loan_type + income_band

    Fallback:
        loan_type

    Accounts without sufficient peers are not scored.
    """

    if min_peer_accounts < 2 or threshold <= 0:
        raise ValueError(
            "Use min_peer_accounts >= 2 and threshold > 0."
        )

    required_features = {
        "account_id",
        "missed_payment_rate_pct",
        "late_payment_rate_pct",
        "average_days_past_due",
        "maximum_days_past_due",
    }

    missing_features = required_features - set(repayment_features.columns)

    if missing_features:
        raise ValueError(
            f"repayment_features missing columns: {sorted(missing_features)}"
        )

    if repayment_features["account_id"].duplicated().any():
        raise ValueError(
            "repayment_features must have one row per account."
        )

    if account_data["account_id"].duplicated().any():
        raise ValueError(
            "account_data must have one row per account."
        )

    required_account_columns = {
        "account_id",
        "loan_type",
        "income_band",
    }

    missing = required_account_columns - set(account_data.columns)

    if missing:
        raise ValueError(
            f"account_data missing columns: {sorted(missing)}"
        )

    data = repayment_features.merge(
        account_data[
            ["account_id", "loan_type", "income_band"]
        ],
        on="account_id",
        how="left",
        validate="one_to_one",
    )

    # Determine peer-group sizes.
    combined_n = data.groupby(
        ["loan_type", "income_band"],
        dropna=False,
    )["account_id"].transform("size")

    loan_n = data.groupby(
        "loan_type",
        dropna=False,
    )["account_id"].transform("size")

    use_combined = combined_n >= min_peer_accounts

    use_loan = (
        (~use_combined)
        & (loan_n >= min_peer_accounts)
    )

    data["peer_group_level"] = np.select(
        [use_combined, use_loan],
        [
            "loan_type + income_band",
            "loan_type",
        ],
        default="insufficient peers",
    )

    data["peer_group"] = np.where(
        use_combined,
        (
            data["loan_type"].fillna("Unknown").astype(str)
            + " | "
            + data["income_band"].fillna("Unknown").astype(str)
        ),
        np.where(
            use_loan,
            data["loan_type"].fillna("Unknown").astype(str),
            "Not scored",
        ),
    )

    data["peer_account_count"] = np.where(
        use_combined,
        combined_n,
        np.where(use_loan, loan_n, 0),
    ).astype(int)

    eligible = (
        data["peer_group_level"] != "insufficient peers"
    )

    group_cols = [
        "peer_group_level",
        "peer_group",
    ]

    # Calculate modified Z-scores for the four selected metrics.
    for feature in HIGH_SIDE:

        score_col = f"{feature}_modified_z"

        data[score_col] = np.nan

        for _, idx in data.loc[eligible].groupby(
            group_cols,
            dropna=False,
        ).groups.items():

            data.loc[idx, score_col] = _modified_z(
                data.loc[idx, feature]
            )

    # Generate anomaly explanations.
    reasons = []

    for _, row in data.iterrows():

        why = []

        for feature, label in HIGH_SIDE.items():

            score = row.get(
                f"{feature}_modified_z"
            )

            value = row.get(feature)

            if (
                pd.notna(score)
                and pd.notna(value)
                and score >= threshold
            ):
                why.append(
                    f"{label}: {value:.2f} "
                    f"(score {score:.2f})"
                )

        reasons.append("; ".join(why))

    data["anomaly_reason"] = reasons

    data["is_peer_anomaly"] = (
        data["anomaly_reason"].ne("")
    )

    data["anomaly_status"] = np.where(
        ~eligible,
        "Not scored - insufficient peers",
        np.where(
            data["is_peer_anomaly"],
            "Flagged",
            "Normal",
        ),
    )

    return data
