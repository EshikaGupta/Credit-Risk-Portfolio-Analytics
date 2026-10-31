"""Run repayment anomaly detection and save the combined results."""

from __future__ import annotations

import pandas as pd

from src.config import PROCESSED_DATA_DIR
from src.ingestion import load_raw_tables, build_account_data
from src.anomaly_features import build_repayment_anomaly_features
from src.anomaly_detection import detect_peer_anomalies
from src.own_history_anomalies import detect_own_history_anomalies
from src.config import PROCESSED_DATA_DIR
from src.anomaly_reviews import initialize_decisions_file, build_review_queue

def combine_anomaly_results(
    peer_anomalies: pd.DataFrame,
    own_history_anomalies: pd.DataFrame,
) -> pd.DataFrame:
    """Combine peer and own-history anomaly flags into one account-level result."""
    result = peer_anomalies.merge(
        own_history_anomalies,
        on="account_id",
        how="outer",
        validate="one_to_one",
    )

    for column in ["is_peer_anomaly", "is_own_history_anomaly"]:
        result[column] = result[column].fillna(False).astype(bool)

    result["is_anomaly"] = (
        result["is_peer_anomaly"] | result["is_own_history_anomaly"]
    )

    result["anomaly_status"] = result["is_anomaly"].map(
        {True: "Flagged", False: "Normal"}
    )

    result["anomaly_reason"] = (
        result["anomaly_reason"].fillna("").astype(str)
        + result["own_history_anomaly_reason"].fillna("").astype(str).radd(
            "; ", fill_value=""
        )
    ).str.strip("; ")

        # Fields required by anomaly_reviews.build_review_queue()
    result["anomaly_classification"] = result.apply(
        lambda row: (
            "Peer and Own-History"
            if row["is_peer_anomaly"] and row["is_own_history_anomaly"]
            else "Peer"
            if row["is_peer_anomaly"]
            else "Own-History"
            if row["is_own_history_anomaly"]
            else "Normal"
        ),
        axis=1,
    )

    result["consolidated_anomaly_reason"] = (
        result["anomaly_reason"].fillna("").astype(str)
        + "; "
        + result["own_history_anomaly_reason"].fillna("").astype(str)
    ).str.strip("; ")

    result["review_priority"] = result.apply(
        lambda row: (
            "High"
            if row["is_peer_anomaly"] and row["is_own_history_anomaly"]
            else "Medium"
            if row["is_peer_anomaly"] or row["is_own_history_anomaly"]
            else "Low"
        ),
        axis=1,
    )
    
    return result

def run_anomaly_pipeline():
    output_dir = PROCESSED_DATA_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data using the same ingestion functions as the project.
    tables = load_raw_tables()
    account_data = build_account_data(tables)

    # Build repayment features and run both detectors.
    repayment_features = build_repayment_anomaly_features(
        tables["repayment_history"]
    )

    peer_anomalies = detect_peer_anomalies(
        repayment_features=repayment_features,
        account_data=account_data,
        min_peer_accounts=30,
        threshold=3.5,
    )

    own_history_anomalies = detect_own_history_anomalies(
        repayment_features=repayment_features
    )

    # Save individual detector results.
    peer_anomalies.to_csv(
        output_dir / "peer_anomalies.csv", index=False
    )
    own_history_anomalies.to_csv(
        output_dir / "own_history_anomalies.csv", index=False
    )

    # Combine detector results and save the full result.
    combined_anomalies = combine_anomaly_results(
        peer_anomalies,
        own_history_anomalies,
    )
    combined_anomalies.to_csv(
        output_dir / "combined_anomalies.csv", index=False
    )

    # Save only flagged accounts for the analyst review list.
    review_list = combined_anomalies.loc[
        combined_anomalies["is_anomaly"].fillna(False)
    ].copy()
    review_list.to_csv(
        output_dir / "anomaly_review_list.csv", index=False
    )

    # Preserve existing decisions; create the file only if it is missing.
    decisions_path = output_dir / "analyst_decisions.json"
    initialize_decisions_file(str(decisions_path))

    # Build and save the current review queue with saved decisions.
    review_queue = build_review_queue(
        combined_anomalies=combined_anomalies,
        json_path=str(decisions_path),
    )
    review_queue.to_csv(
        output_dir / "anomaly_review_list.csv", index=False
    )

    print(f"Saved anomaly outputs to: {output_dir}")
    return combined_anomalies


if __name__ == "__main__":
    run_anomaly_pipeline()

