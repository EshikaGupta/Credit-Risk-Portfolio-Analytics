"""JSON-based persistence for anomaly analyst decisions.

Detection outputs remain separate from analyst decisions. This module stores
only decisions/comments and builds a review queue by merging them with the
latest combined anomaly results.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd


VALID_DECISIONS = {"Confirmed", "Dismissed"}


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def initialize_decisions_file(json_path: str) -> None:
    """Create an empty decisions JSON file if it does not exist."""
    path = Path(json_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not path.exists():
        _write_json(path, {})

def _write_json(path: Path, data: dict) -> None:
    """Write JSON atomically to reduce the chance of a partially written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")

    with temp_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)

    os.replace(temp_path, path)


def load_analyst_decisions(json_path: str) -> dict:
    """Load saved analyst decisions. Return an empty dict if file is absent."""
    path = Path(json_path)

    if not path.exists():
        initialize_decisions_file(json_path)
        return {}

    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON in analyst decisions file: {path}"
        ) from exc

    if not isinstance(data, dict):
        raise ValueError("Analyst decisions JSON must contain an object.")

    return data


def save_analyst_decision(
    json_path: str,
    account_id: str,
    decision: str,
    comment: Optional[str] = None,
) -> dict:
    """Save or update one account's analyst decision."""
    normalized_decision = decision.strip().title()

    if normalized_decision not in VALID_DECISIONS:
        raise ValueError(
            "decision must be either 'Confirmed' or 'Dismissed'."
        )

    account_id = str(account_id).strip()
    if not account_id:
        raise ValueError("account_id cannot be empty.")

    decisions = load_analyst_decisions(json_path)
    previous = decisions.get(account_id, {})

    now = _timestamp()

    decisions[account_id] = {
        "decision": normalized_decision,
        "analyst_comment": comment.strip() if comment else "",
        "reviewed_at": now,
        "created_at": previous.get("created_at", now),
    }

    _write_json(Path(json_path), decisions)
    return decisions[account_id]

def build_review_queue(
    combined_anomalies: pd.DataFrame,
    json_path: str,
) -> pd.DataFrame:
    """Merge current anomaly results with saved analyst decisions.

    Accounts not yet reviewed receive review_status='Pending'. Accounts that
    are no longer flagged are not included in the active review queue.
    """
    required = {
        "account_id",
        "is_anomaly",
        "anomaly_classification",
        "review_priority",
        "consolidated_anomaly_reason",
    }
    missing = required - set(combined_anomalies.columns)

    if missing:
        raise ValueError(
            f"combined_anomalies missing columns: {sorted(missing)}"
        )

    if combined_anomalies["account_id"].duplicated().any():
        raise ValueError(
            "combined_anomalies must have unique account_id values."
        )

    decisions = load_analyst_decisions(json_path)

    queue = combined_anomalies.loc[
        combined_anomalies["is_anomaly"].fillna(False)
    ].copy()

    queue["account_id"] = queue["account_id"].astype(str)

    queue["analyst_decision"] = queue["account_id"].map(
        lambda account_id: decisions.get(account_id, {}).get("decision", "")
    )
    queue["analyst_comment"] = queue["account_id"].map(
        lambda account_id: decisions.get(account_id, {}).get(
            "analyst_comment", ""
        )
    )
    queue["reviewed_at"] = queue["account_id"].map(
        lambda account_id: decisions.get(account_id, {}).get(
            "reviewed_at", ""
        )
    )

    queue["review_status"] = queue["analyst_decision"].replace(
        {"Confirmed": "Confirmed", "Dismissed": "Dismissed"}
    )
    queue.loc[queue["analyst_decision"].eq(""), "review_status"] = "Pending"

    return queue


def get_review_summary(review_queue: pd.DataFrame) -> dict:
    """Return counts for total, pending, confirmed and dismissed reviews."""
    if "review_status" not in review_queue.columns:
        raise ValueError("review_queue must contain review_status.")

    counts = review_queue["review_status"].value_counts()

    return {
        "total_flagged": int(len(review_queue)),
        "pending": int(counts.get("Pending", 0)),
        "confirmed": int(counts.get("Confirmed", 0)),
        "dismissed": int(counts.get("Dismissed", 0)),
    }
