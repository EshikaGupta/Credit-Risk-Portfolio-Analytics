"""Grounded natural-language Q&A for the credit-risk portfolio.

Gemini translates a question into a constrained query plan. All metrics and
figures are calculated locally from the filtered DataFrame; Gemini never
calculates or invents portfolio numbers.
"""
from __future__ import annotations

import json
import os
from typing import Any
from dotenv import load_dotenv
import pandas as pd
import streamlit as st

load_dotenv()
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

DIMENSIONS = {
    "loan_type": "Loan type",
    "region": "Region",
    "employment_type": "Employment type",
    "income_band": "Income band",
    "age_band": "Age band",
}
METRICS = {
    "default_rate_pct": "default rate",
    "accounts": "number of loan accounts",
    "customers": "number of customers",
    "defaulted_accounts": "number of defaulted accounts",
    "exposure": "original loan exposure",
    "exposure_share_pct": "share of original loan exposure",
    "avg_loan_amount": "average original loan amount",
}
INTENTS = ["overall", "rank", "segment_metric", "compare", "unsupported"]


def _get_api_key() -> str | None:
    """Read key from Streamlit Cloud secrets, then local environment."""
    try:
        key = os.getenv("GEMINI_API_KEY")
        if key:
            return str(key).strip()
    except Exception:
        # st.secrets may be unavailable when running locally without secrets.toml.
        pass
    key = os.getenv("GEMINI_API_KEY")
    return key.strip() if key else None

@st.cache_data(ttl=3600, show_spinner=False)
def _query_plan(question: str, api_key: str) -> dict[str, Any]:
    """Use Gemini only to map natural language to a small, validated query plan."""
    schema = {
        "type": "OBJECT",
        "properties": {
            "intent": {
                "type": "STRING",
                "enum": INTENTS,
                "description": "Whether this is an overall metric, a ranking, a metric for one segment, a comparison of segments, or unsupported."
            },
            "metric": {
                "type": "STRING",
                "enum": list(METRICS.keys()),
                "description": "Metric being asked about."
            },
            "dimension": {
                "type": "STRING",
                "enum": list(DIMENSIONS.keys()),
                "description": "Dimension used to group/filter the portfolio."
            },
            "direction": {
                "type": "STRING",
                "enum": ["max", "min"],
                "description": "For ranking questions, whether to return the highest or lowest value."
            },
            "segment_value": {
                "type": "STRING",
                "description": "One exact category value from the selected dimension for a segment_metric question; otherwise empty."
            },
            "segment_values": {
                "type": "ARRAY",
                "items": {"type": "STRING"},
                "description": "Two or more exact category values for compare questions; otherwise empty."
            }
        },
        "required": ["intent", "metric", "dimension", "direction", "segment_value", "segment_values"]
    }

    prompt = f"""
You are a query router for a credit-risk portfolio dashboard.
Convert the user's question into a query plan, not an answer.

Supported metrics:
{json.dumps(METRICS)}
Supported dimensions:
{json.dumps(DIMENSIONS)}
Rules:
- Use "overall" for a portfolio-wide metric without a dimension (e.g. overall default rate, total accounts, total original exposure).
- Use "rank" for highest/lowest/best/worst by a supported dimension. "Highest default rate by loan type" means metric=default_rate_pct, dimension=loan_type, direction=max.
- Use "segment_metric" when the question asks for a metric for one category, e.g. "default rate for mortgages" means dimension=loan_type, segment_value="Mortgage".
- Use "compare" only when the user explicitly compares two or more categories of one dimension. Put their exact category labels in segment_values.
- Use "unsupported" if the question requires data not represented by these metrics/dimensions, asks for causes/forecasts/recommendations, or is unrelated to this portfolio.
- Do not infer or invent category names. If the category or dimension is ambiguous, use unsupported.
- Set unused strings to "" and unused arrays to [].
- This is only a query plan. Do not calculate or include any numeric result.

User question: {question}
"""
    # Lazy import keeps the non-AI dashboard usable if the optional SDK is absent.
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key,http_options=types.HttpOptions(timeout=15000))
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=schema,
            temperature=0,
            max_output_tokens=300,
        ),
    )
    if not response.text:
        raise ValueError("Gemini returned an empty query plan.")
    plan = json.loads(response.text)

    # Validate independently of the LLM response.
    if plan.get("intent") not in INTENTS:
        raise ValueError("Invalid query intent.")
    if plan.get("metric") not in METRICS:
        raise ValueError("Unsupported metric.")
    if plan.get("dimension") not in DIMENSIONS:
        raise ValueError("Unsupported dimension.")
    if plan.get("direction") not in {"max", "min"}:
        raise ValueError("Invalid ranking direction.")
    if not isinstance(plan.get("segment_values"), list):
        raise ValueError("Invalid comparison values.")
    return plan

def _eligible(df: pd.DataFrame) -> pd.DataFrame:
    required = {"account_id", "customer_id", "target_default", "loan_amount"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dashboard data is missing required fields: {sorted(missing)}")
    return df.loc[df["target_default"].isin([0, 1])].copy()


def _summary(df: pd.DataFrame, dimension: str | None = None) -> pd.DataFrame:
    """Compute metrics deterministically; never use LLM output for numbers."""
    df = _eligible(df)
    if dimension:
        if dimension not in df.columns:
            raise ValueError(f"Dashboard data does not contain '{dimension}'.")
        grouped = df.groupby(dimension, dropna=False, observed=True)
    else:
        grouped = [(None, df)]

    rows = []
    for label, part in grouped:
        amounts = pd.to_numeric(part["loan_amount"], errors="coerce")
        valid_amounts = amounts[amounts.notna() & amounts.ge(0)]
        total_exposure = float(valid_amounts.sum()) if len(valid_amounts) else None
        rows.append({
            "segment": None if label is None else str(label),
            "accounts": int(part["account_id"].nunique()),
            "customers": int(part["customer_id"].nunique()),
            "defaulted_accounts": int(part["target_default"].sum()),
            "default_rate_pct": float(part["target_default"].mean() * 100) if len(part) else None,
            "exposure": total_exposure,
            "avg_loan_amount": float(valid_amounts.mean()) if len(valid_amounts) else None,
        })

    result = pd.DataFrame(rows)
    exposure_total = pd.to_numeric(df["loan_amount"], errors="coerce")
    exposure_total = exposure_total[exposure_total.notna() & exposure_total.ge(0)].sum()
    if exposure_total > 0:
        result["exposure_share_pct"] = result["exposure"].fillna(0) / float(exposure_total) * 100
    else:
        result["exposure_share_pct"] = None
    return result


def _fmt(metric: str, value: Any) -> str:
    if value is None or pd.isna(value):
        return "not available"
    if metric in {"default_rate_pct", "exposure_share_pct"}:
        return f"{float(value):.2f}%"
    if metric in {"exposure", "avg_loan_amount"}:
        return f"${float(value):,.2f}"
    return f"{int(value):,}"


import re


def _normalize_text(value: str) -> str:
    value = value.lower().replace("-", " ").replace("_", " ")
    return re.sub(r"\s+", " ", value).strip()


def _deterministic_query_plan(
    question: str,
    df: pd.DataFrame,
) -> dict[str, Any] | None:
    """Parse clearly supported questions using actual dataframe categories."""

    q = _normalize_text(question)

    # Infer the requested metric.
    metric_patterns = {
        "default_rate_pct": [
            r"\bdefault rates?\b",
            r"\bdefault percentages?\b",
            r"\bdefaulting rate\b",
            r"\bdelinquency rates?\b",
        ],
        "defaulted_accounts": [
            r"\bdefaulted accounts?\b",
            r"\baccounts? in default\b",
            r"\bnumber of defaults?\b",
        ],
        "exposure_share_pct": [
            r"\bexposure share\b",
            r"\bshare of (?:the )?(?:original )?(?:loan )?exposure\b",
            r"\bpercentage of (?:the )?exposure\b",
        ],
        "avg_loan_amount": [
            r"\baverage loan amounts?\b",
            r"\baverage original loan amounts?\b",
            r"\bmean loan amounts?\b",
        ],
        "exposure": [
            r"\bexposure\b",
            r"\btotal (?:original )?loan amounts?\b",
            r"\boriginal loan amounts?\b",
            r"\btotal lending\b",
        ],
        "customers": [
            r"\bcustomers?\b",
            r"\bborrowers?\b",
        ],
        "accounts": [
            r"\bloan accounts?\b",
            r"\bnumber of accounts?\b",
            r"\baccount count\b",
            r"\bhow many accounts?\b",
        ],
    }

    metric = None
    for candidate, patterns in metric_patterns.items():
        if any(re.search(pattern, q) for pattern in patterns):
            metric = candidate
            break

    if metric is None:
        return None

    # Find exact category labels from the current dashboard data.
    # Matching longer labels first avoids matching "Loan" before "Auto Loan".
    dimensions = {
        "loan_type": "loan_type",
        "region": "region",
        "employment_type": "employment_type",
        "income_band": "income_band",
        "age_band": "age_band",
    }

    matches_by_dimension = {}

    for dimension in dimensions:
        if dimension not in df.columns:
            continue

        categories = (
            df[dimension]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        matches = []
        for category in sorted(categories, key=len, reverse=True):
            label = _normalize_text(category)
            if not label:
                continue

            # Match category phrases, not substrings within other words.
            pattern = r"(?<!\w)" + re.escape(label) + r"(?!\w)"
            if re.search(pattern, q):
                matches.append(category)

        if matches:
            matches_by_dimension[dimension] = matches

    # Prefer a dimension with two or more explicitly mentioned categories
    # for comparisons.
    comparison_words = re.search(
        r"\b(compare|comparison|versus|vs|against)\b", q
    )

    comparison_dimensions = [
        dim for dim, values in matches_by_dimension.items()
        if len(values) >= 2
    ]

    if comparison_words and comparison_dimensions:
        # If multiple dimensions match, choose the one with the most
        # explicitly mentioned categories. Decline ties as ambiguous.
        max_count = max(
            len(matches_by_dimension[d])
            for d in comparison_dimensions
        )
        best_dimensions = [
            d for d in comparison_dimensions
            if len(matches_by_dimension[d]) == max_count
        ]

        if len(best_dimensions) != 1:
            return None

        dimension = best_dimensions[0]
        return {
            "intent": "compare",
            "metric": metric,
            "dimension": dimension,
            "direction": "max",
            "segment_value": "",
            "segment_values": matches_by_dimension[dimension],
        }

    # Rank questions, such as "Which region has the lowest default rate?"
    is_highest = bool(re.search(
        r"\b(highest|largest|most|maximum|max|top|best)\b", q
    ))
    is_lowest = bool(re.search(
        r"\b(lowest|smallest|least|minimum|min|bottom|worst)\b", q
    ))
    asks_rank = is_highest or is_lowest

    if asks_rank:
        explicit_dimensions = [
            dimension for dimension in dimensions
            if re.search(
                r"\bby\s+" + re.escape(_normalize_text(dimension)) + r"\b",
                q,
            )
        ]

        # Also recognize natural names such as "by loan type" or "by age".
        dimension_aliases = {
            "loan_type": r"\bloan types?\b",
            "region": r"\bregions?\b",
            "employment_type": r"\bemployment types?\b|\bemployment\b",
            "income_band": r"\bincome bands?\b|\bincome groups?\b|\bincome\b",
            "age_band": r"\bage bands?\b|\bage groups?\b|\bage\b",
        }

        mentioned_dimensions = [
            dim for dim, pattern in dimension_aliases.items()
            if re.search(pattern, q)
        ]

        candidates = list(dict.fromkeys(
            explicit_dimensions + mentioned_dimensions
        ))

        # If a category is mentioned, its dimension is a useful clue.
        if not candidates:
            candidates = [
                dim for dim, values in matches_by_dimension.items()
                if values
            ]

        if len(candidates) != 1:
            return None

        return {
            "intent": "rank",
            "metric": metric,
            "dimension": candidates[0],
            "direction": "min" if is_lowest and not is_highest else "max",
            "segment_value": "",
            "segment_values": [],
        }

    # A single recognized category indicates a segment-level question.
    matched_segments = [
        (dimension, values[0])
        for dimension, values in matches_by_dimension.items()
        if len(values) == 1
    ]

    if len(matched_segments) == 1:
        dimension, segment_value = matched_segments[0]
        return {
            "intent": "segment_metric",
            "metric": metric,
            "dimension": dimension,
            "direction": "max",
            "segment_value": segment_value,
            "segment_values": [],
        }

    # If no segment is specified, treat it as a portfolio-wide metric.
    if not matches_by_dimension:
        return {
            "intent": "overall",
            "metric": metric,
            "dimension": "loan_type",
            "direction": "max",
            "segment_value": "",
            "segment_values": [],
        }

    # Multiple categories without a clear comparison are ambiguous.
    return None


def answer_portfolio_question(question: str, df: pd.DataFrame) -> dict[str, str]:
    """Answer a question from the current filtered dashboard rows.

    Returns {'answer': ..., 'source': ...}. Unsupported or unavailable questions
    are declined rather than answered from model knowledge.
    """
    question = question.strip()
    if not question:
        return {"answer": "Please enter a question about this portfolio.", "source": ""}

    
    # Parse supported questions locally first.
    plan = _deterministic_query_plan(question, df)

    if plan is None:
        # Use Gemini when local parsing cannot confidently interpret
        # the question. Keep all numeric calculations in pandas.
        api_key = _get_api_key()

        if not api_key:
            return {
                "answer": (
                    "I couldn't confidently interpret that question locally, "
                    "and Gemini is not configured. Please check your API key "
                    "or rephrase the question."
                ),
                "source": "",
            }

        try:
            plan = _query_plan(question, api_key)
        except Exception as exc:
            import logging
            logging.exception("Gemini query planning failed")
            return {
                "answer": (
                    "I couldn't interpret the question reliably. "
                    "Try specifying the metric and segment explicitly."
                ),
                "source": (
                    f"Query planning failed: {type(exc).__name__}. "
                    "See application logs for details."
                ),
            }

    # Continue with your existing code starting at:
    # intent = plan["intent"]
    # metric = plan["metric"]
    # dimension = plan["dimension"]


    # api_key = _get_api_key()
    # if not api_key:
    #     return {
    #         "answer": "Ask the Portfolio is not configured yet. Add GEMINI_API_KEY to your local environment or Streamlit Cloud secrets.",
    #         "source": "",
    #     }

    # try:
    #     plan = _query_plan(question, api_key)

    # except Exception as exc:
    #     import logging

    #     logging.exception("Ask the Portfolio query planning failed")

    #     return {
    #         "answer": (
    #             "I couldn't process this question. Please try again, "
    #             "or ask about a supported portfolio metric."
    #         ),
    #         "source": (
    #             f"Query planning failed: {type(exc).__name__}. "
    #             "See the application logs for details."
    #         ),
    #     }
    # except Exception as exc:
    #     # Keep the UI helpful without exposing secrets or raw provider details.
    #     return {
    #         "answer": "I couldn't interpret that question safely. Try asking about default rate, account count, customers, original exposure, or a comparison across loan type, region, employment, income band, or age band.",
    #         "source": "No portfolio query was executed.",
    #     }

    intent = plan["intent"]
    metric = plan["metric"]
    dimension = plan["dimension"]

    if intent == "unsupported":
        return {
            "answer": "I can't answer that reliably from the fields available in this dashboard. Try a question about default rates, account counts, customer counts, original exposure, or supported portfolio segments.",
            "source": "Question declined; no numeric answer generated.",
        }

    try:
        if intent == "overall":
            summary = _summary(df)
            value = summary.iloc[0][metric]
            answer = f"The {METRICS[metric]} for the currently filtered portfolio is {_fmt(metric, value)}."
            source = f"Source: current dashboard-filtered account data; {int(summary.iloc[0]['accounts']):,} eligible accounts with target_default in {{0, 1}}. Exposure metrics use non-missing, non-negative loan_amount values. Metric: {metric}."
            return {"answer": answer, "source": source}

        summary = _summary(df, dimension)
        if summary.empty:
            return {"answer": "There are no eligible accounts in the current filter selection.", "source": "Current dashboard filters."}

        if intent == "rank":
            ascending = plan["direction"] == "min"
            ranked = summary.dropna(subset=[metric]).sort_values(metric, ascending=ascending)
            if ranked.empty:
                return {"answer": "That metric is not available for the current selection.", "source": f"Grouped by {DIMENSIONS[dimension].lower()} from the filtered account data."}
            top = ranked.iloc[0]
            label = top["segment"] if pd.notna(top["segment"]) else "Unknown"
            answer = (
                f"{label} has the {'lowest' if ascending else 'highest'} {METRICS[metric]} "
                f"at {_fmt(metric, top[metric])} ({int(top['accounts']):,} accounts; "
                f"{int(top['defaulted_accounts']):,} defaulted)."
            )
            source = (
                f"Source: current dashboard-filtered account data, grouped by {DIMENSIONS[dimension].lower()}. "
                f"Metric: {metric}; denominator for default rate is all eligible accounts in each group."
            )
            return {"answer": answer, "source": source}

        if intent == "segment_metric":
            requested = str(plan.get("segment_value", "")).strip()
            if not requested:
                return {"answer": "I couldn't identify the segment you meant. Please specify an exact category, such as Mortgage or South.", "source": ""}
            matches = summary[summary["segment"].str.casefold() == requested.casefold()]
            if matches.empty:
                values = ", ".join(sorted(summary["segment"].dropna().astype(str).unique()))
                return {
                    "answer": f"I can't find '{requested}' in the current filtered data for {DIMENSIONS[dimension].lower()}.",
                    "source": f"Available categories in this filtered view: {values or 'none'}.",
                }
            row = matches.iloc[0]
            label = row["segment"]
            answer = f"For {label}, the {METRICS[metric]} is {_fmt(metric, row[metric])} across {int(row['accounts']):,} accounts."
            source = f"Source: current dashboard-filtered account data, filtered to {DIMENSIONS[dimension].lower()} = {label}. Metric: {metric}."
            return {"answer": answer, "source": source}

        if intent == "compare":
            values = [str(v).strip() for v in plan.get("segment_values", []) if str(v).strip()]
            if len(values) < 2:
                return {"answer": "Please name at least two categories to compare, such as Mortgage and Auto Loan.", "source": ""}
            rows = []
            for value in values:
                match = summary[summary["segment"].str.casefold() == value.casefold()]
                if match.empty:
                    return {
                        "answer": f"I can't find '{value}' in the current filtered data for {DIMENSIONS[dimension].lower()}.",
                        "source": "No comparison was generated because one or more categories were unavailable.",
                    }
                rows.append(match.iloc[0])
            parts = [f"{r['segment']}: {_fmt(metric, r[metric])}" for r in rows]
            answer = f"Comparison of {METRICS[metric]} — " + "; ".join(parts) + "."
            source = f"Source: current dashboard-filtered account data, grouped by {DIMENSIONS[dimension].lower()}; metric: {metric}."
            return {"answer": answer, "source": source}

    except Exception:
        return {
            "answer": "I couldn't calculate that answer from the current dashboard data. Try a simpler supported question.",
            "source": "No unverified figures were returned.",
        }

    return {"answer": "I can't answer that question from the supported portfolio metrics.", "source": ""}
