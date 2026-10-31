"""Streamlit dashboard entry point.

Portfolio view is implemented first. AI features are intentionally deferred.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analytics import (
    build_segment_summary,
    filter_portfolio,
    income_employment_crosstab,
    portfolio_kpis,
)
from src.config import ACCOUNT_DATA_PATH, MODEL_PATH, MODEL_FEATURES_PATH, FEATURE_IMPORTANCE_PATH
from src.model import load_model_artifacts, predict_default_probability
from src.portfolio_qa import answer_portfolio_question
from src.anomaly_reviews import (
    build_review_queue,
    get_review_summary,
    save_analyst_decision,
)

st.set_page_config(page_title="Credit Risk Portfolio Analytics", page_icon="📊", layout="wide")

st.markdown(
    """
    <style>
    /* Visual-only styling: layout, colors, spacing, and typography. */
    .stApp {
        background: #f6f8fc;
        color: #172033;
    }
    [data-testid="stHeader"] {
        background: rgba(246, 248, 252, 0.9);
    }
    [data-testid="stSidebar"] {
        background: #eef2f8;
        border-right: 1px solid #dce3ef;
    }
    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 14px 16px;
        box-shadow: 0 2px 8px rgba(25, 45, 80, 0.04);
    }
    [data-testid="stMetricLabel"] {
        color: #64748b;
        font-weight: 600;
    }
    [data-testid="stMetricValue"] {
        color: #172033;
        font-weight: 700;
    }
    [data-testid="stPlotlyChart"],
    [data-testid="stDataFrame"],
    [data-testid="stTable"] {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 6px;
    }
    div[data-testid="stForm"],
    div[data-testid="stExpander"] {
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        background: #ffffff;
    }
    .stButton > button,
    .stFormSubmitButton > button {
        border-radius: 8px;
        font-weight: 600;
    }
    hr {
        border-color: #dce3ef;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("Credit Risk Portfolio Analytics")
st.caption("Portfolio monitoring and account-level risk exploration")


@st.cache_data
def load_account_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ("origination_date", "customer_since", "event_max_date"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    if "income_band" not in df.columns and "annual_income" in df.columns:
        df["income_band"] = pd.cut(
            df["annual_income"],
            [-np.inf, 25_000, 50_000, 75_000, 100_000, np.inf],
            labels=["Below $25k", "$25k–$50k", "$50k–$75k", "$75k–$100k", "$100k+"],
            right=False,
        )
    if "age_band" not in df.columns and "age" in df.columns:
        df["age_band"] = pd.cut(
            df["age"], [-np.inf, 25, 35, 45, 55, 65, np.inf],
            labels=["Under 25", "25–34", "35–44", "45–54", "55–64", "65+"],
            right=False,
        )
    return df


if not ACCOUNT_DATA_PATH.exists():
    st.error(
        f"Processed data not found at {ACCOUNT_DATA_PATH}. "
        "Place raw CSVs in data/raw and run `python -m src.ingestion`."
    )
    st.stop()

data = load_account_data(str(ACCOUNT_DATA_PATH))

st.sidebar.header("Portfolio filters")
status = st.sidebar.selectbox("Default status", ["All", "Defaulted", "Performing"])

loan_types = st.sidebar.multiselect(
    "Loan type", sorted(data["loan_type"].dropna().astype(str).unique()),
    default=sorted(data["loan_type"].dropna().astype(str).unique()),
)
regions = st.sidebar.multiselect(
    "Region", sorted(data["region"].dropna().astype(str).unique()),
    default=sorted(data["region"].dropna().astype(str).unique()),
)
employment_types = st.sidebar.multiselect(
    "Employment type",
    sorted(data["employment_type"].dropna().astype(str).unique()),
    default=sorted(data["employment_type"].dropna().astype(str).unique()),
)

filtered = filter_portfolio(
    data,
    loan_types=loan_types,
    regions=regions,
    employment_types=employment_types,
    default_status=status,
)

if filtered.empty:
    st.warning("No accounts match the selected filters.")
    st.stop()

kpis = portfolio_kpis(filtered)
st.subheader("Portfolio overview")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Loan accounts", f'{kpis["accounts"]:,}')
c2.metric("Customers", f'{kpis["customers"]:,}')
c3.metric("Default rate", f'{kpis["default_rate_pct"]:.2f}%')
c4.metric("Defaulted accounts", f'{kpis["defaulted_accounts"]:,}')
c5.metric("Original exposure", f'${kpis["original_exposure"]:,.0f}')

loan_summary = build_segment_summary(filtered, "loan_type")
income_summary = build_segment_summary(filtered, "income_band")
employment_summary = build_segment_summary(filtered, "employment_type")
region_summary = build_segment_summary(filtered, "region")
age_summary = build_segment_summary(filtered, "age_band")

left, right = st.columns(2)
with left:
    st.markdown("#### Default rate by loan type")
    fig = px.bar(
        loan_summary.sort_values("default_rate_pct"),
        x="loan_type", y="default_rate_pct",
        hover_data=["accounts", "defaulted_accounts", "exposure"],
        labels={"loan_type": "Loan type", "default_rate_pct": "Default rate (%)"},
    )
    st.plotly_chart(fig, use_container_width=True)
with right:
    st.markdown("#### Original exposure by loan type")
    fig = px.bar(
        loan_summary.sort_values("exposure", ascending=False),
        x="loan_type", y="exposure",
        hover_data=["accounts", "default_rate_pct"],
        labels={"loan_type": "Loan type", "exposure": "Original exposure ($)"},
    )
    st.plotly_chart(fig, use_container_width=True)

left, right = st.columns(2)
with left:
    st.markdown("#### Default rate by income")
    fig = px.bar(
        income_summary, x="income_band", y="default_rate_pct",
        category_orders={"income_band": ["Below $25k", "$25k–$50k", "$50k–$75k", "$75k–$100k", "$100k+"]},
        hover_data=["accounts", "defaulted_accounts", "exposure"],
        labels={"income_band": "Annual income", "default_rate_pct": "Default rate (%)"},
    )
    st.plotly_chart(fig, use_container_width=True)
with right:
    st.markdown("#### Default rate by employment")
    fig = px.bar(
        employment_summary.sort_values("default_rate_pct"),
        x="employment_type", y="default_rate_pct",
        hover_data=["accounts", "defaulted_accounts", "exposure"],
        labels={"employment_type": "Employment type", "default_rate_pct": "Default rate (%)"},
    )
    st.plotly_chart(fig, use_container_width=True)

left, right = st.columns(2)
with left:
    st.markdown("#### Default rate by region")
    fig = px.bar(
        region_summary.sort_values("default_rate_pct"),
        x="region", y="default_rate_pct",
        hover_data=["accounts", "defaulted_accounts", "exposure"],
        labels={"region": "Region", "default_rate_pct": "Default rate (%)"},
    )
    st.plotly_chart(fig, use_container_width=True)
with right:
    st.markdown("#### Default rate by age")
    fig = px.bar(
        age_summary, x="age_band", y="default_rate_pct",
        category_orders={"age_band": ["Under 25", "25–34", "35–44", "45–54", "55–64", "65+"]},
        hover_data=["accounts", "defaulted_accounts", "exposure"],
        labels={"age_band": "Age group", "default_rate_pct": "Default rate (%)"},
    )
    st.plotly_chart(fig, use_container_width=True)


st.markdown("#### Default rate by income and employment")
cross_tab = income_employment_crosstab(filtered)
rate_matrix = cross_tab.pivot(index="income_band", columns="employment_type", values="default_rate_pct")
count_matrix = cross_tab.pivot(index="income_band", columns="employment_type", values="accounts")
display_matrix = rate_matrix.where(count_matrix >= 30)
fig = go.Figure(data=go.Heatmap(
    z=display_matrix.to_numpy(),
    x=display_matrix.columns.astype(str),
    y=display_matrix.index.astype(str),
    colorscale="YlOrRd",
    zmin=0, zmax=50,
    colorbar={"title": "Default rate (%)"},
    hovertemplate="Income: %{y}<br>Employment: %{x}<br>Default rate: %{z:.1f}%<extra></extra>",
))
fig.update_layout(xaxis_title="Employment type", yaxis_title="Income band")
st.plotly_chart(fig, use_container_width=True)
st.caption("Heatmap cells with fewer than 30 accounts are hidden.")


st.divider()
st.subheader("Ask the Portfolio")
st.caption(
    "Ask a question about the currently filtered portfolio. Gemini maps your question "
    "to a supported query; all figures are calculated locally from the dashboard data."
)
with st.expander("What can I ask? / How this works", expanded=False):
    st.markdown(
        "- Which loan type has the highest default rate?\n"
        "- What is the overall default rate?\n"
        "- How many accounts are in the South region?\n"
        "- What is the original exposure for Mortgage loans?\n"
        "- Compare default rates for Mortgage and Auto Loan.\n\n"
        "**Trigger:** submit a question. **False positive:** a question is mapped to the wrong supported metric or segment. "
        "**False negative:** a valid question is declined. **Human review:** verify the displayed source and the relevant chart before using the result."
    )

with st.form("ask_portfolio_form", clear_on_submit=False):
    portfolio_question = st.text_input(
        "Your question",
        placeholder="Which loan type has the highest default rate?",
        key="portfolio_question",
    )
    ask_submitted = st.form_submit_button("Ask", type="primary")

if ask_submitted:
    with st.spinner("Checking the portfolio data..."):
        qa_result = answer_portfolio_question(portfolio_question, filtered)
    st.session_state["portfolio_qa_result"] = qa_result

qa_result = st.session_state.get("portfolio_qa_result")
if qa_result:
    st.markdown("**Answer**")
    st.write(qa_result.get("answer", "No answer returned."))
    if qa_result.get("source"):
        st.caption(qa_result["source"])


# ---------------------------------------------------------
# ANOMALY REVIEW
# ---------------------------------------------------------

st.divider()
st.subheader("Anomaly Detection & Analyst Review")

st.caption(
    "Accounts are flagged when their repayment behaviour differs "
    "from comparable peers or deteriorates relative to their own "
    "observed repayment history."
)

st.markdown(
    """
    **How this works**

    - **Trigger:** unusual repayment patterns compared with peer accounts,
      or a significant deterioration between the first and last six
      observed payment periods.
    - **False positive:** an unusual pattern is flagged but is explainable
      by a legitimate account circumstance.
    - **False negative:** an unusual pattern is not flagged by the
      selected statistical thresholds.
    - **Human review:** an analyst examines the account and its reasons,
      then records a Confirmed or Dismissed decision with an optional comment.
    """
)

ANOMALY_PATH = PROJECT_ROOT / "data" / "processed" / "combined_anomalies.csv"
DECISIONS_PATH = PROJECT_ROOT / "data" / "processed" / "analyst_decisions.json"

if not ANOMALY_PATH.exists():
    st.warning(
        "Combined anomaly results were not found. "
        "Run the anomaly detection notebook and export "
        "combined_anomalies.csv to data/processed/."
    )
else:
    try:
        combined_anomalies = pd.read_csv(ANOMALY_PATH)

        # Build the current review queue using saved analyst decisions.
        review_queue = build_review_queue(
            combined_anomalies=combined_anomalies,
            json_path=str(DECISIONS_PATH),
        )

        # Add account profile columns for analyst context.
        profile_columns = [
            "account_id",
            "customer_id",
            "loan_type",
            "loan_amount",
            "region",
            "employment_type",
            "income_band",
            "age_band",
            "target_default",
        ]

        profile_columns = [
            col for col in profile_columns
            if col in data.columns
        ]

        # Avoid duplicate columns when merging.
        profile_columns = [
            col for col in profile_columns
            if col == "account_id" or col not in review_queue.columns
        ]

        review_queue = review_queue.merge(
            data[profile_columns].drop_duplicates("account_id"),
            on="account_id",
            how="left",
        )

        summary = get_review_summary(review_queue)

        # Summary KPIs.
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Flagged accounts", f"{summary['total_flagged']:,}")
        k2.metric("Pending review", f"{summary['pending']:,}")
        k3.metric("Confirmed", f"{summary['confirmed']:,}")
        k4.metric("Dismissed", f"{summary['dismissed']:,}")

        st.divider()

        # Review filters.
        filter_col1, filter_col2 = st.columns(2)

        with filter_col1:
            priority_options = ["All"] + sorted(
                review_queue["review_priority"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            selected_priority = st.selectbox(
                "Filter by priority",
                priority_options,
                key="anomaly_priority_filter",
            )

        with filter_col2:
            review_status_options = [
                "All",
                "Pending",
                "Confirmed",
                "Dismissed",
            ]

            selected_review_status = st.selectbox(
                "Filter by review status",
                review_status_options,
                key="anomaly_status_filter",
            )

        filtered_reviews = review_queue.copy()

        if selected_priority != "All":
            filtered_reviews = filtered_reviews[
                filtered_reviews["review_priority"].astype(str)
                == selected_priority
            ]

        if selected_review_status != "All":
            filtered_reviews = filtered_reviews[
                filtered_reviews["review_status"]
                == selected_review_status
            ]

        st.markdown("#### Flagged accounts")

        display_columns = [
            "account_id",
            "loan_type",
            "loan_amount",
            "region",
            "anomaly_classification",
            "review_priority",
            "consolidated_anomaly_reason",
            "review_status",
        ]

        display_columns = [
            col for col in display_columns
            if col in filtered_reviews.columns
        ]

        st.dataframe(
            filtered_reviews[display_columns],
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            f"Showing {len(filtered_reviews):,} of "
            f"{len(review_queue):,} flagged accounts."
        )

        st.divider()
        st.markdown("#### Review an account")

        if filtered_reviews.empty:
            st.info("No flagged accounts match the selected filters.")
        else:
            account_options = (
                filtered_reviews["account_id"]
                .astype(str)
                .tolist()
            )

            selected_anomaly_id = st.selectbox(
                "Select an account to review",
                account_options,
                key="selected_anomaly_account",
            )

            anomaly_row = filtered_reviews.loc[
                filtered_reviews["account_id"].astype(str)
                == selected_anomaly_id
            ].iloc[0]

            # Account context.
            a1, a2, a3, a4 = st.columns(4)

            a1.metric(
                "Loan type",
                str(anomaly_row.get("loan_type", "N/A")),
            )

            loan_amount = anomaly_row.get("loan_amount", np.nan)
            a2.metric(
                "Original loan amount",
                f"${loan_amount:,.0f}"
                if pd.notna(loan_amount)
                else "N/A",
            )

            a3.metric(
                "Anomaly classification",
                str(anomaly_row.get("anomaly_classification", "N/A")),
            )

            a4.metric(
                "Review priority",
                str(anomaly_row.get("review_priority", "N/A")),
            )

            st.markdown("**Why was this account flagged?**")

            st.info(
                str(
                    anomaly_row.get(
                        "consolidated_anomaly_reason",
                        "No anomaly explanation available.",
                    )
                )
            )

            # Show useful supporting repayment metrics, if present.
            supporting_metrics = [
                "missed_payment_rate_pct",
                "late_payment_rate_pct",
                "average_days_past_due",
                "maximum_days_past_due",
                "repayment_rate_pct",
                "repayment_rate_drop_pp",
                "first_period_repayment_rate_pct",
                "last_period_repayment_rate_pct",
            ]

            available_metrics = [
                col for col in supporting_metrics
                if col in anomaly_row.index
            ]

            if available_metrics:
                st.markdown("**Supporting repayment metrics**")

                metric_data = pd.DataFrame({
                    "Metric": available_metrics,
                    "Value": [
                        anomaly_row[col]
                        for col in available_metrics
                    ],
                })

                st.dataframe(
                    metric_data,
                    use_container_width=True,
                    hide_index=True,
                )

            current_status = anomaly_row.get(
                "review_status",
                "Pending",
            )

            st.markdown("**Analyst decision**")

            if current_status in {"Confirmed", "Dismissed"}:
                st.caption(
                    f"Previously reviewed: {current_status}. "
                    "Submitting a new decision will update the saved record."
                )

            with st.form(
                "anomaly_review_form",
                clear_on_submit=False,
            ):
                decision_options = [
                    "Select decision",
                    "Confirmed",
                    "Dismissed",
                ]

                default_decision_index = (
                    decision_options.index(current_status)
                    if current_status in decision_options
                    else 0
                )

                decision = st.selectbox(
                    "Decision",
                    decision_options,
                    index=default_decision_index,
                )

                comment = st.text_area(
                    "Analyst comment",
                    value=str(
                        anomaly_row.get("analyst_comment", "") or ""
                    ),
                    placeholder=(
                        "Explain why you confirmed or dismissed "
                        "this anomaly..."
                    ),
                )

                submitted = st.form_submit_button(
                    "Save review decision",
                    type="primary",
                )

                if submitted:
                    try:
                        if decision == "Select decision":
                            st.warning("Please select Confirmed or Dismissed.")
                        else:
                            save_analyst_decision(
                                json_path=str(DECISIONS_PATH),
                                account_id=selected_anomaly_id,
                                decision=decision,
                                comment=comment,
                            )
                            st.success(f"Decision saved for {selected_anomaly_id}.")
                            st.rerun()

                    except Exception as exc:
                        st.error("Could not save the analyst decision.")
                        st.exception(exc)

    except Exception as exc:
        st.error("Could not load the anomaly review section.")
        st.exception(exc)



st.divider()
st.subheader("Account explorer")
options = filtered["account_id"].astype(str).tolist()
selected_id = st.selectbox("Select account ID", options)
selected = filtered.loc[filtered["account_id"].astype(str).eq(selected_id)].iloc[0]
# ---------------------------------------------------------
# SELECTED ACCOUNT: ANOMALY PROFILE
# ---------------------------------------------------------

st.markdown("#### Anomaly profile")

ANOMALY_PATH = (
    PROJECT_ROOT / "data" / "processed" / "combined_anomalies.csv"
)
DECISIONS_PATH = (
    PROJECT_ROOT / "data" / "processed" / "analyst_decisions.json"
)

if ANOMALY_PATH.exists():
    try:
        account_anomalies = pd.read_csv(ANOMALY_PATH)

        account_review_queue = build_review_queue(
            combined_anomalies=account_anomalies,
            json_path=str(DECISIONS_PATH),
        )

        account_anomaly = account_review_queue.loc[
            account_review_queue["account_id"].astype(str).eq(selected_id)
        ]

        if account_anomaly.empty:
            st.success("No anomaly detected for this account.")

        else:
            anomaly = account_anomaly.iloc[0]

            status_col, priority_col, class_col = st.columns(3)

            status_col.metric(
                "Anomaly status",
                "Flagged",
            )

            priority_col.metric(
                "Review priority",
                str(anomaly.get("review_priority", "N/A")),
            )

            class_col.metric(
                "Classification",
                str(anomaly.get("anomaly_classification", "N/A")),
            )

            st.markdown("**Why was this account flagged?**")

            st.info(
                str(
                    anomaly.get(
                        "consolidated_anomaly_reason",
                        "No explanation available.",
                    )
                )
            )

            st.markdown("**Analyst review**")

            review_status = anomaly.get("review_status", "Pending")

            if review_status == "Confirmed":
                st.success("Anomaly confirmed by analyst.")
            elif review_status == "Dismissed":
                st.warning("Anomaly dismissed by analyst.")
            else:
                st.info("Pending analyst review.")

            comment = anomaly.get("analyst_comment", "")

            if pd.notna(comment) and str(comment).strip():
                st.caption(f"Analyst comment: {comment}")

            reviewed_at = anomaly.get("reviewed_at", "")

            if pd.notna(reviewed_at) and str(reviewed_at).strip():
                st.caption(f"Reviewed at: {reviewed_at}")

            # Supporting repayment metrics
            supporting_metrics = {
                "missed_payment_rate_pct": "Missed payment rate",
                "late_payment_rate_pct": "Late payment rate",
                "average_days_past_due": "Average days past due",
                "maximum_days_past_due": "Maximum days past due",
                "repayment_rate_pct": "Repayment rate",
                "repayment_rate_drop_pp": "Repayment rate deterioration (pp)",
            }

            available_metrics = [
                (column, label)
                for column, label in supporting_metrics.items()
                if column in anomaly.index
                and pd.notna(anomaly[column])
            ]

            if available_metrics:
                st.markdown("**Repayment indicators**")

                metric_df = pd.DataFrame({
                    "Metric": [label for _, label in available_metrics],
                    "Value": [
                        round(float(anomaly[column]), 2)
                        for column, _ in available_metrics
                    ],
                })

                st.dataframe(
                    metric_df,
                    use_container_width=True,
                    hide_index=True,
                )

    except Exception as exc:
        st.warning("Could not load this account's anomaly information.")
        st.exception(exc)

else:
    st.info(
        "Anomaly results are not available yet. "
        "Run the anomaly detection pipeline first."
    )
details = st.columns(4)
details[0].metric("Loan type", str(selected.get("loan_type", "N/A")))
amount = selected.get("loan_amount", np.nan)
details[1].metric("Original loan amount", f"${amount:,.0f}" if pd.notna(amount) else "N/A")
details[2].metric("Observed label", "Defaulted" if selected["target_default"] == 1 else "Performing")
details[3].metric("Region", str(selected.get("region", "N/A")))

st.write("**Borrower profile**")
profile_fields = ["age", "annual_income", "employment_type", "education_level", "region", "years_employed"]
profile_fields = [col for col in profile_fields if col in selected.index]
st.dataframe(pd.DataFrame({"Field": profile_fields, "Value": [selected[col] for col in profile_fields]}), hide_index=True, use_container_width=True)

if MODEL_PATH.exists() and MODEL_FEATURES_PATH.exists():
    try:
        model, features = load_model_artifacts()
        probability = predict_default_probability(model, pd.DataFrame([selected]), features)[0]
        st.markdown("#### Model-estimated default probability")
        st.metric("Estimated probability", f"{probability:.1%}")
        st.progress(float(np.clip(probability, 0, 1)))
        st.caption("This is a model estimate, not a guarantee. Feature timing limitations are documented in the README.")
    except Exception as exc:
        st.error("Model scoring failed. Verify that the saved pipeline and input feature columns match.")
        st.exception(exc)
else:
    st.info("Model artifacts are not present yet. The portfolio dashboard works, but risk scores require the fitted pipeline and feature list.")

st.markdown("#### Main model drivers")
if FEATURE_IMPORTANCE_PATH.exists():
    importance = pd.read_csv(FEATURE_IMPORTANCE_PATH)
    importance.columns = [c.lower() for c in importance.columns]
    if {"feature", "importance_mean"}.issubset(importance.columns):
        top = importance.nlargest(10, "importance_mean").sort_values("importance_mean")
        fig = px.bar(top, x="importance_mean", y="feature", orientation="h",
                     labels={"importance_mean": "Permutation importance", "feature": "Feature"})
        st.plotly_chart(fig, use_container_width=True)
        st.caption("These are global model drivers, not account-specific causal explanations.")
    else:
        st.info("Importance CSV must contain feature and importance_mean columns.")
else:
    st.info("Export feature_importance.csv from the notebook to display model-wide drivers.")

with st.expander("View filtered accounts"):
    cols = [c for c in ["account_id", "customer_id", "loan_type", "loan_amount", "region", "employment_type", "target_default"] if c in filtered.columns]
    st.dataframe(filtered[cols], use_container_width=True, hide_index=True)
