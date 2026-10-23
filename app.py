
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.express as px

from src.features import build_features
from src.scoring import score_transaction
from src.sequences import detect_suspicious_sequences
from src.ml_engine import detect_anomalies
from src.copilot import explain_investigation

DATA_PATH = Path("data/transactions.csv")

st.set_page_config(
    page_title="FraudLens | Fraud Investigation",
    page_icon="🛡️",
    layout="wide",
)




st.markdown("""
<style>
/* ===== FraudLens: Light Theme ===== */

.stApp,
[data-testid="stAppViewContainer"] {
    background-color: #F5F7FB !important;
    color: #172033 !important;
}

/* Main page header */
[data-testid="stHeader"] {
    background-color: #FFFFFF !important;
}

/* Main content spacing */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Typography */
.stApp h1,
.stApp h2,
.stApp h3,
.stApp h4 {
    color: #14213D !important;
}

.stApp p,
.stApp label,
.stApp li {
    color: #334155;
}

/* Metric cards */
[data-testid="stMetric"] {
    background-color: #FFFFFF !important;
    border: 1px solid #DFE6F0 !important;
    border-radius: 14px !important;
    padding: 18px !important;
    box-shadow: 0 2px 8px rgba(15, 23, 42, 0.035);
}

[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] p {
    color: #64748B !important;
}

[data-testid="stMetricValue"],
[data-testid="stMetricValue"] div {
    color: #172033 !important;
}

/* Containers and bordered panels */
[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: #FFFFFF;
    border: 1px solid #DFE6F0 !important;
    border-radius: 14px !important;
}

/* Dropdowns and input fields */
[data-baseweb="select"] > div {
    background-color: #FFFFFF !important;
    color: #172033 !important;
    border-color: #CBD5E1 !important;
    border-radius: 9px !important;
}

[data-baseweb="select"] input {
    color: #172033 !important;
}

.stTextInput input,
.stNumberInput input,
.stTextArea textarea {
    background-color: #FFFFFF !important;
    color: #172033 !important;
    border: 1px solid #CBD5E1 !important;
    border-radius: 9px !important;
}

/* Buttons */
.stButton > button {
    background-color: #2563EB;
    color: #FFFFFF !important;
    border: none;
    border-radius: 9px;
    font-weight: 600;
}

.stButton > button:hover {
    background-color: #1D4ED8;
    color: #FFFFFF !important;
}

/* Informational messages */
[data-testid="stAlert"] {
    border-radius: 12px !important;
}

/* Tables */
[data-testid="stDataFrame"] {
    border: 1px solid #DFE6F0;
    border-radius: 10px;
    overflow: hidden;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #FFFFFF !important;
    border-right: 1px solid #DFE6F0;
}

/* Dividers */
hr {
    border-color: #DFE6F0 !important;
}

/* Links */
.stApp a {
    color: #2563EB;
}

/* Footer */
footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)



st.title("🛡️ FraudLens")
st.caption("Evidence-driven transaction risk investigation")
st.info(
    "Prototype notice: this demo uses synthetic transactions and "
    "transparent heuristic scores, not a production fraud model."
)

if not DATA_PATH.exists():
    st.error("Transaction dataset not found. Run: python -m src.generate_data")
    st.stop()

raw = pd.read_csv(DATA_PATH)
features = build_features(raw)
ml_results = detect_anomalies(features)
sequences = detect_suspicious_sequences(raw)

results = pd.DataFrame([
    score_transaction(row)
    for row in features.to_dict(orient="records")
])

results = results.merge(
    ml_results,
    on="transaction_id",
    how="left"
)

data = features.merge(
    results,
    on="transaction_id",
    validate="one_to_one",
)

total = len(data)
high_risk = int((data["status"] == "high_risk_review").sum())
review = int((data["status"] == "review_recommended").sum())
low_risk = int((data["status"] == "low_risk").sum())

m1, m2, m3, m4 = st.columns(4)
m1.metric("Transactions analysed", f"{total:,}")
m2.metric("High-risk reviews", f"{high_risk:,}")
m3.metric("Review recommended", f"{review:,}")
m4.metric("Low risk", f"{low_risk:,}")

st.divider()

left, right = st.columns([1, 2])

with left:
    st.subheader("Investigate a transaction")

    transaction_id = st.selectbox(
        "Choose transaction ID",
        data["transaction_id"].tolist(),
    )

    selected = data.loc[data["transaction_id"] == transaction_id].iloc[0]

    st.write("**Amount**")
    st.write(f"₹{selected['amount']:,.2f}")

    st.write("**Customer**")
    st.write(str(selected["customer_id"]))

    st.write("**Timestamp**")
    st.write(str(selected["timestamp"]))

    st.write("**City**")
    st.write(str(selected["city"]))

    st.write("**Merchant category**")
    st.write(str(selected["merchant_category"]))

    st.write("**Scenario (synthetic ground truth)**")
    st.write(str(selected["scenario"]))
    
    
    st.divider()
    st.subheader("🤖 FraudLens Investigation Copilot")

    selected_ml = {
        "ml_anomaly_score": selected.get("ml_anomaly_score", 0),
        "ml_anomaly_flag": selected.get("ml_anomaly_flag", False),
    }

    explanation = explain_investigation(
        selected.to_dict(),
        selected_ml,
    )


    st.metric(
            "ML anomaly percentile",
            f"{explanation['ml_anomaly_score']:.1f}/100",
        )

    
    st.write("**Evidence identified**")

    if explanation["evidence"]:
            for item in explanation["evidence"]:
                st.write(f"• {item}")
    else:
            st.write("No specific rule evidence recorded.")
            st.write("**Recommended investigation action**")
    st.info(explanation["recommended_action"])

    st.caption(explanation["decision_note"])


with right:
    st.subheader("Risk assessment")

    score = int(selected["risk_score"])
    status = str(selected["status"])

    st.metric("Heuristic risk score", f"{score}/100")

    if status == "high_risk_review":
        st.error("HIGH RISK — Urgent review recommended")
    elif status == "review_recommended":
        st.warning("REVIEW RECOMMENDED")
    else:
        st.success("LOW RISK — No immediate review trigger")

    st.write("**Evidence behind the score**")


    evidence = selected["evidence"]

    if isinstance(evidence, list) and evidence:
        for item in evidence:
            st.markdown(
                f"- **{item['signal']}** ({item['points']:+} points): "
                f"{item['explanation']}"
            )
    else:
        st.write("No risk signals were triggered.")

    st.caption(
        "The score is a heuristic, not a probability. "
        "A low-risk result does not guarantee a transaction is safe."
    )

st.divider()

st.divider()
st.subheader("🧭 Behaviour Anomaly Map")
st.caption(
    "Compare transaction amounts with ML anomaly scores. "
    "Highlighted points are model flags, not confirmed fraud."
)

chart_data = data.copy()

chart_data["ml_anomaly_score"] = pd.to_numeric(
    chart_data["ml_anomaly_score"], errors="coerce"
).fillna(0)

chart_data["amount"] = pd.to_numeric(
    chart_data["amount"], errors="coerce"
)

chart_data["ML signal"] = chart_data["ml_anomaly_flag"].map(
    {True: "Anomaly flagged", False: "Not flagged"}
)

fig = px.scatter(
    chart_data,
    x="amount",
    y="ml_anomaly_score",
    color="ML signal",
    hover_data=[
        "transaction_id",
        "customer_id",
        "city",
        "risk_score",
        "status",
    ],
    title="Transaction Behaviour vs. ML Anomaly Score",
    labels={
        "amount": "Transaction amount (₹)",
        "ml_anomaly_score": "Anomaly percentile (0–100)",
    },
    color_discrete_map={
        "Anomaly flagged": "#E45756",
        "Not flagged": "#2B83BA",
    },
)

fig.update_layout(
    template="plotly_white",
    height=430,
    margin=dict(l=20, r=20, t=60, b=20),
    legend_title_text="ML signal",
)

fig.update_yaxes(range=[0, 100])

st.plotly_chart(fig, use_container_width=True)

# Behaviour pattern distribution
st.divider()
st.subheader("🧬 Behaviour Pattern Intelligence")
st.caption(
    "Explore how transaction behaviour differs across the "
    "investigation dataset."
)

pattern_data = data.copy()

pattern_data["Behaviour pattern"] = "Routine activity"

pattern_data.loc[
    pattern_data["ml_anomaly_flag"].fillna(False).astype(bool),
    "Behaviour pattern"
] = "ML anomaly"

pattern_data.loc[
    pattern_data["risk_score"] >= 30,
    "Behaviour pattern"
] = "Rule-based risk"

pattern_counts = (
    pattern_data["Behaviour pattern"]
    .value_counts()
    .rename_axis("Behaviour pattern")
    .reset_index(name="Transactions")
)

fig_pattern = px.treemap(
    pattern_counts,
    path=["Behaviour pattern"],
    values="Transactions",
    color="Transactions",
    color_continuous_scale="Blues",
)

fig_pattern.update_layout(
    template="plotly_white",
    height=350,
    margin=dict(l=10, r=10, t=25, b=10),
)

st.plotly_chart(fig_pattern, use_container_width=True)
st.success("Behaviour Pattern Intelligence loaded!")
st.subheader("Transaction investigation queue")

scenario_filter = st.selectbox(
    "Filter by synthetic scenario",
    ["All scenarios"] + sorted(data["scenario"].unique().tolist()),
)

queue = data.copy()

if scenario_filter != "All scenarios":
    queue = queue[queue["scenario"] == scenario_filter]

queue = queue.sort_values("risk_score", ascending=False)

display_columns = [
    "transaction_id",
    "customer_id",
    "amount",
    "city",
    "risk_score",
    "status",
]

st.dataframe(
    queue[display_columns],
    use_container_width=True,
    hide_index=True,
)

st.caption(
    "FraudLens demonstration • Synthetic data only • "
    "All final decisions require human investigation."
)


st.divider()
st.subheader("🔎 Suspicious Transaction Sequences")
st.caption(
    "Groups rapid, low-value payments into investigation cases. "
    "A detected pattern is a review signal, not proof of fraud."
)

if sequences.empty:
    st.success("No rapid low-value sequences detected.")
else:
    st.metric("Sequences detected", len(sequences))

    st.dataframe(
        sequences[
            [
                "case_id",
                "customer_id",
                "start_time",
                "end_time",
                "transaction_count",
                "total_amount",
                "pattern",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    selected_case_id = st.selectbox(
        "Inspect a sequence",
        sequences["case_id"].tolist(),
    )

    selected_case = sequences.loc[
        sequences["case_id"] == selected_case_id
    ].iloc[0]

    st.write(f"**Case:** {selected_case['case_id']}")
    st.write(f"**Evidence:** {selected_case['explanation']}")
    st.write("**Transactions in this sequence:**")

    case_transactions = raw[
        raw["transaction_id"].isin(selected_case["transaction_ids"])
    ].sort_values("timestamp")

    st.dataframe(
        case_transactions[
            [
                "transaction_id",
                "timestamp",
                "amount",
                "city",
                "merchant_category",
                
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )
