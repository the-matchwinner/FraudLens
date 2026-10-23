# FraudLens 🛡️

**Evidence-driven transaction fraud investigation**

FraudLens is a prototype that combines transparent rule-based risk scoring, machine-learning anomaly detection, suspicious transaction sequence detection, and an investigation copilot.

## Key Features

- **Behavioural risk scoring:** identifies unusual transaction amounts, unfamiliar locations, merchant categories, and transaction bursts.
- **ML anomaly detection:** uses Isolation Forest to rank unusual transactions.
- **Sequence intelligence:** groups rapid, low-value transactions into investigation cases.
- **Investigation copilot:** summarises detected evidence and recommends investigation steps.
- **Interactive dashboard:** explores transactions, risk signals, and investigation priorities.

## Tech Stack

Python · Pandas · NumPy · Scikit-learn · Streamlit · Plotly · Pytest

## Run Locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.generate_data
streamlit run app.py
```

## Testing

```bash
python -m pytest -q
```

## Important Limitations

This prototype uses synthetic transaction data. Its scores and anomaly rankings are investigative signals, not calibrated fraud probabilities or proof of fraud. Results on synthetic data do not establish real-world detection performance. Human review is required before decisions are made.

The investigation copilot currently produces evidence-based explanations using application logic; it should not be described as a live external LLM integration unless one has actually been connected.
