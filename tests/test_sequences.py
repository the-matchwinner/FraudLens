
import pandas as pd

from src.sequences import detect_suspicious_sequences


def test_detects_rapid_low_value_sequence():
    df = pd.DataFrame({
        "transaction_id": ["T1", "T2", "T3", "T4"],
        "customer_id": ["C1"] * 4,
        "timestamp": pd.date_range(
            "2026-10-23 12:00:00",
            periods=4,
            freq="min",
        ),
        "amount": [1.0, 2.0, 5.0, 9.0],
        "scenario": ["normal"] * 4,
        "is_fraud": [0] * 4,
    })

    result = detect_suspicious_sequences(df)

    assert len(result) == 1
    assert result.iloc[0]["transaction_count"] == 4
    assert result.iloc[0]["total_amount"] == 17.0
    assert result.iloc[0]["pattern"] == "rapid_low_value_burst"


def test_does_not_flag_large_transactions():
    df = pd.DataFrame({
        "transaction_id": ["T1", "T2", "T3", "T4"],
        "customer_id": ["C1"] * 4,
        "timestamp": pd.date_range(
            "2026-10-23 12:00:00",
            periods=4,
            freq="min",
        ),
        "amount": [100.0, 200.0, 300.0, 400.0],
    })

    result = detect_suspicious_sequences(df)

    assert result.empty
