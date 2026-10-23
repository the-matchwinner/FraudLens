
import pandas as pd

from src.features import build_features


def test_current_transaction_does_not_change_its_own_baseline():
    data = pd.DataFrame([
        {
            "transaction_id": "T1",
            "customer_id": "C1",
            "timestamp": "2026-09-01 10:00:00",
            "amount": 100.0,
            "city": "A",
            "merchant_category": "food",
        },
        {
            "transaction_id": "T2",
            "customer_id": "C1",
            "timestamp": "2026-09-01 10:05:00",
            "amount": 1000.0,
            "city": "B",
            "merchant_category": "electronics",
        },
    ])

    result = build_features(data).set_index("transaction_id")

    assert result.loc["T1", "prior_transaction_count"] == 0
    assert result.loc["T2", "historical_median_amount"] == 100.0
    assert result.loc["T2", "amount_vs_median"] == 10.0
    assert result.loc["T2", "new_city"] == 1
    assert result.loc["T2", "new_merchant_category"] == 1
    assert result.loc["T2", "transactions_previous_10m"] == 1


def test_input_is_not_modified():
    data = pd.DataFrame([{
        "transaction_id": "T1",
        "customer_id": "C1",
        "timestamp": "2026-09-01 10:00:00",
        "amount": 100.0,
        "city": "A",
        "merchant_category": "food",
    }])
    original = data.copy(deep=True)

    build_features(data)

    pd.testing.assert_frame_equal(data, original)
