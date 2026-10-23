
from src.scoring import score_transaction



def test_high_amount_and_new_city_raise_risk():
    result = score_transaction({
        "transaction_id": "T1",
        "amount_vs_median": 10,
        "new_city": 1,
        "new_merchant_category": 0,
        "transactions_previous_10m": 0,
        "is_unusual_hour": 0,
        "city": "CITY_NEW",
    })

    assert result["risk_score"] == 35
    assert result["status"] == "review_recommended"
    assert len(result["evidence"]) == 2
    assert result["is_fraud_confirmed"] is False





def test_transaction_burst_triggers_high_risk_review():
    result = score_transaction({
        "transaction_id": "T2",
        "amount": 100,
        "amount_vs_median": 1,
        "new_city": 0,
        "new_merchant_category": 0,
        "transactions_previous_10m": 4,
        "is_unusual_hour": 0,
    })

    # A burst without low-value transactions receives the burst score only.
    assert result["risk_score"] == 30
    assert result["status"] == "review_recommended"



def test_same_input_produces_same_score():
    tx = {
        "transaction_id": "T3",
        "amount_vs_median": 10,
        "new_city": 1,
        "new_merchant_category": 1,
        "transactions_previous_10m": 4,
        "is_unusual_hour": 1,
    }

    first = score_transaction(tx)
    second = score_transaction(tx)

    assert first == second


def test_low_risk_transaction_has_no_evidence():
    result = score_transaction({
        "transaction_id": "T4",
        "amount_vs_median": 1,
        "new_city": 0,
        "new_merchant_category": 0,
        "transactions_previous_10m": 0,
        "is_unusual_hour": 0,
    })

    assert result["risk_score"] == 0
    assert result["status"] == "low_risk"
    assert result["evidence"] == []
