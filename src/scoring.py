
from typing import Any


def score_transaction(transaction: dict[str, Any]) -> dict[str, Any]:
    """Return a transparent risk heuristic with auditable evidence.

    This is a prototype score, not a calibrated fraud probability.
    """
    tx = dict(transaction)
    evidence = []

    ratio = float(tx.get("amount_vs_median", 1.0))
    amount = float(tx.get("amount", 0))
    velocity = int(tx.get("transactions_previous_10m", 0))
    new_city = int(tx.get("new_city", 0)) == 1
    new_category = int(tx.get("new_merchant_category", 0)) == 1
    unusual_hour = int(tx.get("is_unusual_hour", 0)) == 1

    # Group 1: behavioural deviation.
    # Amount and novelty alone should not automatically produce high risk.
    deviation_score = 0

    if ratio >= 8:
        deviation_score += 25
        evidence.append({
            "signal": "amount_deviation",
            "points": 25,
            "observed": round(ratio, 2),
            "explanation": (
                f"Amount is {ratio:.1f}x the customer's historical median."
            ),
        })
    elif ratio >= 4:
        deviation_score += 15
        evidence.append({
            "signal": "amount_deviation",
            "points": 15,
            "observed": round(ratio, 2),
            "explanation": (
                f"Amount is {ratio:.1f}x the customer's historical median."
            ),
        })

    if new_city:
        deviation_score += 10
        evidence.append({
            "signal": "new_city",
            "points": 10,
            "observed": tx.get("city", "unknown"),
            "explanation": "The city is new in the available history.",
        })

    if new_category:
        deviation_score += 5
        evidence.append({
            "signal": "new_merchant_category",
            "points": 5,
            "observed": tx.get("merchant_category", "unknown"),
            "explanation": (
                "The merchant category is new in the available history."
            ),
        })

    # Group 2: transaction-pattern evidence.
    # Repeated low-value transactions are more informative in combination.
    pattern_score = 0

    if amount <= 10 and velocity >= 3:
        pattern_score += 25
        evidence.append({
            "signal": "low_value_burst",
            "points": 25,
            "observed": {
                "amount": amount,
                "prior_transactions_10m": velocity,
            },
            "explanation": (
                f"Low-value transaction of {amount:g} followed "
                f"{velocity} earlier transactions within 10 minutes."
            ),
        })

    if velocity >= 4:
        pattern_score += 30
        evidence.append({
            "signal": "transaction_burst",
            "points": 30,
            "observed": velocity,
            "explanation": (
                f"{velocity} earlier transactions occurred in the previous 10 minutes."
            ),
        })
    elif velocity >= 2:
        pattern_score += 15
        evidence.append({
            "signal": "transaction_burst",
            "points": 15,
            "observed": velocity,
            "explanation": (
                f"{velocity} earlier transactions occurred in the previous 10 minutes."
            ),
        })

    # Timing is supporting evidence, not proof of fraud.
    if unusual_hour:
        deviation_score += 5
        evidence.append({
            "signal": "unusual_hour",
            "points": 5,
            "observed": tx.get("transaction_hour"),
            "explanation": "The transaction occurred during a flagged hour.",
        })

    # Keep the two groups inspectable.
    score = min(deviation_score + pattern_score, 100)

    # Simulated context: verified customer confirmation mitigates risk.
    # This is synthetic demo data, not proof of real-world safety.
    if int(tx.get("customer_confirmed", 0)) == 1:
        score = max(0, score - 20)
        evidence.append({
            "signal": "customer_confirmed",
            "points": -20,
            "observed": True,
            "explanation": (
                "Simulated customer confirmation reduced the review score."
            ),
        })

    if score >= 45:
        status = "high_risk_review"
        recommendation = "Urgent review recommended"
    elif score >= 30:
        status = "review_recommended"
        recommendation = "Review recommended"
    else:
        status = "low_risk"
        recommendation = "No immediate review trigger"

    return {
        "transaction_id": tx.get("transaction_id"),
        "risk_score": score,
        "status": status,
        "recommendation": recommendation,
        "evidence": evidence,
        "is_fraud_confirmed": False,
    }
