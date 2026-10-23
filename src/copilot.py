
def explain_investigation(
    transaction: dict,
    ml_result: dict,
    sequence_cases: list | None = None,
) -> dict:
    """Generate a grounded, deterministic investigation summary."""

    evidence = transaction.get("evidence", [])
    if isinstance(evidence, str):
        evidence = [evidence] if evidence else []

    score = float(transaction.get("risk_score", 0))
    anomaly_score = float(ml_result.get("ml_anomaly_score", 0))
    is_anomaly = bool(ml_result.get("ml_anomaly_flag", False))

    related_cases = sequence_cases or []

    reasons = list(evidence)

    if is_anomaly:
        reasons.append(
            f"ML model ranked this transaction at anomaly percentile "
            f"{anomaly_score:.1f}/100."
        )

    if related_cases:
        for case in related_cases:
            explanation = case.get("explanation")
            if explanation:
                reasons.append(explanation)

    if score >= 45 or is_anomaly or related_cases:
        recommendation = (
            "Review transaction history and verify the activity with "
            "the authorised account holder before making a decision."
        )
    else:
        recommendation = (
            "No strong signal was produced by the current checks. "
            "Continue routine monitoring; this is not proof of legitimacy."
        )

    return {
        "risk_score": score,
        "ml_anomaly_score": anomaly_score,
        "ml_anomaly_flag": is_anomaly,
        "evidence": reasons,
        "recommended_action": recommendation,
        "decision_note": (
            "Automated signals support investigation; they do not establish fraud."
        ),
    }
