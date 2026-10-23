
import pandas as pd
from sklearn.ensemble import IsolationForest


ML_FEATURES = [
    "amount",
    "prior_transaction_count",
    "historical_median_amount",
    "amount_vs_median",
    "new_city",
    "new_merchant_category",
    "transactions_previous_10m",
    "transaction_hour",
    "is_unusual_hour",
]


def detect_anomalies(features: pd.DataFrame) -> pd.DataFrame:
    """Generate unsupervised anomaly signals, not fraud probabilities."""

    missing = set(ML_FEATURES + ["transaction_id"]) - set(features.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    X = features[ML_FEATURES].copy()

    for column in X.columns:
        X[column] = pd.to_numeric(X[column], errors="raise")

    # First transactions may not have historical baselines.
    X["historical_median_amount"] = X["historical_median_amount"].fillna(0)
    X["amount_vs_median"] = X["amount_vs_median"].fillna(1)

    if X.isna().any().any():
        raise ValueError("Unexpected missing values remain.")

    model = IsolationForest(
        n_estimators=150,
        contamination="auto",
        random_state=42,
    )

    model.fit(X)

    # Higher values indicate greater relative anomaly.
    raw_scores = -model.decision_function(X)

    anomaly_percentile = (
        pd.Series(raw_scores, index=features.index)
        .rank(pct=True)
        .mul(100)
        .round(1)
    )

    predictions = model.predict(X)

    return pd.DataFrame({
        "transaction_id": features["transaction_id"].values,
        "ml_anomaly_score": anomaly_percentile.values,
        "ml_anomaly_flag": predictions == -1,
    })
