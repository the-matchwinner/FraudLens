
from pathlib import Path

import pandas as pd

from src.features import build_features
from src.scoring import score_transaction


DATA_PATH = Path("data/transactions.csv")


def main():
    transactions = pd.read_csv(DATA_PATH)
    features = build_features(transactions)

    results = pd.DataFrame([
        score_transaction(row)
        for row in features.to_dict(orient="records")
    ])

    evaluated = features[
        ["transaction_id", "scenario", "is_fraud"]
    ].merge(
        results[["transaction_id", "risk_score", "status"]],
        on="transaction_id",
        validate="one_to_one",
    )

    print(f"Transactions evaluated: {len(evaluated):,}")
    print("\nRisk status by scenario:")
    print(pd.crosstab(
        evaluated["scenario"],
        evaluated["status"],
        margins=True,
    ))

    print("\nDetection results by scenario:")
    print(evaluated.groupby("scenario").agg(
        transactions=("transaction_id", "count"),
        actual_fraud=("is_fraud", "sum"),
        mean_risk_score=("risk_score", "mean"),
        high_risk_alerts=(
            "status",
            lambda s: (s == "high_risk_review").sum(),
        ),
        review_alerts=(
            "status",
            lambda s: (s != "low_risk").sum(),
        ),
    ).round(2))

    print("\nHighest-risk transactions:")
    print(evaluated.sort_values(
        "risk_score", ascending=False
    ).head(10).to_string(index=False))


if __name__ == "__main__":
    main()
