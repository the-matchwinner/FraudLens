
import pandas as pd

from src.features import build_features
from src.scoring import score_transaction


DATA_PATH = "data/transactions.csv"


def main():
    raw = pd.read_csv(DATA_PATH)
    features = build_features(raw)

    results = []
    for row in features.to_dict(orient="records"):
        scored = score_transaction(row)
        results.append({
            "transaction_id": row["transaction_id"],
            "scenario": row["scenario"],
            "is_fraud": row["is_fraud"],
            "amount": row["amount"],
            "amount_vs_median": row["amount_vs_median"],
            "new_city": row["new_city"],
            "new_merchant_category": row["new_merchant_category"],
            "transactions_previous_10m": row[
                "transactions_previous_10m"
            ],
            "is_unusual_hour": row["is_unusual_hour"],
            "risk_score": scored["risk_score"],
            "status": scored["status"],
            "evidence": " | ".join(
                item["signal"] for item in scored["evidence"]
            ) or "none",
        })

    report = pd.DataFrame(results)

    # Focus on legitimate unusual transactions.
    legitimate = report[
        report["scenario"] == "legitimate_unusual"
    ].copy()

    print(f"Legitimate unusual transactions: {len(legitimate)}")
    print(
        "\nRisk status distribution:\n",
        legitimate["status"].value_counts().to_string(),
    )

    print("\nEvidence signal counts:")
    print(
        legitimate["evidence"]
        .str.split(" | ")
        .explode()
        .value_counts()
        .to_string()
    )

    print("\nSample transactions:")
    print(
        legitimate[
            [
                "transaction_id",
                "amount",
                "amount_vs_median",
                "new_city",
                "new_merchant_category",
                "transactions_previous_10m",
                "is_unusual_hour",
                "risk_score",
                "evidence",
            ]
        ].head(15).to_string(index=False)
    )

    output_path = "data/legitimate_alerts.csv"
    legitimate.to_csv(output_path, index=False)
    print(f"\nSaved detailed report to: {output_path}")


if __name__ == "__main__":
    main()
