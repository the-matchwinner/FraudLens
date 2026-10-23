
import pandas as pd

from src.features import build_features
from src.scoring import score_transaction


DATA_PATH = "data/transactions.csv"


def metrics(y_true, y_pred):
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())

    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
    }


def main():
    df = pd.read_csv(DATA_PATH)
    features = build_features(df)

    scored = pd.DataFrame([
        score_transaction(row)
        for row in features.to_dict(orient="records")
    ])

    evaluated = features.merge(
        scored[["transaction_id", "risk_score"]],
        on="transaction_id",
        validate="one_to_one",
    )

    y = evaluated["is_fraud"]

    # A deliberately simple, fixed-rule baseline.
    baseline_pred = (
        (evaluated["amount"] >= 10000)
        | (evaluated["transactions_previous_10m"] >= 4)
    ).astype(int)

    # FraudLens at its current review threshold.
    fraudlens_pred = (
        evaluated["risk_score"] >= 30
    ).astype(int)

    print("Baseline fixed rules:")
    print(metrics(y, baseline_pred))

    print("\nFraudLens (risk score >= 30):")
    print(metrics(y, fraudlens_pred))

   
    # Find the threshold whose recall is closest to the baseline.
    target_recall = metrics(y, baseline_pred)["recall"]
    candidates = []

    for threshold in sorted(evaluated["risk_score"].unique()):
        pred = (evaluated["risk_score"] >= threshold).astype(int)
        result = metrics(y, pred)
        candidates.append((threshold, result))

    print(f"\nBaseline target recall: {target_recall:.3f}")

    threshold, result = min(
        candidates,
        key=lambda item: (
            abs(item[1]["recall"] - target_recall),
            item[1]["FP"],
        ),
    )

    print(f"FraudLens closest-recall threshold: {threshold}")
    print(f"Recall difference: {result['recall'] - target_recall:+.3f}")
    print(result)


if __name__ == "__main__":
    main()

