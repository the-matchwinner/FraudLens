
from pathlib import Path

import numpy as np
import pandas as pd


SEED = 42
OUTPUT_PATH = Path("data/transactions.csv")


def generate_transactions(n_customers: int = 100, days: int = 30) -> pd.DataFrame:
    """Generate reproducible synthetic transactions with known scenario labels."""
    rng = np.random.default_rng(SEED)
    rows = []
    start = pd.Timestamp("2026-09-01")

    for customer_num in range(n_customers):
        customer_id = f"C{customer_num + 1:04d}"
        home_city = f"CITY_{customer_num % 10 + 1}"
        usual_amount = float(rng.choice([300, 600, 900, 1500, 2500]))

        count = int(rng.integers(20, 45))
        for _ in range(count):
            amount = round(float(rng.lognormal(
                mean=np.log(usual_amount), sigma=0.45
            )), 2)
            timestamp = start + pd.Timedelta(
                days=int(rng.integers(0, days)),
                minutes=int(rng.integers(8 * 60, 22 * 60)),
            )

            rows.append({
                "customer_id": customer_id,
                "timestamp": timestamp,
                "amount": amount,
                "city": home_city,
                "merchant_category": str(
                    rng.choice(["grocery", "food", "fuel", "shopping"])
                ),
                "scenario": "normal",
                "is_fraud": 0,
            })

    # Add clearly labelled, controlled test cases.
    # These are synthetic scenarios, not evidence of real-world accuracy.
    for i in range(max(10, n_customers // 2)):
        customer_id = f"C{i + 1:04d}"
        base_time = start + pd.Timedelta(days=days - 1, hours=23)

        # Account-takeover-like: large purchase from an unfamiliar city.
        rows.append({
            "customer_id": customer_id,
            "timestamp": base_time,
            "amount": 48000.0,
            "city": "UNFAMILIAR_CITY",
            "merchant_category": "electronics",
            "scenario": "account_takeover",
            "is_fraud": 1,
        })

        # Card-testing-like: repeated small transactions in a short burst.
        for j in range(4):
            rows.append({
                "customer_id": customer_id,
                "timestamp": base_time + pd.Timedelta(minutes=j),
                "amount": float(rng.choice([1, 2, 5, 9])),
                "city": "CITY_1",
                "merchant_category": "online",
                "scenario": "card_testing",
                "is_fraud": 1,
            })

        # Legitimate but unusual: high-value festival purchase.
        rows.append({
            "customer_id": customer_id,
            "timestamp": base_time - pd.Timedelta(hours=2),
            "amount": 18000.0,
            "city": "CITY_1",
            "merchant_category": "jewellery",
            "scenario": "legitimate_unusual",
            "is_fraud": 0,
        })

    df = pd.DataFrame(rows)
    df["customer_confirmed"] = (
    df["scenario"] == "legitimate_unusual"
).astype(int)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df.insert(0, "transaction_id", [f"T{i + 1:06d}" for i in range(len(df))])
    return df


if __name__ == "__main__":
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    transactions = generate_transactions()
    transactions.to_csv(OUTPUT_PATH, index=False)

    print(f"Generated {len(transactions):,} transactions")
    print("\nScenario counts:")
    print(transactions["scenario"].value_counts())
    print(f"\nSaved to: {OUTPUT_PATH}")
