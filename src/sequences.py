
import pandas as pd


def detect_suspicious_sequences(
    transactions: pd.DataFrame,
    window_minutes: int = 10,
    min_transactions: int = 4,
    low_value_limit: float = 10.0,
) -> pd.DataFrame:
    """Group rapid, low-value transactions into investigation cases.

    Uses transaction attributes and timestamps only. Does not use scenario
    labels or fraud ground-truth labels.
    """
    required = {
        "transaction_id",
        "customer_id",
        "timestamp",
        "amount",
    }
    missing = required - set(transactions.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = transactions.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="raise")
    df["amount"] = pd.to_numeric(df["amount"], errors="raise")

    if df["timestamp"].isna().any():
        raise ValueError("Timestamps cannot be missing.")
    if df["amount"].isna().any() or (df["amount"] < 0).any():
        raise ValueError("Amounts must be present and non-negative.")
    if window_minutes <= 0 or min_transactions < 2:
        raise ValueError("Window must be positive and minimum count at least 2.")

    df = df.sort_values(
        ["customer_id", "timestamp", "transaction_id"],
        kind="stable",
    ).reset_index(drop=True)

    cases = []
    case_number = 1

    for customer_id, group in df.groupby("customer_id", sort=False):
        group = group.reset_index(drop=True)
        times = group["timestamp"]

        left = 0
        for right in range(len(group)):
            while left < right and (
                times.iloc[right] - times.iloc[left]
            ) > pd.Timedelta(minutes=window_minutes):
                left += 1

            window = group.iloc[left:right + 1]

            # Only flag a sequence when enough transactions in this window
            # are individually low-value. Labels are never consulted.
            low_value = window[window["amount"] <= low_value_limit]

            if len(low_value) >= min_transactions:
                cases.append({
                    "case_id": f"SEQ-{case_number:04d}",
                    "customer_id": customer_id,
                    "start_time": low_value["timestamp"].min(),
                    "end_time": low_value["timestamp"].max(),
                    "transaction_count": len(low_value),
                    "total_amount": round(float(low_value["amount"].sum()), 2),
                    "transaction_ids": low_value["transaction_id"].tolist(),
                    "pattern": "rapid_low_value_burst",
                    "explanation": (
                        f"{len(low_value)} low-value transactions "
                        f"totalling ₹{low_value['amount'].sum():.2f} "
                        f"occurred within {window_minutes} minutes."
                    ),
                })
                case_number += 1

                # Avoid reporting overlapping cases for the same burst.
                left = right + 1

    return pd.DataFrame(
        cases,
        columns=[
            "case_id",
            "customer_id",
            "start_time",
            "end_time",
            "transaction_count",
            "total_amount",
            "transaction_ids",
            "pattern",
            "explanation",
        ],
    )
