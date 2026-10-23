
import pandas as pd


def build_features(transactions: pd.DataFrame) -> pd.DataFrame:
    """Build transaction features using only earlier transactions.

    The input must contain transaction_id, customer_id, timestamp,
    amount, city, and merchant_category.
    """
    required = {
        "transaction_id",
        "customer_id",
        "timestamp",
        "amount",
        "city",
        "merchant_category",
    }
    missing = required - set(transactions.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = transactions.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="raise")

    if df["timestamp"].isna().any():
        raise ValueError("Timestamps cannot be missing.")
    if df["amount"].isna().any() or (df["amount"] < 0).any():
        raise ValueError("Amounts must be present and non-negative.")

    df = df.sort_values(
        ["customer_id", "timestamp", "transaction_id"],
        kind="stable",
    ).reset_index(drop=True)

    # Prior transactions only: current transaction is excluded.
    grouped = df.groupby("customer_id", sort=False)

    df["prior_transaction_count"] = grouped.cumcount()

    # Historical median; use a neutral fallback until history exists.
    df["historical_median_amount"] = grouped["amount"].transform(
        lambda s: s.shift(1).expanding().median()
    )
    df["amount_vs_median"] = (
        df["amount"] / df["historical_median_amount"].replace(0, float("nan"))
    ).fillna(1.0)

    # Has the customer used this city/category in an earlier transaction?
    df["new_city"] = 0
    df["new_merchant_category"] = 0

    for _, indices in df.groupby("customer_id", sort=False).groups.items():
        seen_cities = set()
        seen_categories = set()

        for idx in indices:
            city = df.at[idx, "city"]
            category = df.at[idx, "merchant_category"]

            df.at[idx, "new_city"] = int(city not in seen_cities)
            df.at[idx, "new_merchant_category"] = int(
                category not in seen_categories
            )

            seen_cities.add(city)
            seen_categories.add(category)

    # Number of earlier transactions in the previous 10 minutes.
    # The current transaction is not included.
    velocity = []
    for _, group in df.groupby("customer_id", sort=False):
        times = group["timestamp"]
        counts = []
        left = 0

        for right, current_time in enumerate(times):
            while left < right and (
                current_time - times.iloc[left]
            ) > pd.Timedelta(minutes=10):
                left += 1

            counts.append(right - left)

        velocity.extend(zip(group.index, counts))

    df["transactions_previous_10m"] = 0
    for idx, count in velocity:
        df.at[idx, "transactions_previous_10m"] = count

    df["transaction_hour"] = df["timestamp"].dt.hour
    df["is_unusual_hour"] = (
        (df["transaction_hour"] < 6) | (df["transaction_hour"] >= 23)
    ).astype(int)

    return df
