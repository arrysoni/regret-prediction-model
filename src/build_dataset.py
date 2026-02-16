import pandas as pd

from src.config import DATA_PROCESSED_DIR, FEATURE_DATASET_PATH, LABELS_CSV, TRANSACTIONS_CSV
from src.schema import REQUIRED_LABEL_COLS, REQUIRED_TRANSACTION_COLS
from src.utils import ensure_dirs, parse_timestamp
from src.features import (
    add_temporal_features,
    add_time_since_previous_transaction,
    add_user_spend_deviation_features,
    add_behavioral_features,
    add_interaction_features
)


def _assert_columns(df: pd.DataFrame, required_cols, name: str):
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def main():
    ensure_dirs(DATA_PROCESSED_DIR)

    txns = pd.read_csv(TRANSACTIONS_CSV)
    labels = pd.read_csv(LABELS_CSV)

    _assert_columns(txns, REQUIRED_TRANSACTION_COLS, "transactions.csv")
    _assert_columns(labels, REQUIRED_LABEL_COLS, "labels.csv")

    txns = parse_timestamp(txns, "timestamp")
    labels["regret"] = labels["regret"].astype(int)

    merged = txns.merge(
        labels[["transaction_id", "regret"]], on="transaction_id", how="left")

    total = len(merged)
    labeled = int(merged["regret"].notna().sum())
    dropped = total - labeled

    merged = merged[merged["regret"].notna()].copy()
    merged["regret"] = merged["regret"].astype(int)

    merged = merged.sort_values(
        ["user_id", "timestamp"]).reset_index(drop=True)

    merged = add_temporal_features(merged)
    merged = add_time_since_previous_transaction(merged)
    merged = add_user_spend_deviation_features(merged)
    merged = add_behavioral_features(merged)
    merged = add_interaction_features(merged)

    merged.to_parquet(FEATURE_DATASET_PATH, index=False)

    regret_rate = float(merged["regret"].mean())

    print("Saved engineered feature dataset:")
    print(f"  {FEATURE_DATASET_PATH}")
    print(f"Rows total: {total}")
    print(f"Rows labeled: {labeled}")
    print(f"Rows dropped (unlabeled): {dropped}")
    print(f"Regret rate (positive class %): {regret_rate * 100:.2f}%")


if __name__ == "__main__":
    main()
