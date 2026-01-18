import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.config import FEATURE_DATASET_PATH, MODELS_DIR, RANDOM_SEED
from src.utils import ensure_dirs, time_split_per_user


FEATURE_COLUMNS = [
    "hour",
    "day_of_week",
    "is_weekend",
    "is_late_night",
    "minutes_since_prev_txn",
    "amount_z_user",
    "amount_ratio_user_median",
]

TARGET_COLUMN = "regret"


def main():
    ensure_dirs(MODELS_DIR)

    df = pd.read_parquet(FEATURE_DATASET_PATH)

    # Split by time per user
    splits = time_split_per_user(df)

    X_train = splits.train[FEATURE_COLUMNS]
    y_train = splits.train[TARGET_COLUMN]

    X_val = splits.val[FEATURE_COLUMNS]
    y_val = splits.val[TARGET_COLUMN]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_SEED,
    )

    model.fit(X_train_scaled, y_train)

    val_probs = model.predict_proba(X_val_scaled)[:, 1]
    val_auc = roc_auc_score(y_val, val_probs)

    print(f"Validation ROC AUC: {val_auc:.4f}")

    joblib.dump(
        {"model": model, "scaler": scaler, "features": FEATURE_COLUMNS},
        MODELS_DIR / "logistic_regression.joblib",
    )


if __name__ == "__main__":
    main()
