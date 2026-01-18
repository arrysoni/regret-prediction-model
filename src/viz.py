import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay

from src.config import FEATURE_DATASET_PATH, MODELS_DIR, REPORTS_DIR
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
    ensure_dirs(REPORTS_DIR)

    bundle = joblib.load(MODELS_DIR / "logistic_regression.joblib")
    model = bundle["model"]
    scaler = bundle["scaler"]
    features = bundle["features"]

    df = pd.read_parquet(FEATURE_DATASET_PATH)
    splits = time_split_per_user(df)

    X_test = splits.test[FEATURE_COLUMNS]
    y_test = splits.test[TARGET_COLUMN].astype(int).values

    X_test_scaled = scaler.transform(X_test)
    probs = model.predict_proba(X_test_scaled)[:, 1]

    # 1) Probability distribution
    plt.figure()
    plt.hist(probs, bins=30)
    plt.title("Predicted regret probability distribution (test set)")
    plt.xlabel("Predicted probability")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "probability_distribution.png", dpi=160)
    plt.close()

    # 2) Coefficient importance (global explainability)
    coefs = model.coef_[0]
    order = np.argsort(np.abs(coefs))[::-1]

    plt.figure()
    plt.bar([features[i] for i in order], np.abs(coefs[order]))
    plt.title("Global importance (abs logistic regression coefficients)")
    plt.xlabel("Feature")
    plt.ylabel("Abs coefficient")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "coefficients_importance.png", dpi=160)
    plt.close()

    # 3) ROC curve
    plt.figure()
    RocCurveDisplay.from_predictions(y_test, probs)
    plt.title("ROC curve (test set)")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "roc_curve.png", dpi=160)
    plt.close()

    # 4) Precision-Recall curve (important for imbalance)
    plt.figure()
    PrecisionRecallDisplay.from_predictions(y_test, probs)
    plt.title("Precision-Recall curve (test set)")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "pr_curve.png", dpi=160)
    plt.close()

    print("Saved plots to:")
    print(f"  {REPORTS_DIR}")


if __name__ == "__main__":
    main()
