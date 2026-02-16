import joblib
import pandas as pd
import json
import matplotlib.pyplot as plt

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance

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
    "is_want",
    "is_high_stress",
    "is_low_mood",
    "is_impulse_category",
    "log_amount",
    "want_x_stress",
    "late_x_impulse",
]

TARGET_COLUMN = "regret"


def main():
    ensure_dirs(MODELS_DIR)

    df = pd.read_parquet(FEATURE_DATASET_PATH)
    splits = time_split_per_user(df)

    X_train = splits.train[FEATURE_COLUMNS]
    y_train = splits.train[TARGET_COLUMN]

    X_val = splits.val[FEATURE_COLUMNS]
    y_val = splits.val[TARGET_COLUMN]

    model = HistGradientBoostingClassifier(
        learning_rate=0.07,
        max_depth=5,
        max_iter=300,              # reduced from 500
        min_samples_leaf=30,
        l2_regularization=1.0,
        random_state=RANDOM_SEED,
    )

    model.fit(X_train, y_train)

    val_probs = model.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, val_probs)

    print(f"\nValidation ROC AUC: {val_auc:.4f}")

    # Save model
    joblib.dump(
        {"model": model, "features": FEATURE_COLUMNS},
        MODELS_DIR / "gradient_boosting.joblib",
    )

    # ---- PERMUTATION IMPORTANCE ----
    print("\nComputing permutation feature importance...")
    perm = permutation_importance(
        model,
        X_val,
        y_val,
        n_repeats=3,
        random_state=RANDOM_SEED,
        scoring="roc_auc",
    )

    importances = perm.importances_mean
    feat_imp = sorted(
        zip(FEATURE_COLUMNS, importances),
        key=lambda x: -x[1]
    )

    print("\nTop Feature Importances (Permutation):")
    for f, v in feat_imp[:8]:
        print(f"{f}: {v:.4f}")

    # ---- SAVE PLOT ----
    top_features = feat_imp[:8]
    features = [x[0] for x in top_features]
    values = [x[1] for x in top_features]

    plt.figure(figsize=(8, 5))
    plt.barh(features[::-1], values[::-1])
    plt.xlabel("Decrease in ROC-AUC")
    plt.title("Top Feature Importances (Permutation)")
    plt.tight_layout()
    plt.savefig(MODELS_DIR / "feature_importance.png")
    plt.close()

    # ---- SAVE METRICS ----
    metrics = {
        "validation_auc": round(float(val_auc), 4),
        "num_features": len(FEATURE_COLUMNS),
        "model_type": "HistGradientBoostingClassifier"
    }

    with open(MODELS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print("\nArtifacts saved:")
    print("  - gradient_boosting.joblib")
    print("  - feature_importance.png")
    print("  - metrics.json")


if __name__ == "__main__":
    main()
