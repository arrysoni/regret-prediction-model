import joblib
import numpy as np
import pandas as pd

from src.config import FEATURE_DATASET_PATH, MODELS_DIR


FEATURE_COLUMNS = [
    "hour",
    "day_of_week",
    "is_weekend",
    "is_late_night",
    "minutes_since_prev_txn",
    "amount_z_user",
    "amount_ratio_user_median",
]


def explain_linear_prediction(row, model, feature_names):
    coefs = model.coef_[0]
    contributions = row.values * coefs

    explanation = (
        pd.DataFrame(
            {
                "feature": feature_names,
                "value": row.values,
                "coefficient": coefs,
                "contribution": contributions,
            }
        )
        .assign(abs_contribution=lambda x: x["contribution"].abs())
        .sort_values("abs_contribution", ascending=False)
    )

    return explanation


def main():
    bundle = joblib.load(MODELS_DIR / "logistic_regression.joblib")
    model = bundle["model"]
    scaler = bundle["scaler"]

    df = pd.read_parquet(FEATURE_DATASET_PATH)

    # Pick a single transaction (you can change index later)
    example = df.sample(1, random_state=42)

    X = example[FEATURE_COLUMNS]
    X_scaled = scaler.transform(X)

    prob = model.predict_proba(X_scaled)[0, 1]

    explanation = explain_linear_prediction(
        pd.Series(X_scaled[0], index=FEATURE_COLUMNS),
        model,
        FEATURE_COLUMNS,
    )

    print("\n=== Prediction ===")
    print(f"Regret probability: {prob:.3f}")
    print(f"Predicted label: {'REGRET' if prob >= 0.5 else 'NO REGRET'}")

    print("\n=== Top contributing factors ===")
    print(explanation.head(5).to_string(index=False))

    print("\n=== Plain English explanation ===")
    for _, r in explanation.head(3).iterrows():
        direction = "increases" if r["contribution"] > 0 else "decreases"
        print(f"- {r['feature']} {direction} regret risk")


if __name__ == "__main__":
    main()
