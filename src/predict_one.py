import joblib
import pandas as pd

from src.config import FEATURE_DATASET_PATH, MODELS_DIR


def main():
    bundle = joblib.load(MODELS_DIR / "gradient_boosting.joblib")
    model = bundle["model"]
    features = bundle["features"]

    df = pd.read_parquet(FEATURE_DATASET_PATH)

    example = df.sample(1, random_state=42)

    X = example[features]

    prob = model.predict_proba(X)[0, 1]

    print("\n=== Prediction ===")
    print(f"Regret probability: {prob:.3f}")
    print(f"Predicted label: {'REGRET' if prob >= 0.5 else 'NO REGRET'}")

    print("\nTop feature values:")
    print(X.T)


if __name__ == "__main__":
    main()
