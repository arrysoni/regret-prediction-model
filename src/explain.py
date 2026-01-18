import joblib
import numpy as np
import pandas as pd

from src.config import MODELS_DIR


def main():
    bundle = joblib.load(MODELS_DIR / "logistic_regression.joblib")

    model = bundle["model"]
    features = bundle["features"]

    coefs = model.coef_[0]

    explanation = pd.DataFrame(
        {
            "feature": features,
            "coefficient": coefs,
            "effect": np.where(coefs > 0, "increases regret risk", "decreases regret risk"),
            "abs_strength": np.abs(coefs),
        }
    ).sort_values("abs_strength", ascending=False)

    print("\nModel explanation (log-odds scale):\n")
    print(explanation.to_string(index=False))


if __name__ == "__main__":
    main()
