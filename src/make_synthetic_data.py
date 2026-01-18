import numpy as np
import pandas as pd

from src.config import DATA_RAW_DIR, LABELS_CSV, RANDOM_SEED, TRANSACTIONS_CSV
from src.utils import ensure_dirs


def _pick(rng, items, p=None):
    return rng.choice(items, p=p)


def main():
    rng = np.random.default_rng(RANDOM_SEED)
    ensure_dirs(DATA_RAW_DIR)

    n_users = 120
    txns_per_user = rng.integers(30, 90, size=n_users)

    categories = ["groceries", "food", "electronics",
                  "fashion", "subscriptions", "health", "travel", "home"]
    merchants = {
        "groceries": ["Walmart", "Target", "TraderJoes"],
        "food": ["UberEats", "DoorDash", "LocalCafe"],
        "electronics": ["Amazon", "BestBuy", "Newegg"],
        "fashion": ["Zara", "H&M", "ASOS"],
        "subscriptions": ["Netflix", "Spotify", "YouTubePremium"],
        "health": ["CVS", "Walgreens", "GNC"],
        "travel": ["Expedia", "Delta", "Airbnb"],
        "home": ["Ikea", "HomeDepot", "Wayfair"],
    }

    moods = ["low", "neutral", "high"]
    stress_levels = ["low", "medium", "high"]
    intents = ["need", "want"]

    rows = []
    label_rows = []

    base_start = pd.Timestamp("2025-06-01", tz="UTC")

    for ui in range(n_users):
        user_id = f"U{ui:04d}"
        user_spend_scale = rng.lognormal(mean=3.0, sigma=0.35)
        user_regret_bias = rng.normal(0.0, 0.6)

        t = base_start + pd.Timedelta(days=int(rng.integers(0, 60)))

        for j in range(int(txns_per_user[ui])):
            gap_hours = float(rng.gamma(shape=2.0, scale=16.0))
            t = t + pd.Timedelta(hours=gap_hours)

            category = _pick(rng, categories, p=[
                             0.22, 0.18, 0.12, 0.12, 0.10, 0.10, 0.08, 0.08])
            merchant = _pick(rng, merchants[category])

            hour = int(t.hour)
            late_night = 1 if hour in [0, 1, 2, 3, 4, 5] else 0

            mood = _pick(rng, moods, p=[0.25, 0.55, 0.20])
            stress = _pick(rng, stress_levels, p=[0.30, 0.50, 0.20])
            intent = _pick(rng, intents, p=[0.55, 0.45])

            category_multiplier = {
                "groceries": 0.7,
                "food": 0.9,
                "electronics": 1.8,
                "fashion": 1.2,
                "subscriptions": 0.5,
                "health": 0.8,
                "travel": 2.2,
                "home": 1.4,
            }[category]

            amount = float(rng.lognormal(mean=np.log(
                user_spend_scale * category_multiplier), sigma=0.45))
            amount = max(2.0, min(amount, 2500.0))

            transaction_id = f"T{ui:04d}_{j:04d}"

            rows.append(
                {
                    "transaction_id": transaction_id,
                    "user_id": user_id,
                    "timestamp": t.isoformat(),
                    "amount": round(amount, 2),
                    "merchant": merchant,
                    "category": category,
                    "mood": mood,
                    "stress": stress,
                    "intent": intent,
                }
            )

            # Regret probability model (hidden truth)
            p = 0.10
            if intent == "want":
                p += 0.10
            if late_night:
                p += 0.06
            if category in ["electronics", "fashion", "travel"]:
                p += 0.05
            if stress == "high":
                p += 0.05
            if mood == "low":
                p += 0.04

            p += 0.03 * (np.log(amount + 1.0) - 5.0)

            # Keep p in a valid range before converting to log-odds
            p = float(np.clip(p, 1e-4, 1 - 1e-4))

            logit = np.log(p / (1 - p)) + user_regret_bias
            p = 1.0 / (1.0 + np.exp(-logit))

            regret = int(rng.random() < p)

            # Simulate missing labels
            if rng.random() < 0.08:
                continue

            label_rows.append(
                {"transaction_id": transaction_id, "regret": regret})

    transactions = pd.DataFrame(rows)
    labels = pd.DataFrame(label_rows)

    transactions.to_csv(TRANSACTIONS_CSV, index=False)
    labels.to_csv(LABELS_CSV, index=False)

    print("Wrote:")
    print(f"  {TRANSACTIONS_CSV}")
    print(f"  {LABELS_CSV}")
    print(f"Transactions: {len(transactions)} rows")
    print(f"Labels: {len(labels)} rows (some missing by design)")


if __name__ == "__main__":
    main()
