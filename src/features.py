import pandas as pd
import numpy as np
from src.config import EPS


def _ensure_sorted(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "timestamp" not in out.columns:
        raise ValueError("timestamp column missing")
    if not pd.api.types.is_datetime64_any_dtype(out["timestamp"]):
        raise ValueError(
            "timestamp must be datetime. Did you parse it in build_dataset.py?")

    out["hour"] = out["timestamp"].dt.hour
    out["day_of_week"] = out["timestamp"].dt.dayofweek
    out["is_weekend"] = out["day_of_week"].isin([5, 6]).astype(int)
    out["is_late_night"] = out["hour"].isin([0, 1, 2, 3, 4, 5]).astype(int)

    return out


def add_time_since_previous_transaction(df: pd.DataFrame) -> pd.DataFrame:
    out = _ensure_sorted(df.copy())

    if "user_id" not in out.columns:
        raise ValueError("user_id column missing")
    if "timestamp" not in out.columns:
        raise ValueError("timestamp column missing")

    out["minutes_since_prev_txn"] = (
        out.groupby("user_id")["timestamp"]
        .diff()
        .dt.total_seconds()
        .div(60.0)
    )

    # First transaction for each user has no previous
    out["minutes_since_prev_txn"] = out["minutes_since_prev_txn"].fillna(-1.0)

    return out


def add_user_spend_deviation_features(df: pd.DataFrame, eps: float = EPS) -> pd.DataFrame:
    out = _ensure_sorted(df.copy())

    if "user_id" not in out.columns:
        raise ValueError("user_id column missing")
    if "amount" not in out.columns:
        raise ValueError("amount column missing")

    out["amount"] = pd.to_numeric(out["amount"], errors="coerce")
    if out["amount"].isna().any():
        raise ValueError("amount has NaNs after numeric conversion")

    g = out.groupby("user_id")["amount"]

    # Past-only expanding stats: shift so the current transaction is not included
    out["user_amount_mean_past"] = g.transform(
        lambda s: s.expanding().mean().shift(1))
    out["user_amount_median_past"] = g.transform(
        lambda s: s.expanding().median().shift(1))
    out["user_amount_std_past"] = g.transform(
        lambda s: s.expanding().std(ddof=0).shift(1))

    # Global fallbacks for early history
    global_mean = float(out["amount"].mean())
    global_median = float(out["amount"].median())
    global_std = float(out["amount"].std(ddof=0))

    out["user_amount_mean_past"] = out["user_amount_mean_past"].fillna(
        global_mean)
    out["user_amount_median_past"] = out["user_amount_median_past"].fillna(
        global_median)
    out["user_amount_std_past"] = out["user_amount_std_past"].fillna(
        global_std)

    # Fix tiny / zero std to avoid exploding z-scores
    tiny_std_mask = out["user_amount_std_past"] < 1e-3
    out.loc[tiny_std_mask, "user_amount_std_past"] = global_std

    out["amount_z_user"] = (
        out["amount"] - out["user_amount_mean_past"]) / (out["user_amount_std_past"] + eps)
    out["amount_z_user"] = out["amount_z_user"].clip(-10.0, 10.0)

    out["amount_ratio_user_median"] = out["amount"] / \
        (out["user_amount_median_past"] + eps)

    return out


def add_behavioral_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "intent" not in out.columns:
        raise ValueError("intent column missing")
    if "stress" not in out.columns:
        raise ValueError("stress column missing")
    if "mood" not in out.columns:
        raise ValueError("mood column missing")
    if "category" not in out.columns:
        raise ValueError("category column missing")

    out["is_want"] = (out["intent"] == "want").astype(int)
    out["is_high_stress"] = (out["stress"] == "high").astype(int)
    out["is_low_mood"] = (out["mood"] == "low").astype(int)

    out["is_impulse_category"] = out["category"].isin(
        ["electronics", "fashion", "travel"]
    ).astype(int)

    return out


def add_interaction_features(df: pd.DataFrame, eps: float = EPS) -> pd.DataFrame:
    out = df.copy()

    if "amount" not in out.columns:
        raise ValueError("amount column missing")

    out["log_amount"] = np.log(out["amount"].astype(float) + eps)

    if "is_want" in out.columns and "is_high_stress" in out.columns:
        out["want_x_stress"] = (
            out["is_want"] * out["is_high_stress"]).astype(int)

    if "is_late_night" in out.columns and "is_impulse_category" in out.columns:
        out["late_x_impulse"] = (
            out["is_late_night"] * out["is_impulse_category"]).astype(int)

    return out
