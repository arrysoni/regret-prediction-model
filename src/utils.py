from typing import Iterable
import pandas as pd
import numpy as np


def ensure_dirs(*paths):
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def parse_timestamp(df: pd.DataFrame, col: str = "timestamp") -> pd.DataFrame:
    out = df.copy()
    out[col] = pd.to_datetime(out[col], utc=True)
    return out


def time_split_per_user(
    df,
    user_col="user_id",
    time_col="timestamp",
    train_frac=0.7,
    val_frac=0.15,
):
    parts_train, parts_val, parts_test = [], [], []

    for _, g in df.sort_values(time_col).groupby(user_col):
        n = len(g)
        if n < 5:
            parts_train.append(g)
            continue

        train_end = int(n * train_frac)
        val_end = int(n * (train_frac + val_frac))

        parts_train.append(g.iloc[:train_end])
        parts_val.append(g.iloc[train_end:val_end])
        parts_test.append(g.iloc[val_end:])

    return type(
        "Splits",
        (),
        {
            "train": pd.concat(parts_train),
            "val": pd.concat(parts_val) if parts_val else df.iloc[:0],
            "test": pd.concat(parts_test) if parts_test else df.iloc[:0],
        },
    )
