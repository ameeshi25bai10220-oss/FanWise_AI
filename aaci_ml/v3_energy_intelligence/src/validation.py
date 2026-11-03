"""
AACI Model 3 – Validation & Splitting Module
============================================
Leakage-free temporal and cross-fan spatial splitting strategies for Model 3.

Key Responsibilities:
1. Chronological 70/15/15 train/val/test splitting per fan without time overlaps.
2. Leave-One-Fan-Out (LOFO) cross-validation for zero-shot deployment testing.
3. Few-shot fan adaptation split (training on 8 fans + first N days of target fan).
4. Automated temporal and feature leakage audit assertions.
"""

from typing import Dict, Generator, List, Optional, Tuple
import numpy as np
import pandas as pd


def temporal_train_val_test_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    per_fan: bool = True,
    time_col: str = "datetime",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataset strictly chronologically into train, validation, and test partitions.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset sorted or sortable chronologically.
    train_ratio : float, default 0.70
    val_ratio : float, default 0.15
    test_ratio : float, default 0.15
    per_fan : bool, default True
        If True, applies 70/15/15 split within each fan_id and concatenates.
        If False, applies 70/15/15 split globally across the chronological timeline.
    time_col : str, default 'datetime'

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (train_df, val_df, test_df)
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-5:
        raise ValueError("Ratios must sum to 1.0.")

    df = df.copy()
    if not np.issubdtype(df[time_col].dtype, np.datetime64):
        df[time_col] = pd.to_datetime(df[time_col], dayfirst=True)

    if per_fan and "fan_id" in df.columns:
        train_dfs, val_dfs, test_dfs = [], [], []
        for fan_id, group in df.groupby("fan_id"):
            group = group.sort_values(by=time_col).reset_index(drop=True)
            n = len(group)
            n_train = int(n * train_ratio)
            n_val = int(n * val_ratio)

            train_part = group.iloc[:n_train].copy()
            val_part = group.iloc[n_train : n_train + n_val].copy()
            test_part = group.iloc[n_train + n_val :].copy()

            # Verify no temporal overlap within this fan
            validate_no_temporal_leakage(train_part, val_part, test_part, time_col=time_col)

            train_dfs.append(train_part)
            val_dfs.append(val_part)
            test_dfs.append(test_part)

        train_df = pd.concat(train_dfs, ignore_index=True)
        val_df = pd.concat(val_dfs, ignore_index=True)
        test_df = pd.concat(test_dfs, ignore_index=True)
        return train_df, val_df, test_df
    else:
        df = df.sort_values(by=time_col).reset_index(drop=True)
        n = len(df)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train_df = df.iloc[:n_train].copy()
        val_df = df.iloc[n_train : n_train + n_val].copy()
        test_df = df.iloc[n_train + n_val :].copy()

        validate_no_temporal_leakage(train_df, val_df, test_df, time_col=time_col)
        return train_df, val_df, test_df


def leave_one_fan_out_split(
    df: pd.DataFrame,
    test_fan_id: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Partition dataset for Leave-One-Fan-Out (LOFO) evaluation.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing 'fan_id'.
    test_fan_id : str
        The held-out fan to serve as the zero-shot unseen test domain.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (train_df_8_fans, test_df_1_fan)
    """
    if "fan_id" not in df.columns:
        raise KeyError("Dataset must contain 'fan_id' column for LOFO split.")

    unique_fans = df["fan_id"].unique()
    if test_fan_id not in unique_fans:
        raise ValueError(f"Fan '{test_fan_id}' not found in dataset. Available: {list(unique_fans)}")

    train_df = df[df["fan_id"] != test_fan_id].copy().reset_index(drop=True)
    test_df = df[df["fan_id"] == test_fan_id].copy().reset_index(drop=True)

    # Assert mutual exclusivity of fan identities
    assert set(train_df["fan_id"].unique()).isdisjoint(set(test_df["fan_id"].unique()))

    return train_df, test_df


def generate_all_lofo_splits(
    df: pd.DataFrame,
) -> Generator[Tuple[str, pd.DataFrame, pd.DataFrame], None, None]:
    """
    Yield all LOFO folds across all unique fans in the dataset.
    """
    unique_fans = sorted(df["fan_id"].unique())
    for test_fan in unique_fans:
        train_df, test_df = leave_one_fan_out_split(df, test_fan)
        yield test_fan, train_df, test_df


def few_shot_fan_split(
    df: pd.DataFrame,
    target_fan_id: str,
    adaptation_days: int = 7,
    time_col: str = "datetime",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Partition dataset for few-shot adaptation benchmarking.

    Structure:
    - Train base: All other 8 fans (spatial source).
    - Adaptation subset: First `adaptation_days` days of target fan.
    - Test evaluation: Remaining days of target fan (strictly chronologically subsequent).

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (train_other_fans_df, target_adaptation_df, target_test_df)
    """
    if "fan_id" not in df.columns:
        raise KeyError("Dataset must contain 'fan_id' column.")

    df = df.copy()
    if not np.issubdtype(df[time_col].dtype, np.datetime64):
        df[time_col] = pd.to_datetime(df[time_col], dayfirst=True)

    other_fans_df = df[df["fan_id"] != target_fan_id].copy().reset_index(drop=True)
    target_fan_df = df[df["fan_id"] == target_fan_id].sort_values(by=time_col).reset_index(drop=True)

    if len(target_fan_df) == 0:
        raise ValueError(f"Target fan '{target_fan_id}' has no records.")

    min_date = target_fan_df[time_col].min()
    cutoff_date = min_date + pd.Timedelta(days=adaptation_days)

    adaptation_df = target_fan_df[target_fan_df[time_col] <= cutoff_date].copy().reset_index(drop=True)
    target_test_df = target_fan_df[target_fan_df[time_col] > cutoff_date].copy().reset_index(drop=True)

    if len(target_test_df) == 0:
        raise ValueError(
            f"Adaptation period ({adaptation_days} days) exhausts all data for fan '{target_fan_id}'."
        )

    # Assert strict chronological boundary
    assert adaptation_df[time_col].max() < target_test_df[time_col].min(), (
        "Temporal overlap between adaptation and test subsets."
    )

    return other_fans_df, adaptation_df, target_test_df


def validate_no_temporal_leakage(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    time_col: str = "datetime",
) -> None:
    """
    Assert that train timestamps strictly precede val, and val strictly precedes test.
    """
    if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
        return

    train_max = train_df[time_col].max()
    val_min = val_df[time_col].min()
    val_max = val_df[time_col].max()
    test_min = test_df[time_col].min()

    if train_max > val_min:
        raise AssertionError(
            f"TEMPORAL LEAKAGE: train_max ({train_max}) > val_min ({val_min})."
        )
    if val_max > test_min:
        raise AssertionError(
            f"TEMPORAL LEAKAGE: val_max ({val_max}) > test_min ({test_min})."
        )


def validate_no_feature_leakage(
    feature_cols: List[str],
    prohibited_substrings: Optional[List[str]] = None,
) -> None:
    """
    Verify that no contemporaneous target or forbidden telemetry columns
    are present in the feature set.
    """
    prohibited = prohibited_substrings or [
        "opTime",
        "target",
    ]
    for col in feature_cols:
        for p in prohibited:
            if p == col or (p == "target" and col.startswith("target_")):
                raise AssertionError(
                    f"FEATURE LEAKAGE: Feature column '{col}' matches prohibited pattern '{p}'."
                )

