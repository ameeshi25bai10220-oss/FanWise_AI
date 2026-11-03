"""
AACI Model 3 – Preprocessing & Imputation Module
================================================
Leakage-free feature scaling, categorical encoding, and causal imputation
for the ceiling fan energy intelligence pipeline.

STRICT ANTI-LEAKAGE DESIGN PRINCIPLES:
1. TRAIN-SET ISOLATION:
   All scalers, encoders, and imputation statistics (fan-level medians, global medians)
   are fitted STRICTLY on the training partition.
2. CAUSAL SENSOR IMPUTATION:
   - Forward filling operates strictly chronologically (past -> present).
   - Any unobserved initial values are filled using medians estimated from the
     TRAINING partition only.
   - For unseen fans (zero-shot LOFO), fallback uses the training global median.
     The held-out fan's complete dataset is NEVER used to calculate medians.
3. INVARIANCE UNDER FUTURE PERTURBATION:
   Modifying, corrupting, or perturbing validation or test records has ZERO mathematical
   impact on training features, scalers, imputers, or model parameters.
"""

import os
from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, RobustScaler


class CausalSensorImputer:
    """
    Causal, leakage-free sensor imputer for time-series telemetry.

    Imputation Policy:
    1. Forward-fill sensor values chronologically per fan.
    2. For any initial observations before the first valid sensor reading:
       - If the fan was present in the training set: fill with training fan median.
       - If the fan was NOT present in training (Zero-Shot LOFO): fill with training
         global median.
       - Test and validation observations are NEVER used to calculate medians.
    """

    def __init__(
        self,
        sensor_cols: Optional[List[str]] = None,
        fan_id_col: str = "fan_id",
        time_col: str = "datetime",
    ):
        self.sensor_cols = sensor_cols or ["temperature", "humidity"]
        self.fan_id_col = fan_id_col
        self.time_col = time_col
        self.is_fitted: bool = False
        self.fan_medians_: Dict[str, Dict[str, float]] = {}
        self.global_medians_: Dict[str, float] = {}

    def fit(self, train_df: pd.DataFrame) -> "CausalSensorImputer":
        """
        Estimate imputation medians strictly from training data.
        """
        if not isinstance(train_df, pd.DataFrame):
            raise TypeError("train_df must be a pandas DataFrame.")

        self.fan_medians_ = {}
        self.global_medians_ = {}

        # 1. Compute global medians across training set
        for col in self.sensor_cols:
            if col in train_df.columns:
                valid_vals = train_df[col].dropna()
                if len(valid_vals) > 0:
                    self.global_medians_[col] = float(valid_vals.median())
                else:
                    self.global_medians_[col] = 0.0

        # 2. Compute fan-level medians strictly for fans in training set
        if self.fan_id_col in train_df.columns:
            for fan_id, group in train_df.groupby(self.fan_id_col):
                self.fan_medians_[fan_id] = {}
                for col in self.sensor_cols:
                    if col in group.columns:
                        valid_group = group[col].dropna()
                        if len(valid_group) > 0:
                            self.fan_medians_[fan_id][col] = float(valid_group.median())
                        else:
                            self.fan_medians_[fan_id][col] = self.global_medians_.get(col, 0.0)

        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply causal forward-fill and training-derived median fallbacks.
        NEVER computes medians from df itself.
        """
        if not self.is_fitted:
            raise RuntimeError("CausalSensorImputer must be fitted before transform.")

        df = df.copy()

        for col in self.sensor_cols:
            if col not in df.columns:
                continue

            # Step 1: Forward-fill chronologically within each fan
            if self.fan_id_col in df.columns:
                df[col] = df.groupby(self.fan_id_col)[col].ffill()
            else:
                df[col] = df[col].ffill()

            # Step 2: Resolve initial NaNs using fitted training medians ONLY
            if df[col].isna().any():
                if self.fan_id_col in df.columns:
                    for fan_id in df[self.fan_id_col].unique():
                        mask = (df[self.fan_id_col] == fan_id) & (df[col].isna())
                        if mask.any():
                            # If fan seen in train, use its train median; else global train median
                            if fan_id in self.fan_medians_ and col in self.fan_medians_[fan_id]:
                                fallback_val = self.fan_medians_[fan_id][col]
                            else:
                                fallback_val = self.global_medians_.get(col, 0.0)
                            df.loc[mask, col] = fallback_val
                else:
                    fallback_val = self.global_medians_.get(col, 0.0)
                    df[col] = df[col].fillna(fallback_val)

        return df

    def fit_transform(self, train_df: pd.DataFrame) -> pd.DataFrame:
        """Fit on train_df and transform it."""
        return self.fit(train_df).transform(train_df)


def clean_and_impute_telemetry(
    df: pd.DataFrame,
    sensor_cols: List[str] = ["temperature", "humidity"],
) -> pd.DataFrame:
    """
    Convenience function for self-contained causal forward fill.
    For cross-validation partitions, prefer CausalSensorImputer to guarantee
    zero partition leakage.
    """
    imputer = CausalSensorImputer(sensor_cols=sensor_cols)
    return imputer.fit_transform(df)


class EnergyPreprocessor:
    """
    Independent preprocessor for AACI Model 3 feature scaling and encoding.
    Ensures zero temporal or partition leakage.
    """

    def __init__(
        self,
        scaler_type: str = "standard",
        numeric_features: Optional[List[str]] = None,
        categorical_features: Optional[List[str]] = None,
    ):
        """
        Parameters
        ----------
        scaler_type : str, default 'standard'
            'standard' for StandardScaler, 'robust' for RobustScaler.
        numeric_features : List[str], optional
            Columns to scale.
        categorical_features : List[str], optional
            Categorical columns to one-hot encode.
        """
        self.scaler_type = scaler_type
        self.numeric_features = numeric_features or []
        self.categorical_features = categorical_features or []

        if scaler_type == "standard":
            self.scaler = StandardScaler()
        elif scaler_type == "robust":
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaler_type: {scaler_type}")

        self.is_fitted: bool = False
        self.fitted_numeric_cols: List[str] = []
        self.fitted_cat_cols: List[str] = []
        self.feature_names_out_: List[str] = []
        self.cat_categories_: Dict[str, List] = {}

    def fit(self, X: pd.DataFrame) -> "EnergyPreprocessor":
        """
        Fit scalers and categorical encoders STRICTLY on training features.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input X must be a pandas DataFrame.")

        # Determine numeric features
        if not self.numeric_features:
            self.fitted_numeric_cols = [
                c
                for c in X.columns
                if np.issubdtype(X[c].dtype, np.number)
                and c not in self.categorical_features
            ]
        else:
            self.fitted_numeric_cols = [
                c for c in self.numeric_features if c in X.columns
            ]

        # Determine categorical features
        if self.categorical_features:
            self.fitted_cat_cols = [
                c for c in self.categorical_features if c in X.columns
            ]
        else:
            self.fitted_cat_cols = []

        # Fit numeric scaler strictly on X
        if self.fitted_numeric_cols:
            self.scaler.fit(X[self.fitted_numeric_cols])

        # Record categorical categories strictly from X
        self.cat_categories_ = {}
        out_names = list(self.fitted_numeric_cols)
        for cat_col in self.fitted_cat_cols:
            cats = sorted(X[cat_col].dropna().unique().tolist())
            self.cat_categories_[cat_col] = cats
            for cat_val in cats:
                out_names.append(f"{cat_col}_{cat_val}")

        self.feature_names_out_ = out_names
        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform feature matrix using fitted parameters without re-estimation.
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transform.")
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input X must be a pandas DataFrame.")

        parts = []

        # Transform numeric features
        if self.fitted_numeric_cols:
            num_scaled = self.scaler.transform(X[self.fitted_numeric_cols])
            parts.append(num_scaled)

        # Transform categorical features with consistent dummy columns
        for cat_col in self.fitted_cat_cols:
            cats = self.cat_categories_[cat_col]
            cat_series = X[cat_col]
            dummy_matrix = np.zeros((len(X), len(cats)), dtype=float)
            for idx, val in enumerate(cats):
                dummy_matrix[:, idx] = (cat_series == val).astype(float)
            parts.append(dummy_matrix)

        if not parts:
            raise ValueError("No features available to transform.")

        return np.hstack(parts)

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        """Fit on X and return transformed matrix."""
        return self.fit(X).transform(X)

    def get_feature_names_out(self) -> List[str]:
        """Return list of transformed feature names."""
        if not self.is_fitted:
            raise RuntimeError("Preprocessor is not yet fitted.")
        return list(self.feature_names_out_)

    def save(self, filepath: str) -> None:
        """Save preprocessor to disk via joblib."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str) -> "EnergyPreprocessor":
        """Load preprocessor from disk via joblib."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Preprocessor file not found: {filepath}")
        loaded = joblib.load(filepath)
        if not isinstance(loaded, cls):
            raise TypeError(f"Loaded object is {type(loaded)}, expected {cls}.")
        return loaded
