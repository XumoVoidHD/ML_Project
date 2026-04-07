"""Shared utilities for RUL training scripts."""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_DIR = Path(__file__).parent / "data" / "preprocessed_rul"

FEATURE_COLS = [
    "Capacity",
    "capacity_fade",
    "Re",
    "Rct",
    "discharge_duration",
    "avg_temperature",
    "voltage_at_100s",
    "voltage_at_300s",
    "voltage_at_600s",
    "discharge_cycle_index",
]


def load_data(feature_cols=None):
    """Load train/test data. Returns X_train, y_train, X_test, y_test, feature_cols, train_df, test_df."""
    train = pd.read_csv(DATA_DIR / "rul_train.csv")
    test = pd.read_csv(DATA_DIR / "rul_test.csv")
    base = feature_cols if feature_cols is not None else FEATURE_COLS
    cols = [c for c in base if c in train.columns]
    X_train = train[cols].fillna(0)
    y_train = train["RUL"]
    X_test = test[cols].fillna(0)
    y_test = test["RUL"]
    return X_train, y_train, X_test, y_test, cols, train, test


def print_metrics(name, y_train, y_pred_train, y_test, y_pred_test):
    """Print train/test metrics."""
    print("=" * 55)
    print(f"RUL Prediction - {name}")
    print("=" * 55)
    print()
    print("Train set:")
    print(f"  MAE:  {mean_absolute_error(y_train, y_pred_train):.2f} cycles")
    print(f"  RMSE: {np.sqrt(mean_squared_error(y_train, y_pred_train)):.2f} cycles")
    print(f"  R2:   {r2_score(y_train, y_pred_train):.4f}")
    print()
    print("Test set (B0018):")
    print(f"  MAE:  {mean_absolute_error(y_test, y_pred_test):.2f} cycles")
    print(f"  RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_test)):.2f} cycles")
    print(f"  R2:   {r2_score(y_test, y_pred_test):.4f}")
    print("=" * 55)
