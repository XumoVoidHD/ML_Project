"""
Basic RUL prediction baseline for review.
Uses Random Forest on preprocessed train/test data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_DIR = Path(__file__).parent / "data" / "preprocessed_rul"

# Features to use
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


def main():
    train = pd.read_csv(DATA_DIR / "rul_train.csv")
    test = pd.read_csv(DATA_DIR / "rul_test.csv")

    # Filter to available features
    feature_cols = [c for c in FEATURE_COLS if c in train.columns]
    X_train = train[feature_cols].fillna(0)
    y_train = train["RUL"]
    X_test = test[feature_cols].fillna(0)
    y_test = test["RUL"]

    # Train
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predict
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Metrics
    print("=" * 50)
    print("RUL Prediction - Baseline Results")
    print("=" * 50)
    print("Model: Random Forest (100 trees)")
    print(f"Features: {feature_cols}")
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
    print("=" * 50)


if __name__ == "__main__":
    main()
