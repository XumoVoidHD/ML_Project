"""
Basic RUL prediction baseline for review.
Uses Random Forest on preprocessed train/test data.
"""

import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from rul_utils import load_data, print_metrics
from rul_visualize import generate_all_plots

DATA_DIR = Path(__file__).parent / "data" / "preprocessed_rul"
FEATURE_COLS = [
    "Capacity", "capacity_fade", "Re", "Rct", "discharge_duration",
    "avg_temperature", "voltage_at_100s", "voltage_at_300s", "voltage_at_600s",
    "discharge_cycle_index",
]


def main():
    X_train, y_train, X_test, y_test, feature_cols, train_df, test_df = load_data()

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    print_metrics("Baseline (Random Forest)", y_train, y_pred_train, y_test, y_pred_test)
    print(f"\nModel: Random Forest (100 trees)")
    print(f"Features: {feature_cols}")

    generate_all_plots(model, X_train, y_train, X_test, y_test, train_df, test_df, feature_cols, "Baseline", "baseline")


if __name__ == "__main__":
    main()
