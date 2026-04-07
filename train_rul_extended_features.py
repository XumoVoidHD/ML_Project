"""
RUL prediction with extended feature set.
Uses normalized capacity features and ambient temperature.
"""

from sklearn.ensemble import RandomForestRegressor
from rul_utils import load_data, print_metrics
from rul_visualize import generate_all_plots

EXTENDED_FEATURES = [
    "Capacity", "capacity_fade", "Capacity_normalized", "capacity_fade_normalized",
    "capacity_derivative", "Re", "Rct", "discharge_duration", "avg_temperature",
    "voltage_at_100s", "voltage_at_300s", "voltage_at_600s", "ambient_temperature",
    "discharge_cycle_index",
]


def main():
    X_train, y_train, X_test, y_test, feature_cols, train_df, test_df = load_data(EXTENDED_FEATURES)

    model = RandomForestRegressor(n_estimators=150, max_depth=12, random_state=42)
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    print_metrics("Extended Features (RF)", y_train, y_pred_train, y_test, y_pred_test)
    print(f"\nModel: Random Forest (n_estimators=150, max_depth=12)")
    print(f"Features ({len(feature_cols)}): {feature_cols}")

    generate_all_plots(model, X_train, y_train, X_test, y_test, train_df, test_df, feature_cols, "Extended Features", "extended_features")


if __name__ == "__main__":
    main()
