"""
Basic RUL prediction baseline for review.
Uses Random Forest on preprocessed train/test data.
"""

from sklearn.ensemble import RandomForestRegressor

from rul.utils import load_data, print_metrics
from rul.visualize import generate_all_plots


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
