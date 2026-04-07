"""
RUL prediction using sklearn Gradient Boosting.
Strong baseline without extra dependencies.
"""

from sklearn.ensemble import GradientBoostingRegressor
from rul.utils import load_data, print_metrics
from rul.visualize import generate_all_plots


def main():
    X_train, y_train, X_test, y_test, feature_cols, train_df, test_df = load_data()

    model = GradientBoostingRegressor(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        subsample=0.8, random_state=42,
    )
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    print_metrics("Gradient Boosting", y_train, y_pred_train, y_test, y_pred_test)
    print(f"\nModel: GradientBoostingRegressor (n_estimators=200, max_depth=5)")
    print(f"Features: {feature_cols}")

    generate_all_plots(model, X_train, y_train, X_test, y_test, train_df, test_df, feature_cols, "Gradient Boosting", "gradient_boosting")


if __name__ == "__main__":
    main()
