"""
RUL prediction with hyperparameter tuning via GridSearchCV.
Searches over Random Forest parameters for better generalization.
"""

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from rul.utils import load_data, print_metrics
from rul.visualize import generate_all_plots


def main():
    X_train, y_train, X_test, y_test, feature_cols, train_df, test_df = load_data()

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [5, 10, 15, None],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
    }

    gs = GridSearchCV(
        RandomForestRegressor(random_state=42),
        param_grid, cv=5, scoring="neg_mean_absolute_error",
        n_jobs=-1, verbose=1,
    )
    gs.fit(X_train, y_train)

    print(f"\nBest params: {gs.best_params_}")

    model = gs.best_estimator_
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    print_metrics("Tuned Random Forest", y_train, y_pred_train, y_test, y_pred_test)
    print(f"\nBest CV MAE: {-gs.best_score_:.2f} cycles")
    print(f"Features: {feature_cols}")

    generate_all_plots(model, X_train, y_train, X_test, y_test, train_df, test_df, feature_cols, "Tuned RF", "tuned")


if __name__ == "__main__":
    main()
