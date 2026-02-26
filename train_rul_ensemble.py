"""
RUL prediction using ensemble of multiple models.
Averages predictions from Random Forest, Gradient Boosting, and optionally XGBoost.
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from rul_utils import load_data, print_metrics

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


def main():
    X_train, y_train, X_test, y_test, feature_cols = load_data()

    estimators = [
        ("rf", RandomForestRegressor(n_estimators=100, random_state=42)),
        ("gb", GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)),
    ]
    if HAS_XGB:
        estimators.append(("xgb", xgb.XGBRegressor(n_estimators=100, max_depth=5, random_state=42)))

    model = VotingRegressor(estimators=estimators)
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    name = "Ensemble (RF + GB + XGB)" if HAS_XGB else "Ensemble (RF + GB)"
    print_metrics(name, y_train, y_pred_train, y_test, y_pred_test)
    print(f"\nModels: {[e[0] for e in estimators]}")
    print(f"Features: {feature_cols}")


if __name__ == "__main__":
    main()
