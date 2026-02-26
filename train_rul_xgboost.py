"""
RUL prediction using XGBoost.
Often outperforms Random Forest on tabular data.
"""

import numpy as np
from rul_utils import load_data, print_metrics
import xgboost as xgb


def main():
    X_train, y_train, X_test, y_test, feature_cols = load_data()

    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    print_metrics("XGBoost", y_train, y_pred_train, y_test, y_pred_test)
    print(f"\nModel: XGBoost (n_estimators=200, max_depth=6)")
    print(f"Features: {feature_cols}")


if __name__ == "__main__":
    main()
