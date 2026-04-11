from __future__ import annotations

from pathlib import Path

import numpy as np
from xgboost import XGBRegressor


class XGBoostRULModel:
    def __init__(
        self,
        seed: int = 42,
        n_estimators: int = 500,
        learning_rate: float = 0.05,
        max_depth: int = 4,
        min_child_weight: float = 2.0,
        subsample: float = 0.9,
        colsample_bytree: float = 0.9,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
    ) -> None:
        self.seed = seed
        self.model = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_child_weight=min_child_weight,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            eval_metric="rmse",
            random_state=seed,
        )

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
    ) -> None:
        self.model.fit(
            x_train,
            y_train,
            eval_set=[(x_train, y_train), (x_val, y_val)],
            verbose=False,
        )

    def predict(self, x_values: np.ndarray) -> np.ndarray:
        return self.model.predict(x_values)

    def save(self, output_path: Path) -> None:
        self.model.save_model(output_path)

    @classmethod
    def load(cls, model_path: Path) -> "XGBoostRULModel":
        instance = cls()
        instance.model.load_model(model_path)
        return instance

    @property
    def feature_importances_(self) -> np.ndarray:
        return self.model.feature_importances_

    def evals_result(self) -> dict:
        return self.model.evals_result()
