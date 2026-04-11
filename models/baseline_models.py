from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge


class LinearRULModel:
    def __init__(self) -> None:
        self.model = LinearRegression()

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        self.model.fit(x_train, y_train)

    def predict(self, x_values: np.ndarray) -> np.ndarray:
        return self.model.predict(x_values)

    def save(self, output_path: Path) -> None:
        with output_path.open("wb") as file:
            pickle.dump(self.model, file)

    @classmethod
    def load(cls, model_path: Path) -> "LinearRULModel":
        instance = cls()
        with model_path.open("rb") as file:
            instance.model = pickle.load(file)
        return instance

    @property
    def feature_importances_(self) -> np.ndarray:
        return np.abs(self.model.coef_)


class RidgeRULModel:
    def __init__(self, alpha: float = 1.0) -> None:
        self.model = Ridge(alpha=alpha)

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        self.model.fit(x_train, y_train)

    def predict(self, x_values: np.ndarray) -> np.ndarray:
        return self.model.predict(x_values)

    def save(self, output_path: Path) -> None:
        with output_path.open("wb") as file:
            pickle.dump(self.model, file)

    @classmethod
    def load(cls, model_path: Path) -> "RidgeRULModel":
        instance = cls()
        with model_path.open("rb") as file:
            instance.model = pickle.load(file)
        return instance

    @property
    def feature_importances_(self) -> np.ndarray:
        return np.abs(self.model.coef_)


class RandomForestRULModel:
    def __init__(
        self,
        seed: int = 42,
        n_estimators: int = 300,
        max_depth: int | None = None,
        min_samples_leaf: int = 1,
    ) -> None:
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=seed,
            n_jobs=1,
        )

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        self.model.fit(x_train, y_train)

    def predict(self, x_values: np.ndarray) -> np.ndarray:
        return self.model.predict(x_values)

    def save(self, output_path: Path) -> None:
        with output_path.open("wb") as file:
            pickle.dump(self.model, file)

    @classmethod
    def load(cls, model_path: Path) -> "RandomForestRULModel":
        instance = cls()
        with model_path.open("rb") as file:
            instance.model = pickle.load(file)
        return instance

    @property
    def feature_importances_(self) -> np.ndarray:
        return self.model.feature_importances_
