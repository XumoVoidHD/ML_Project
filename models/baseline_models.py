from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR


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


class SVRRULModel:
    def __init__(
        self,
        c: float = 10.0,
        epsilon: float = 0.1,
        kernel: str = "rbf",
        gamma: str | float = "scale",
    ) -> None:
        self.model = SVR(C=c, epsilon=epsilon, kernel=kernel, gamma=gamma)
        self.calibrator: LinearRegression | None = None

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_calibration: np.ndarray | None = None,
        y_calibration: np.ndarray | None = None,
    ) -> None:
        self.model.fit(x_train, y_train)
        if x_calibration is not None and y_calibration is not None:
            calibration_predictions = self.model.predict(x_calibration).reshape(-1, 1)
            self.calibrator = LinearRegression()
            self.calibrator.fit(calibration_predictions, y_calibration)

    def predict(self, x_values: np.ndarray) -> np.ndarray:
        predictions = self.model.predict(x_values)
        if self.calibrator is None:
            return predictions
        return self.calibrator.predict(predictions.reshape(-1, 1))

    def save(self, output_path: Path) -> None:
        with output_path.open("wb") as file:
            pickle.dump({"model": self.model, "calibrator": self.calibrator}, file)

    @classmethod
    def load(cls, model_path: Path) -> "SVRRULModel":
        instance = cls()
        with model_path.open("rb") as file:
            payload = pickle.load(file)
        if isinstance(payload, dict) and "model" in payload:
            instance.model = payload["model"]
            instance.calibrator = payload.get("calibrator")
        else:
            instance.model = payload
        return instance


class MLPRULModel:
    def __init__(
        self,
        seed: int = 42,
        hidden_dim: int = 64,
        alpha: float = 1e-4,
        learning_rate_init: float = 1e-3,
        max_iter: int = 500,
    ) -> None:
        self.model = MLPRegressor(
            hidden_layer_sizes=(hidden_dim, hidden_dim),
            activation="relu",
            solver="adam",
            alpha=alpha,
            learning_rate_init=learning_rate_init,
            max_iter=max_iter,
            early_stopping=True,
            validation_fraction=0.2,
            n_iter_no_change=20,
            random_state=seed,
        )

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        self.model.fit(x_train, y_train)

    def predict(self, x_values: np.ndarray) -> np.ndarray:
        return self.model.predict(x_values)

    def save(self, output_path: Path) -> None:
        with output_path.open("wb") as file:
            pickle.dump(self.model, file)

    @classmethod
    def load(cls, model_path: Path) -> "MLPRULModel":
        instance = cls()
        with model_path.open("rb") as file:
            instance.model = pickle.load(file)
        return instance
