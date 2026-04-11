from __future__ import annotations

import json
import random
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from models.gru_model import BatteryGRURegressor
from models.lstm_model import BatteryLSTMRegressor
from models.baseline_models import LinearRULModel, MLPRULModel, RandomForestRULModel, RidgeRULModel, SVRRULModel
from models.xgboost_model import XGBoostRULModel
from preprocessing.dataset import BatterySequenceDataset, load_feature_columns, load_processed_split
from training.evaluate import compute_metrics, compute_per_battery_metrics, save_all_plots, save_metrics, save_predictions


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
PROCESSED_DIR = PROJECT_ROOT / "processed"


@dataclass
class TrainingConfig:
    model_name: str
    seed: int
    sequence_length: int = 10
    batch_size: int = 16
    learning_rate: float = 1e-3
    max_epochs: int = 100
    patience: int = 12
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    n_estimators: int = 500
    max_depth: int = 4
    min_child_weight: float = 2.0
    subsample: float = 0.9
    colsample_bytree: float = 0.9
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0
    ridge_alpha: float = 1.0
    rf_n_estimators: int = 300
    rf_max_depth: int = 0
    rf_min_samples_leaf: int = 1
    svr_c: float = 10.0
    svr_epsilon: float = 0.1
    svr_kernel: str = "rbf"
    svr_gamma: str = "scale"
    mlp_hidden_dim: int = 64
    mlp_alpha: float = 1e-4
    mlp_learning_rate_init: float = 1e-3
    mlp_max_iter: int = 500


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_experiment_dir(model_name: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = EXPERIMENTS_DIR / f"{model_name}_{timestamp}"
    (output_dir / "plots").mkdir(parents=True, exist_ok=True)
    return output_dir


def ensure_processed_data() -> None:
    required_files = [
        PROCESSED_DIR / "train.csv",
        PROCESSED_DIR / "val.csv",
        PROCESSED_DIR / "test.csv",
        PROCESSED_DIR / "feature_columns.json",
    ]
    missing = [str(path) for path in required_files if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing processed artifacts. Run `python preprocessing/pipeline.py` first.\n"
            + "\n".join(missing)
        )


def load_row_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    feature_columns = load_feature_columns()
    train_frame = load_processed_split("train")
    val_frame = load_processed_split("val")
    test_frame = load_processed_split("test")
    return train_frame, val_frame, test_frame, feature_columns


def build_row_prediction_frame(frame: pd.DataFrame, predictions: np.ndarray) -> pd.DataFrame:
    cycle_column = "raw_discharge_cycle_index" if "raw_discharge_cycle_index" in frame.columns else "discharge_cycle_index"
    result = frame[["battery_id", cycle_column, "RUL"]].copy()
    result = result.rename(columns={cycle_column: "discharge_cycle_index", "RUL": "actual_rul"})
    result["predicted_rul"] = predictions
    return result


def build_sequence_loaders(
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    feature_columns: list[str],
    sequence_length: int,
    batch_size: int,
) -> tuple[DataLoader[Any], DataLoader[Any], DataLoader[Any]]:
    train_dataset = BatterySequenceDataset(train_frame, feature_columns, sequence_length)
    val_dataset = BatterySequenceDataset(val_frame, feature_columns, sequence_length)
    test_dataset = BatterySequenceDataset(test_frame, feature_columns, sequence_length)
    if len(train_dataset) == 0 or len(val_dataset) == 0 or len(test_dataset) == 0:
        raise ValueError(
            f"Sequence length {sequence_length} is too long for at least one split. "
            "Choose a smaller value."
        )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def evaluate_sequence_model(
    model: nn.Module,
    data_loader: DataLoader[Any],
    device: torch.device,
) -> tuple[float, pd.DataFrame]:
    model.eval()
    loss_fn = nn.MSELoss()
    losses: list[float] = []
    records: list[dict[str, Any]] = []

    with torch.no_grad():
        for features, targets, battery_ids, cycle_indices in data_loader:
            features = features.to(device)
            targets = targets.to(device)
            predictions = model(features)
            loss = loss_fn(predictions, targets)
            losses.append(float(loss.item()))

            pred_np = predictions.detach().cpu().numpy()
            target_np = targets.detach().cpu().numpy()
            for prediction, target, battery_id, cycle_index in zip(pred_np, target_np, battery_ids, cycle_indices):
                records.append(
                    {
                        "battery_id": battery_id,
                        "discharge_cycle_index": int(cycle_index),
                        "actual_rul": float(target),
                        "predicted_rul": float(prediction),
                    }
                )

    predictions_frame = pd.DataFrame(records)
    return float(np.mean(losses)), predictions_frame


def train_torch_sequence_model(config: TrainingConfig) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_frame, val_frame, test_frame, feature_columns = load_row_splits()
    train_loader, val_loader, test_loader = build_sequence_loaders(
        train_frame,
        val_frame,
        test_frame,
        feature_columns,
        config.sequence_length,
        config.batch_size,
    )

    input_dim = len(feature_columns)
    if config.model_name == "lstm":
        model = BatteryLSTMRegressor(
            input_dim=input_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout,
            bidirectional=True,
        )
    else:
        model = BatteryGRURegressor(
            input_dim=input_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout,
        )

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.MSELoss()

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
    best_state: dict[str, torch.Tensor] | None = None
    best_val_loss = float("inf")
    patience_counter = 0

    for _epoch in range(config.max_epochs):
        model.train()
        batch_losses: list[float] = []
        for features, targets, _, _ in train_loader:
            features = features.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            predictions = model(features)
            loss = loss_fn(predictions, targets)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.item()))

        train_loss = float(np.mean(batch_losses))
        val_loss, _ = evaluate_sequence_model(model, val_loader, device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= config.patience:
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint.")

    model.load_state_dict(best_state)
    _, val_predictions = evaluate_sequence_model(model, val_loader, device)
    _, test_predictions = evaluate_sequence_model(model, test_loader, device)
    return {
        "model": model,
        "history": history,
        "val_predictions": val_predictions,
        "test_predictions": test_predictions,
        "feature_columns": feature_columns,
        "device": str(device),
    }


def train_xgboost_model(config: TrainingConfig) -> dict[str, Any]:
    train_frame, val_frame, test_frame, feature_columns = load_row_splits()
    x_train = train_frame[feature_columns].to_numpy(dtype=float)
    y_train = train_frame["RUL"].to_numpy(dtype=float)
    x_val = val_frame[feature_columns].to_numpy(dtype=float)
    y_val = val_frame["RUL"].to_numpy(dtype=float)
    x_test = test_frame[feature_columns].to_numpy(dtype=float)

    model = XGBoostRULModel(
        seed=config.seed,
        n_estimators=config.n_estimators,
        learning_rate=config.learning_rate,
        max_depth=config.max_depth,
        min_child_weight=config.min_child_weight,
        subsample=config.subsample,
        colsample_bytree=config.colsample_bytree,
        reg_alpha=config.reg_alpha,
        reg_lambda=config.reg_lambda,
    )
    model.fit(x_train, y_train, x_val, y_val)

    val_predictions = build_row_prediction_frame(val_frame, model.predict(x_val))
    test_predictions = build_row_prediction_frame(test_frame, model.predict(x_test))

    return {
        "model": model,
        "history": {
            "train_loss": model.evals_result().get("validation_0", {}).get("rmse", []),
            "val_loss": model.evals_result().get("validation_1", {}).get("rmse", []),
        },
        "val_predictions": val_predictions,
        "test_predictions": test_predictions,
        "feature_columns": feature_columns,
    }


def train_sklearn_row_model(config: TrainingConfig) -> dict[str, Any]:
    train_frame, val_frame, test_frame, feature_columns = load_row_splits()
    x_train = train_frame[feature_columns].to_numpy(dtype=float)
    y_train = train_frame["RUL"].to_numpy(dtype=float)
    x_val = val_frame[feature_columns].to_numpy(dtype=float)
    x_test = test_frame[feature_columns].to_numpy(dtype=float)

    if config.model_name == "linear":
        model = LinearRULModel()
    elif config.model_name == "ridge":
        model = RidgeRULModel(alpha=config.ridge_alpha)
    elif config.model_name == "random_forest":
        max_depth = config.rf_max_depth if config.rf_max_depth > 0 else None
        model = RandomForestRULModel(
            seed=config.seed,
            n_estimators=config.rf_n_estimators,
            max_depth=max_depth,
            min_samples_leaf=config.rf_min_samples_leaf,
        )
    elif config.model_name == "svr":
        model = SVRRULModel(
            c=config.svr_c,
            epsilon=config.svr_epsilon,
            kernel=config.svr_kernel,
            gamma=config.svr_gamma,
        )
    elif config.model_name == "mlp":
        model = MLPRULModel(
            seed=config.seed,
            hidden_dim=config.mlp_hidden_dim,
            alpha=config.mlp_alpha,
            learning_rate_init=config.mlp_learning_rate_init,
            max_iter=config.mlp_max_iter,
        )
    else:
        raise ValueError(f"Unsupported row model: {config.model_name}")

    if config.model_name == "svr":
        model.fit(x_train, y_train, x_val, val_frame["RUL"].to_numpy(dtype=float))
    else:
        model.fit(x_train, y_train)
    val_predictions = build_row_prediction_frame(val_frame, model.predict(x_val))
    test_predictions = build_row_prediction_frame(test_frame, model.predict(x_test))
    return {
        "model": model,
        "history": {"train_loss": [], "val_loss": []},
        "val_predictions": val_predictions,
        "test_predictions": test_predictions,
        "feature_columns": feature_columns,
    }


def save_model_artifact(model_name: str, model: Any, output_dir: Path, feature_columns: list[str], config: TrainingConfig) -> str:
    if model_name == "xgboost":
        model_path = output_dir / "model.json"
        model.save(model_path)
        return model_path.name

    if model_name in {"linear", "ridge", "random_forest", "svr", "mlp"}:
        model_path = output_dir / "model.pkl"
        model.save(model_path)
        return model_path.name

    model_path = output_dir / "model.pt"
    payload = {
        "state_dict": model.state_dict(),
        "model_name": model_name,
        "feature_columns": feature_columns,
        "sequence_length": config.sequence_length,
        "hidden_dim": config.hidden_dim,
        "num_layers": config.num_layers,
        "dropout": config.dropout,
    }
    torch.save(payload, model_path)
    return model_path.name


def update_summary(output_dir: Path, config: TrainingConfig, metrics: dict[str, Any]) -> None:
    summary_path = EXPERIMENTS_DIR / "summary.csv"
    summary_row = {
        "experiment_dir": output_dir.name,
        "model_name": config.model_name,
        "seed": config.seed,
        "sequence_length": config.sequence_length if config.model_name in {"lstm", "gru"} else None,
        "test_mae": metrics["test"]["overall"]["mae"],
        "test_rmse": metrics["test"]["overall"]["rmse"],
        "test_r2": metrics["test"]["overall"]["r2"],
        "val_mae": metrics["validation"]["overall"]["mae"],
        "val_rmse": metrics["validation"]["overall"]["rmse"],
        "val_r2": metrics["validation"]["overall"]["r2"],
    }
    if summary_path.exists():
        summary_frame = pd.read_csv(summary_path)
        for column in summary_row:
            if column not in summary_frame.columns:
                summary_frame[column] = np.nan
        summary_frame.loc[len(summary_frame)] = {
            column: summary_row.get(column, np.nan) for column in summary_frame.columns
        }
    else:
        summary_frame = pd.DataFrame([summary_row])
    summary_frame.to_csv(summary_path, index=False)


def save_config(output_dir: Path, config: TrainingConfig, feature_columns: list[str]) -> None:
    hyperparameters: dict[str, Any] = {"learning_rate": config.learning_rate}
    if config.model_name == "xgboost":
        hyperparameters.update(
            {
                "n_estimators": config.n_estimators,
                "max_depth": config.max_depth,
                "min_child_weight": config.min_child_weight,
                "subsample": config.subsample,
                "colsample_bytree": config.colsample_bytree,
                "reg_alpha": config.reg_alpha,
                "reg_lambda": config.reg_lambda,
            }
        )
    elif config.model_name == "ridge":
        hyperparameters.update({"alpha": config.ridge_alpha})
    elif config.model_name == "random_forest":
        hyperparameters.update(
            {
                "n_estimators": config.rf_n_estimators,
                "max_depth": config.rf_max_depth if config.rf_max_depth > 0 else None,
                "min_samples_leaf": config.rf_min_samples_leaf,
            }
        )
    elif config.model_name == "svr":
        hyperparameters.update(
            {
                "C": config.svr_c,
                "epsilon": config.svr_epsilon,
                "kernel": config.svr_kernel,
                "gamma": config.svr_gamma,
                "posthoc_calibration": "linear regression fitted on validation predictions",
            }
        )
    elif config.model_name == "mlp":
        hyperparameters.update(
            {
                "hidden_dim": config.mlp_hidden_dim,
                "alpha": config.mlp_alpha,
                "learning_rate_init": config.mlp_learning_rate_init,
                "max_iter": config.mlp_max_iter,
            }
        )
    else:
        hyperparameters.update(
            {
                "sequence_length": config.sequence_length,
                "batch_size": config.batch_size,
                "max_epochs": config.max_epochs,
                "patience": config.patience,
                "hidden_dim": config.hidden_dim,
                "num_layers": config.num_layers,
                "dropout": config.dropout,
            }
        )

    config_payload = {
        "model_name": config.model_name,
        "hyperparameters": hyperparameters,
        "seed": config.seed,
        "feature_list": feature_columns,
        "split_definition": {
            "train": ["B0005", "B0006"],
            "validation": "last 20% of supervised cycles within each training battery",
            "test": ["B0018"],
            "censored": ["B0007"],
        },
        "preprocessing_settings": {
            "supervised_cycle_type": "discharge",
            "rul_definition": "remaining discharge cycles until capacity < 1.4 Ah",
            "capacity_fade_definition": "1 - current_capacity / first_discharge_capacity",
            "scaling": "StandardScaler fit on training rows only",
            "impedance_features": "Re and Rct forward-filled within each battery from prior impedance cycles only",
        },
    }
    with (output_dir / "config.json").open("w", encoding="utf-8") as file:
        json.dump(config_payload, file, indent=2)


def train_experiment(config: TrainingConfig) -> Path:
    ensure_processed_data()
    set_seed(config.seed)
    output_dir = build_experiment_dir(config.model_name)

    if config.model_name == "xgboost":
        results = train_xgboost_model(config)
        feature_importance = (results["feature_columns"], results["model"].feature_importances_)
    elif config.model_name in {"linear", "ridge", "random_forest", "svr", "mlp"}:
        results = train_sklearn_row_model(config)
        if hasattr(results["model"], "feature_importances_"):
            feature_importance = (results["feature_columns"], results["model"].feature_importances_)
        else:
            feature_importance = None
    elif config.model_name in {"lstm", "gru"}:
        results = train_torch_sequence_model(config)
        feature_importance = None
    else:
        raise ValueError(f"Unsupported model: {config.model_name}")

    val_predictions = results["val_predictions"]
    test_predictions = results["test_predictions"]
    metrics = {
        "validation": {
            "overall": compute_metrics(val_predictions),
            "per_battery": compute_per_battery_metrics(val_predictions),
        },
        "test": {
            "overall": compute_metrics(test_predictions),
            "per_battery": compute_per_battery_metrics(test_predictions),
        },
    }

    model_file_name = save_model_artifact(
        config.model_name,
        results["model"],
        output_dir,
        results["feature_columns"],
        config,
    )
    save_predictions(test_predictions, output_dir / "predictions.csv")
    save_metrics(metrics, output_dir / "metrics.json")
    save_config(output_dir, config, results["feature_columns"])
    save_all_plots(
        predictions=test_predictions,
        plot_dir=output_dir / "plots",
        history=results["history"],
        feature_importance=feature_importance,
    )
    update_summary(output_dir, config, metrics)

    with (output_dir / "run_info.json").open("w", encoding="utf-8") as file:
        json.dump({"model_file": model_file_name}, file, indent=2)

    return output_dir
