from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denominator = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if denominator == 0:
        return 0.0
    numerator = float(np.sum((y_true - y_pred) ** 2))
    return float(1.0 - (numerator / denominator))


def compute_metrics(frame: pd.DataFrame) -> dict[str, float]:
    y_true = frame["actual_rul"].to_numpy(dtype=float)
    y_pred = frame["predicted_rul"].to_numpy(dtype=float)
    return {"mae": mae(y_true, y_pred), "rmse": rmse(y_true, y_pred), "r2": r2(y_true, y_pred)}


def compute_per_battery_metrics(frame: pd.DataFrame) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for battery_id, battery_frame in frame.groupby("battery_id"):
        metrics = compute_metrics(battery_frame)
        metrics["battery_id"] = battery_id
        metrics["num_samples"] = int(len(battery_frame))
        results.append(metrics)
    return results


def save_predictions(predictions: pd.DataFrame, output_path: Path) -> None:
    predictions.to_csv(output_path, index=False)


def save_metrics(metrics: dict[str, Any], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)


def save_loss_curve(history: dict[str, list[float]], output_path: Path) -> None:
    if not history.get("train_loss"):
        return
    plt.figure(figsize=(8, 5))
    plt.plot(history["train_loss"], label="Train Loss")
    if history.get("val_loss"):
        plt.plot(history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_scatter_plot(predictions: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(6, 6))
    plt.scatter(predictions["actual_rul"], predictions["predicted_rul"], alpha=0.7)
    min_value = min(predictions["actual_rul"].min(), predictions["predicted_rul"].min())
    max_value = max(predictions["actual_rul"].max(), predictions["predicted_rul"].max())
    plt.plot([min_value, max_value], [min_value, max_value], linestyle="--", color="black")
    plt.xlabel("Actual RUL")
    plt.ylabel("Predicted RUL")
    plt.title("Predicted vs Actual RUL")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_cycle_plot(predictions: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(10, 5))
    for battery_id, battery_frame in predictions.groupby("battery_id"):
        battery_frame = battery_frame.sort_values("discharge_cycle_index")
        plt.plot(
            battery_frame["discharge_cycle_index"],
            battery_frame["actual_rul"],
            label=f"{battery_id} Actual",
            linewidth=2,
        )
        plt.plot(
            battery_frame["discharge_cycle_index"],
            battery_frame["predicted_rul"],
            label=f"{battery_id} Predicted",
            linestyle="--",
        )
    plt.xlabel("Discharge Cycle Index")
    plt.ylabel("RUL")
    plt.title("RUL by Cycle Index")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_residual_histogram(predictions: pd.DataFrame, output_path: Path) -> None:
    residuals = predictions["predicted_rul"] - predictions["actual_rul"]
    plt.figure(figsize=(8, 5))
    plt.hist(residuals, bins=20, alpha=0.8, edgecolor="black")
    plt.xlabel("Residual")
    plt.ylabel("Count")
    plt.title("Residual Histogram")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_feature_relevance_plot(feature_names: list[str], importance_values: np.ndarray, output_path: Path) -> None:
    ordering = np.argsort(importance_values)
    sorted_features = [feature_names[index] for index in ordering]
    sorted_values = importance_values[ordering]
    plt.figure(figsize=(8, 5))
    plt.barh(sorted_features, sorted_values)
    plt.xlabel("Relevance")
    plt.ylabel("Feature")
    plt.title("Feature Relevance")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_all_plots(
    predictions: pd.DataFrame,
    plot_dir: Path,
    history: dict[str, list[float]] | None = None,
    feature_importance: tuple[list[str], np.ndarray] | None = None,
) -> None:
    plot_dir.mkdir(parents=True, exist_ok=True)
    if history is not None:
        save_loss_curve(history, plot_dir / "loss_curve.png")
    save_scatter_plot(predictions, plot_dir / "predicted_vs_actual_scatter.png")
    save_cycle_plot(predictions, plot_dir / "predicted_vs_actual_vs_cycle.png")
    save_residual_histogram(predictions, plot_dir / "residual_histogram.png")
    if feature_importance is not None:
        save_feature_relevance_plot(feature_importance[0], feature_importance[1], plot_dir / "feature_importance.png")
