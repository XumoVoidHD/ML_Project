"""
Train a Remaining Useful Life (RUL) model and export predictions.

This script is the simplest end-to-end entrypoint in the repo:
1. Rebuild preprocessed data if requested or missing
2. Train a regression model on batteries B0005, B0006, B0007
3. Predict RUL on the held-out battery B0018
4. Save the trained model, metrics, and prediction CSV
"""

import argparse
import json
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from rul import preprocess_rul
from rul.paths import ARTIFACTS_DIR, PREPROCESSED_DIR
from rul.utils import load_data

EXTENDED_FEATURES = [
    "Capacity",
    "capacity_fade",
    "Capacity_normalized",
    "capacity_fade_normalized",
    "capacity_derivative",
    "Re",
    "Rct",
    "discharge_duration",
    "avg_temperature",
    "voltage_at_100s",
    "voltage_at_300s",
    "voltage_at_600s",
    "ambient_temperature",
    "discharge_cycle_index",
]


def build_model(model_name: str):
    """Create the requested regression model."""
    if model_name == "gradient_boosting":
        return GradientBoostingRegressor(
            n_estimators=250,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
        )

    return RandomForestRegressor(
        n_estimators=150,
        max_depth=12,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=1,
    )


def ensure_preprocessed_data(force_rebuild: bool) -> None:
    """Create train/test preprocessed data when needed."""
    required_files = [
        PREPROCESSED_DIR / "rul_preprocessed.csv",
        PREPROCESSED_DIR / "rul_train.csv",
        PREPROCESSED_DIR / "rul_test.csv",
    ]
    missing = [path for path in required_files if not path.exists()]

    if force_rebuild or missing:
        print("Preparing preprocessed RUL data...")
        preprocess_rul.main()


def compute_metrics(y_true, y_pred) -> dict:
    """Return regression metrics as JSON-serializable floats."""
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def save_artifacts(model_name, model, feature_cols, train_df, test_df, y_pred_train, y_pred_test, metrics):
    """Persist model, metrics, and prediction tables."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = ARTIFACTS_DIR / f"rul_{model_name}.pkl"
    with open(model_path, "wb") as handle:
        pickle.dump({"model": model, "feature_cols": feature_cols}, handle)

    metrics_path = ARTIFACTS_DIR / f"rul_{model_name}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    train_predictions = train_df.copy()
    train_predictions["predicted_RUL"] = y_pred_train
    train_predictions["absolute_error"] = (train_predictions["RUL"] - train_predictions["predicted_RUL"]).abs()
    train_predictions.to_csv(ARTIFACTS_DIR / f"rul_{model_name}_train_predictions.csv", index=False)

    test_predictions = test_df.copy()
    test_predictions["predicted_RUL"] = y_pred_test
    test_predictions["absolute_error"] = (test_predictions["RUL"] - test_predictions["predicted_RUL"]).abs()
    test_predictions.to_csv(ARTIFACTS_DIR / f"rul_{model_name}_test_predictions.csv", index=False)

    return model_path, metrics_path


def main():
    parser = argparse.ArgumentParser(description="Train an RUL model and export predictions.")
    parser.add_argument(
        "--model",
        choices=["random_forest", "gradient_boosting"],
        default="random_forest",
        help="Regression model to train.",
    )
    parser.add_argument(
        "--rebuild-preprocessed",
        action="store_true",
        help="Recompute preprocessed train/test files before training.",
    )
    args = parser.parse_args()

    ensure_preprocessed_data(force_rebuild=args.rebuild_preprocessed)

    X_train, y_train, X_test, y_test, feature_cols, train_df, test_df = load_data(EXTENDED_FEATURES)
    model = build_model(args.model)
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    metrics = {
        "model": args.model,
        "train": compute_metrics(y_train, y_pred_train),
        "test": compute_metrics(y_test, y_pred_test),
        "feature_cols": feature_cols,
    }

    model_path, metrics_path = save_artifacts(
        args.model,
        model,
        feature_cols,
        train_df,
        test_df,
        y_pred_train,
        y_pred_test,
        metrics,
    )

    print("=" * 60)
    print(f"RUL prediction model: {args.model}")
    print("=" * 60)
    print(f"Train MAE : {metrics['train']['mae']:.3f} cycles")
    print(f"Train RMSE: {metrics['train']['rmse']:.3f} cycles")
    print(f"Train R2  : {metrics['train']['r2']:.4f}")
    print()
    print("Held-out battery: B0018")
    print(f"Test MAE  : {metrics['test']['mae']:.3f} cycles")
    print(f"Test RMSE : {metrics['test']['rmse']:.3f} cycles")
    print(f"Test R2   : {metrics['test']['r2']:.4f}")
    print()
    print(f"Saved model   : {model_path}")
    print(f"Saved metrics : {metrics_path}")
    print(f"Saved outputs : {ARTIFACTS_DIR / f'rul_{args.model}_test_predictions.csv'}")


if __name__ == "__main__":
    main()
