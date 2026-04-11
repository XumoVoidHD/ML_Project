from __future__ import annotations

import argparse
import itertools
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent

COMMON_GRID: dict[str, list[Any]] = {
    "seed": [42],
}

MODEL_GRIDS: dict[str, dict[str, list[Any]]] = {
    "xgboost": {
        "n_estimators": [300, 500, 800],
        "learning_rate": [0.02, 0.03, 0.05],
        "max_depth": [3, 4, 5, 6],
        "min_child_weight": [1.0, 2.0, 4.0],
        "subsample": [0.8, 0.9],
        "colsample_bytree": [0.7, 0.8, 0.9],
        "reg_alpha": [0.0, 0.1],
        "reg_lambda": [1.0, 2.0, 5.0],
    },
    "random_forest": {
        "rf_n_estimators": [200, 400, 600],
        "rf_max_depth": [4, 6, 8, 10],
        "rf_min_samples_leaf": [2, 4, 6],
    },
    "ridge": {
        "ridge_alpha": [1.0, 10.0, 25.0, 50.0, 100.0],
    },
    "linear": {},
    "lstm": {
        "sequence_length": [5, 8, 10],
        "batch_size": [8, 16],
        "learning_rate": [0.001, 0.0005],
        "max_epochs": [100, 200],
        "patience": [12, 20],
        "hidden_dim": [32, 48, 64],
        "num_layers": [1, 2],
        "dropout": [0.2, 0.3],
    },
    "gru": {
        "sequence_length": [5, 8, 10],
        "batch_size": [8, 16],
        "learning_rate": [0.001, 0.0005],
        "max_epochs": [100, 200],
        "patience": [12, 20],
        "hidden_dim": [32, 48, 64],
        "num_layers": [1, 2],
        "dropout": [0.2, 0.3],
    },
}

FOCUSED_MODEL_GRIDS: dict[str, dict[str, list[Any]]] = {
    "xgboost": {
        "n_estimators": [800, 1000, 600],
        "learning_rate": [0.02, 0.03, 0.015],
        "max_depth": [3, 4, 2],
        "min_child_weight": [2.0, 3.0, 1.0],
        "subsample": [0.9, 0.85, 1.0],
        "colsample_bytree": [0.8, 0.9, 0.7],
        "reg_alpha": [0.1, 0.2, 0.0],
        "reg_lambda": [5.0, 7.0, 3.0],
    },
    "random_forest": {
        "rf_n_estimators": [600, 800, 400],
        "rf_max_depth": [8, 10, 6, 12],
        "rf_min_samples_leaf": [6, 5, 4, 8],
    },
    "ridge": {
        "ridge_alpha": [50.0, 40.0, 60.0, 25.0, 75.0, 100.0],
    },
    "lstm": {
        "sequence_length": [5, 4, 6],
        "batch_size": [8, 12],
        "learning_rate": [0.0005, 0.0003, 0.0007],
        "max_epochs": [200, 300],
        "patience": [20, 30],
        "hidden_dim": [32, 24, 40],
        "num_layers": [1],
        "dropout": [0.3, 0.25, 0.35],
    },
}

PRESET_MODELS: dict[str, list[str]] = {
    "focused": ["ridge", "random_forest", "xgboost", "lstm"],
    "optimized": ["linear", "ridge", "random_forest", "xgboost", "lstm", "gru"],
    "recommended": ["random_forest", "xgboost", "ridge"],
    "tabular": ["linear", "ridge", "random_forest", "xgboost"],
    "sequence": ["lstm", "gru"],
    "all": ["linear", "ridge", "random_forest", "xgboost", "lstm", "gru"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run model sweeps to populate experiment runs.")
    parser.add_argument(
        "--preset",
        choices=sorted(PRESET_MODELS),
        default="recommended",
        help="Named group of models to sweep.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=sorted(MODEL_GRIDS),
        help="Optional explicit model list. Overrides --preset.",
    )
    parser.add_argument(
        "--max-runs-per-model",
        type=int,
        default=8,
        help="Caps the number of combinations executed for each model.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Skip the first N combinations in each model grid.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing training.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop immediately if any training command fails.",
    )
    return parser.parse_args()


def iter_grid(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    if not grid:
        return [{}]
    keys = list(grid)
    values_product = itertools.product(*(grid[key] for key in keys))
    return [dict(zip(keys, values, strict=True)) for values in values_product]


def build_command(model_name: str, params: dict[str, Any]) -> list[str]:
    command = [sys.executable, "train.py", "--model", model_name]
    for key, value in params.items():
        command.extend([f"--{key}", str(value)])
    return command


def format_command(command: list[str]) -> str:
    return " ".join(command)


def get_models(args: argparse.Namespace) -> list[str]:
    if args.models:
        return args.models
    return PRESET_MODELS[args.preset]


def run_sweeps(args: argparse.Namespace) -> int:
    models = get_models(args)
    total_runs = 0
    failures = 0

    for model_name in models:
        base_grid = FOCUSED_MODEL_GRIDS.get(model_name, MODEL_GRIDS[model_name]) if args.preset == "focused" and not args.models else MODEL_GRIDS[model_name]
        model_grid = {**COMMON_GRID, **base_grid}
        combinations = iter_grid(model_grid)
        selected_combinations = combinations[args.start_index : args.start_index + args.max_runs_per_model]
        print(f"\n[{model_name}] {len(selected_combinations)} run(s) selected from {len(combinations)} combinations")

        for run_number, params in enumerate(selected_combinations, start=1):
            command = build_command(model_name, params)
            print(f"{model_name} run {run_number}: {format_command(command)}")
            total_runs += 1

            if args.dry_run:
                continue

            process = subprocess.run(command, cwd=PROJECT_ROOT)
            if process.returncode != 0:
                failures += 1
                print(f"Command failed with exit code {process.returncode}")
                if args.stop_on_error:
                    return 1

    print(f"\nFinished {total_runs} run(s) with {failures} failure(s).")
    return 1 if failures else 0


def main() -> None:
    args = parse_args()
    raise SystemExit(run_sweeps(args))


if __name__ == "__main__":
    main()
