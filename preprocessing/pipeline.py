from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_SERIES_DIR = DATA_DIR / "data"
PROCESSED_DIR = PROJECT_ROOT / "processed"

TRAIN_BATTERIES = ("B0005", "B0006")
TEST_BATTERIES = ("B0018",)
CENSORED_BATTERIES = ("B0007",)
FAILURE_THRESHOLD = 1.4
VALIDATION_RATIO = 0.2

FEATURE_COLUMNS = [
    "Capacity",
    "capacity_fade",
    "capacity_derivative",
    "Re",
    "Rct",
    "discharge_duration",
    "avg_temperature",
    "voltage_at_100s",
    "voltage_at_300s",
    "voltage_at_600s",
    "discharge_cycle_index",
]


@dataclass
class PipelineArtifacts:
    full_frame: pd.DataFrame
    train_frame: pd.DataFrame
    val_frame: pd.DataFrame
    test_frame: pd.DataFrame
    censored_frame: pd.DataFrame
    feature_columns: list[str]
    scaler: StandardScaler
    train_impute_values: dict[str, float]


def ensure_directories() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def load_metadata() -> pd.DataFrame:
    metadata = pd.read_csv(DATA_DIR / "metadata.csv")
    metadata["type"] = metadata["type"].str.lower()
    metadata["uid"] = pd.to_numeric(metadata["uid"], errors="coerce")
    metadata["Capacity"] = pd.to_numeric(metadata["Capacity"], errors="coerce")
    metadata["Re"] = pd.to_numeric(metadata["Re"], errors="coerce")
    metadata["Rct"] = pd.to_numeric(metadata["Rct"], errors="coerce")
    return metadata


def interpolate_voltage_at_seconds(frame: pd.DataFrame, second_mark: float) -> float:
    time_values = pd.to_numeric(frame["Time"], errors="coerce").to_numpy(dtype=float)
    voltage_values = pd.to_numeric(frame["Voltage_measured"], errors="coerce").to_numpy(dtype=float)
    valid_mask = np.isfinite(time_values) & np.isfinite(voltage_values)
    time_values = time_values[valid_mask]
    voltage_values = voltage_values[valid_mask]

    if len(time_values) == 0:
        return float("nan")

    order = np.argsort(time_values)
    time_values = time_values[order]
    voltage_values = voltage_values[order]
    second_mark = float(np.clip(second_mark, time_values[0], time_values[-1]))
    return float(np.interp(second_mark, time_values, voltage_values))


def extract_discharge_timeseries_features(csv_filename: str) -> dict[str, float]:
    series_frame = pd.read_csv(RAW_SERIES_DIR / csv_filename)

    time_values = pd.to_numeric(series_frame["Time"], errors="coerce")
    temperature_values = pd.to_numeric(series_frame["Temperature_measured"], errors="coerce")

    discharge_duration = float(time_values.max()) if time_values.notna().any() else float("nan")
    avg_temperature = float(temperature_values.mean()) if temperature_values.notna().any() else float("nan")

    return {
        "discharge_duration": discharge_duration,
        "avg_temperature": avg_temperature,
        "voltage_at_100s": interpolate_voltage_at_seconds(series_frame, 100.0),
        "voltage_at_300s": interpolate_voltage_at_seconds(series_frame, 300.0),
        "voltage_at_600s": interpolate_voltage_at_seconds(series_frame, 600.0),
    }


def compute_failure_cycle_index(discharge_rows: pd.DataFrame) -> int | None:
    below_threshold = discharge_rows.index[discharge_rows["Capacity"] < FAILURE_THRESHOLD]
    if len(below_threshold) == 0:
        return None
    return int(discharge_rows.loc[below_threshold[0], "discharge_cycle_index"])


def create_battery_feature_rows(metadata: pd.DataFrame, battery_id: str) -> pd.DataFrame:
    battery_rows = metadata.loc[metadata["battery_id"] == battery_id].copy()
    battery_rows = battery_rows.sort_values(["uid", "filename"], na_position="last").reset_index(drop=True)

    last_re = np.nan
    last_rct = np.nan
    discharge_cycle_index = 0
    first_capacity = np.nan
    previous_capacity = np.nan
    engineered_rows: list[dict[str, Any]] = []

    for row in battery_rows.itertuples(index=False):
        if row.type == "impedance":
            if pd.notna(row.Re):
                last_re = float(row.Re)
            if pd.notna(row.Rct):
                last_rct = float(row.Rct)
            continue

        if row.type != "discharge":
            continue

        discharge_cycle_index += 1
        capacity = float(row.Capacity) if pd.notna(row.Capacity) else float("nan")
        if discharge_cycle_index == 1:
            first_capacity = capacity
            capacity_derivative = 0.0
        else:
            capacity_derivative = (
                capacity - previous_capacity if pd.notna(capacity) and pd.notna(previous_capacity) else float("nan")
            )

        capacity_fade = (
            1.0 - (capacity / first_capacity)
            if pd.notna(capacity) and pd.notna(first_capacity) and first_capacity != 0
            else float("nan")
        )

        feature_row = {
            "battery_id": battery_id,
            "filename": row.filename,
            "uid": int(row.uid) if pd.notna(row.uid) else discharge_cycle_index,
            "type": row.type,
            "Capacity": capacity,
            "capacity_fade": capacity_fade,
            "capacity_derivative": capacity_derivative,
            "Re": float(last_re) if pd.notna(last_re) else float("nan"),
            "Rct": float(last_rct) if pd.notna(last_rct) else float("nan"),
            "discharge_cycle_index": discharge_cycle_index,
            "ambient_temperature": float(row.ambient_temperature),
        }
        feature_row.update(extract_discharge_timeseries_features(row.filename))
        engineered_rows.append(feature_row)
        previous_capacity = capacity

    battery_frame = pd.DataFrame(engineered_rows)
    if battery_frame.empty:
        return battery_frame

    failure_cycle_index = compute_failure_cycle_index(battery_frame)
    battery_frame["failure_cycle_index"] = failure_cycle_index
    battery_frame["is_censored"] = failure_cycle_index is None
    if failure_cycle_index is None:
        battery_frame["RUL"] = np.nan
        return battery_frame

    battery_frame = battery_frame.loc[battery_frame["discharge_cycle_index"] <= failure_cycle_index].copy()
    battery_frame["RUL"] = failure_cycle_index - battery_frame["discharge_cycle_index"]
    return battery_frame


def build_feature_table(metadata: pd.DataFrame) -> pd.DataFrame:
    selected_batteries = set(TRAIN_BATTERIES + TEST_BATTERIES + CENSORED_BATTERIES)
    tables = [create_battery_feature_rows(metadata, battery_id) for battery_id in sorted(selected_batteries)]
    return pd.concat([table for table in tables if not table.empty], ignore_index=True)


def assign_splits(feature_table: pd.DataFrame) -> pd.DataFrame:
    table = feature_table.copy()
    table["split"] = "unused"

    train_mask = table["battery_id"].isin(TRAIN_BATTERIES) & table["RUL"].notna()
    test_mask = table["battery_id"].isin(TEST_BATTERIES) & table["RUL"].notna()
    censored_mask = table["battery_id"].isin(CENSORED_BATTERIES)

    for battery_id in TRAIN_BATTERIES:
        battery_mask = train_mask & (table["battery_id"] == battery_id)
        battery_rows = table.loc[battery_mask].sort_values("discharge_cycle_index")
        if battery_rows.empty:
            continue
        val_count = max(1, int(np.ceil(len(battery_rows) * VALIDATION_RATIO)))
        val_indices = battery_rows.tail(val_count).index
        train_indices = battery_rows.iloc[:-val_count].index
        table.loc[train_indices, "split"] = "train"
        table.loc[val_indices, "split"] = "val"

    table.loc[test_mask, "split"] = "test"
    table.loc[censored_mask, "split"] = "censored"
    return table


def fit_scaler_from_train(train_frame: pd.DataFrame, full_frame: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler, dict[str, float]]:
    transformed = full_frame.copy()
    for column in FEATURE_COLUMNS:
        transformed[f"raw_{column}"] = transformed[column]
        transformed[column] = transformed[column].astype(float)
    train_impute_values = train_frame[FEATURE_COLUMNS].median().to_dict()

    transformed.loc[:, FEATURE_COLUMNS] = transformed[FEATURE_COLUMNS].fillna(train_impute_values)
    scaler = StandardScaler()
    scaler.fit(transformed.loc[transformed["split"] == "train", FEATURE_COLUMNS])
    scaled_values = scaler.transform(transformed[FEATURE_COLUMNS])
    for index, column in enumerate(FEATURE_COLUMNS):
        transformed[column] = scaled_values[:, index].astype(float)
    return transformed, scaler, {key: float(value) for key, value in train_impute_values.items()}


def split_frames(scaled_frame: pd.DataFrame) -> PipelineArtifacts:
    train_frame = scaled_frame.loc[scaled_frame["split"] == "train"].copy()
    val_frame = scaled_frame.loc[scaled_frame["split"] == "val"].copy()
    test_frame = scaled_frame.loc[scaled_frame["split"] == "test"].copy()
    censored_frame = scaled_frame.loc[scaled_frame["split"] == "censored"].copy()
    artifacts = PipelineArtifacts(
        full_frame=scaled_frame,
        train_frame=train_frame,
        val_frame=val_frame,
        test_frame=test_frame,
        censored_frame=censored_frame,
        feature_columns=FEATURE_COLUMNS,
        scaler=StandardScaler(),
        train_impute_values={},
    )
    return artifacts


def save_artifacts(artifacts: PipelineArtifacts) -> None:
    artifacts.full_frame.to_csv(PROCESSED_DIR / "full.csv", index=False)
    artifacts.train_frame.to_csv(PROCESSED_DIR / "train.csv", index=False)
    artifacts.val_frame.to_csv(PROCESSED_DIR / "val.csv", index=False)
    artifacts.test_frame.to_csv(PROCESSED_DIR / "test.csv", index=False)

    with (PROCESSED_DIR / "feature_columns.json").open("w", encoding="utf-8") as file:
        json.dump(artifacts.feature_columns, file, indent=2)

    scaler_payload = {
        "scaler": artifacts.scaler,
        "feature_columns": artifacts.feature_columns,
        "train_impute_values": artifacts.train_impute_values,
        "failure_threshold": FAILURE_THRESHOLD,
        "split_definition": {
            "train": list(TRAIN_BATTERIES),
            "val": "last 20% of supervised discharge cycles within each training battery",
            "test": list(TEST_BATTERIES),
            "censored": list(CENSORED_BATTERIES),
        },
    }
    with (PROCESSED_DIR / "scaler.pkl").open("wb") as file:
        pickle.dump(scaler_payload, file)


def run_pipeline() -> PipelineArtifacts:
    ensure_directories()
    metadata = load_metadata()
    feature_table = build_feature_table(metadata)
    feature_table = assign_splits(feature_table)

    scaled_frame, scaler, train_impute_values = fit_scaler_from_train(
        train_frame=feature_table.loc[feature_table["split"] == "train"].copy(),
        full_frame=feature_table,
    )

    artifacts = split_frames(scaled_frame)
    artifacts.scaler = scaler
    artifacts.train_impute_values = train_impute_values
    save_artifacts(artifacts)
    return artifacts


def print_summary(artifacts: PipelineArtifacts) -> None:
    counts = artifacts.full_frame.groupby(["battery_id", "split"]).size()
    print("Saved processed artifacts to:", PROCESSED_DIR)
    print("Feature columns:", artifacts.feature_columns)
    print("Split counts:")
    print(counts.to_string())


if __name__ == "__main__":
    artifacts = run_pipeline()
    print_summary(artifacts)
