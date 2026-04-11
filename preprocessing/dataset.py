from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "processed"


def load_feature_columns() -> list[str]:
    with (PROCESSED_DIR / "feature_columns.json").open("r", encoding="utf-8") as file:
        return json.load(file)


def load_processed_split(split: str) -> pd.DataFrame:
    split_path = PROCESSED_DIR / f"{split}.csv"
    if not split_path.exists():
        raise FileNotFoundError(f"Processed split not found: {split_path}")
    return pd.read_csv(split_path)


class BatterySequenceDataset(Dataset[tuple[torch.Tensor, torch.Tensor, str, int]]):
    def __init__(self, frame: pd.DataFrame, feature_columns: list[str], sequence_length: int) -> None:
        self.frame = frame.sort_values(["battery_id", "discharge_cycle_index"]).reset_index(drop=True)
        self.feature_columns = feature_columns
        self.sequence_length = sequence_length
        self.samples: list[tuple[np.ndarray, float, str, int]] = []
        self._build_samples()

    def _build_samples(self) -> None:
        for battery_id, battery_frame in self.frame.groupby("battery_id"):
            battery_frame = battery_frame.sort_values("discharge_cycle_index").reset_index(drop=True)
            if len(battery_frame) < self.sequence_length:
                continue
            features = battery_frame[self.feature_columns].to_numpy(dtype=np.float32)
            targets = battery_frame["RUL"].to_numpy(dtype=np.float32)
            cycle_column = "raw_discharge_cycle_index" if "raw_discharge_cycle_index" in battery_frame.columns else "discharge_cycle_index"
            cycle_indices = battery_frame[cycle_column].to_numpy(dtype=int)
            for end_index in range(self.sequence_length - 1, len(battery_frame)):
                start_index = end_index - self.sequence_length + 1
                feature_window = features[start_index : end_index + 1]
                target_value = float(targets[end_index])
                cycle_index = int(cycle_indices[end_index])
                self.samples.append((feature_window, target_value, battery_id, cycle_index))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str, int]:
        features, target, battery_id, cycle_index = self.samples[index]
        return (
            torch.tensor(features, dtype=torch.float32),
            torch.tensor(target, dtype=torch.float32),
            battery_id,
            cycle_index,
        )


def get_latest_sequence_for_battery(
    frame: pd.DataFrame,
    battery_id: str,
    feature_columns: list[str],
    sequence_length: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    battery_frame = frame.loc[frame["battery_id"] == battery_id].sort_values("discharge_cycle_index").copy()
    if len(battery_frame) < sequence_length:
        raise ValueError(f"Battery {battery_id} has fewer than {sequence_length} cycles available.")
    sequence_frame = battery_frame.tail(sequence_length).copy()
    return sequence_frame[feature_columns].to_numpy(dtype=np.float32), sequence_frame
