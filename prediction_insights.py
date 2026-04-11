from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


HEALTHY_THRESHOLD = 60.0
WARNING_THRESHOLD = 20.0

FEATURE_DIRECTION = {
    "Capacity": 1,
    "capacity_fade": -1,
    "capacity_derivative": 1,
    "Re": -1,
    "Rct": -1,
    "discharge_duration": 1,
    "avg_temperature": -1,
    "voltage_at_100s": 1,
    "voltage_at_300s": 1,
    "voltage_at_600s": 1,
    "discharge_cycle_index": -1,
}

FEATURE_LABELS = {
    "Capacity": "available capacity",
    "capacity_fade": "capacity fade",
    "capacity_derivative": "capacity change from the previous cycle",
    "Re": "electrolyte resistance",
    "Rct": "charge-transfer resistance",
    "discharge_duration": "discharge duration",
    "avg_temperature": "average temperature",
    "voltage_at_100s": "voltage at 100 seconds",
    "voltage_at_300s": "voltage at 300 seconds",
    "voltage_at_600s": "voltage at 600 seconds",
    "discharge_cycle_index": "life-cycle index",
}


@dataclass(frozen=True)
class PredictionAssessment:
    health_status: str
    recommendation: str


def classify_health_status(predicted_rul: float) -> str:
    if predicted_rul > HEALTHY_THRESHOLD:
        return "Healthy"
    if predicted_rul > WARNING_THRESHOLD:
        return "Warning"
    return "Critical"


def generate_maintenance_recommendation(predicted_rul: float) -> str:
    if predicted_rul > HEALTHY_THRESHOLD:
        return "Battery is healthy. Continue normal operation and routine monitoring."
    if predicted_rul > WARNING_THRESHOLD:
        return "Battery is aging. Schedule inspection or replacement planning soon."
    return "Battery is near end of life. Prioritize replacement before the next demanding mission."


def assess_prediction(predicted_rul: float) -> PredictionAssessment:
    return PredictionAssessment(
        health_status=classify_health_status(predicted_rul),
        recommendation=generate_maintenance_recommendation(predicted_rul),
    )


def _safe_scale(value: float, scale: float) -> float:
    if not np.isfinite(scale) or abs(scale) < 1e-9:
        return 0.0
    return float(value / scale)


def summarize_feature_influence(
    feature_columns: list[str],
    raw_inputs: dict[str, float],
    scaled_values: np.ndarray,
    model: Any,
    scaler_payload: dict[str, Any],
    top_k: int = 3,
) -> list[dict[str, Any]]:
    train_impute_values: dict[str, float] = scaler_payload["train_impute_values"]
    scaler = scaler_payload["scaler"]
    importances = np.asarray(getattr(model, "feature_importances_", np.ones(len(feature_columns))), dtype=float)
    if importances.shape[0] != len(feature_columns):
        importances = np.ones(len(feature_columns), dtype=float)

    local_scores: list[dict[str, Any]] = []
    for index, feature_name in enumerate(feature_columns):
        raw_value = float(raw_inputs[feature_name])
        baseline = float(train_impute_values[feature_name])
        scaled_value = float(scaled_values[index])
        importance = float(importances[index])
        direction = FEATURE_DIRECTION.get(feature_name, 1)
        signed_effect = direction * np.sign(scaled_value)
        local_strength = abs(scaled_value) * max(importance, 1e-6)

        if raw_value > baseline:
            relative_state = "above"
        elif raw_value < baseline:
            relative_state = "below"
        else:
            relative_state = "near"

        if signed_effect > 0:
            impact_phrase = "supports a longer predicted life"
        elif signed_effect < 0:
            impact_phrase = "pushes the prediction toward a shorter life"
        else:
            impact_phrase = "has a neutral effect in this case"

        z_score = _safe_scale(raw_value - baseline, float(scaler.scale_[index]))
        local_scores.append(
            {
                "feature": feature_name,
                "label": FEATURE_LABELS.get(feature_name, feature_name),
                "raw_value": raw_value,
                "baseline": baseline,
                "relative_state": relative_state,
                "approx_z_score": z_score,
                "importance": importance,
                "local_strength": float(local_strength),
                "impact_phrase": impact_phrase,
            }
        )

    local_scores.sort(key=lambda item: item["local_strength"], reverse=True)
    return local_scores[:top_k]


def build_forecast_frame(
    feature_columns: list[str],
    raw_inputs: dict[str, float],
    model: Any,
    scaler_payload: dict[str, Any],
    forecast_horizon: int = 15,
) -> pd.DataFrame:
    scaler = scaler_payload["scaler"]
    current_state = {column: float(raw_inputs[column]) for column in feature_columns}
    current_capacity = current_state["Capacity"]
    fade_ratio = float(current_state["capacity_fade"])
    if fade_ratio < 0.99:
        estimated_initial_capacity = current_capacity / max(1e-6, 1.0 - fade_ratio)
    else:
        estimated_initial_capacity = current_capacity

    degradation_step = current_state["capacity_derivative"]
    if degradation_step >= -0.002:
        degradation_step = -0.005

    records: list[dict[str, Any]] = []
    for step in range(forecast_horizon + 1):
        input_frame = pd.DataFrame([current_state])[feature_columns]
        scaled_frame = scaler.transform(input_frame)
        predicted_rul = float(model.predict(scaled_frame)[0])
        assessment = assess_prediction(predicted_rul)
        records.append(
            {
                "forecast_step": step,
                "discharge_cycle_index": int(round(current_state["discharge_cycle_index"])),
                "estimated_capacity": current_state["Capacity"],
                "predicted_rul": predicted_rul,
                "health_status": assessment.health_status,
            }
        )

        current_state["discharge_cycle_index"] += 1.0
        current_state["Capacity"] = max(0.0, current_state["Capacity"] + degradation_step)
        current_state["capacity_derivative"] = degradation_step
        current_state["capacity_fade"] = max(
            0.0,
            1.0 - (current_state["Capacity"] / max(estimated_initial_capacity, 1e-6)),
        )
        current_state["Re"] += 0.00005
        current_state["Rct"] += 0.00008
        current_state["discharge_duration"] = max(0.0, current_state["discharge_duration"] * 0.997)
        current_state["avg_temperature"] += 0.015
        current_state["voltage_at_100s"] -= 0.0015
        current_state["voltage_at_300s"] -= 0.0018
        current_state["voltage_at_600s"] -= 0.002
        degradation_step *= 1.01

    return pd.DataFrame(records)
