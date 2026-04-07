import pandas as pd

from rul.paths import COMBINED_DATA_DIR, DATA_DIR, RAW_CYCLES_DIR

# Paths
METADATA_PATH = DATA_DIR / "metadata.csv"
CYCLES_DIR = RAW_CYCLES_DIR
OUTPUT_DIR = COMBINED_DATA_DIR

# Column renames by cycle type
CHARGE_RENAMES = {
    "Voltage_measured": "Voltage_measured(charge)",
    "Current_measured": "Current_measured(charge)",
    "Temperature_measured": "Temperature_measured(charge)",
    "Current_charge": "Current_charge(charge)",
    "Voltage_charge": "Voltage_charge(charge)",
    "Time": "Time_charge",
}

DISCHARGE_RENAMES = {
    "Voltage_measured": "Voltage_measured(discharge)",
    "Current_measured": "Current_measured(discharge)",
    "Temperature_measured": "Temperature_measured(discharge)",
    "Current_load": "Current_load(discharge)",
    "Voltage_load": "Voltage_load(discharge)",
    "Time": "Time_discharge",
}

# All possible columns in final output (for consistent schema)
ALL_COLUMNS = [
    # Metadata columns
    "type",
    "start_time",
    "ambient_temperature",
    "battery_id",
    "test_id",
    "uid",
    "Capacity",
    "Re",
    "Rct",
    # Charge columns
    "Voltage_measured(charge)",
    "Current_measured(charge)",
    "Temperature_measured(charge)",
    "Current_charge(charge)",
    "Voltage_charge(charge)",
    "Time_charge",
    # Discharge columns
    "Voltage_measured(discharge)",
    "Current_measured(discharge)",
    "Temperature_measured(discharge)",
    "Current_load(discharge)",
    "Voltage_load(discharge)",
    "Time_discharge",
    # Impedance columns
    "Sense_current",
    "Battery_current",
    "Current_ratio",
    "Battery_impedance",
    "Rectified_Impedance",
]


def load_cycle_data(filename: str, cycle_type: str) -> pd.DataFrame:
    """Load a cycle CSV and rename columns based on type."""
    filepath = CYCLES_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(f"Cycle file not found: {filepath}")

    df = pd.read_csv(filepath)

    if cycle_type == "charge":
        df = df.rename(columns=CHARGE_RENAMES)
    elif cycle_type == "discharge":
        df = df.rename(columns=DISCHARGE_RENAMES)
    # impedance: no renames needed

    return df


def combine_battery_data(battery_id: str, metadata_rows: pd.DataFrame) -> pd.DataFrame:
    """Combine all cycles for a battery into one DataFrame."""
    all_dfs = []

    for _, row in metadata_rows.iterrows():
        cycle_type = row["type"]
        filename = str(row["filename"]).strip()
        metadata = {
            "type": cycle_type,
            "start_time": row["start_time"],
            "ambient_temperature": row["ambient_temperature"],
            "battery_id": row["battery_id"],
            "test_id": row["test_id"],
            "uid": row["uid"],
            "Capacity": row["Capacity"],
            "Re": row["Re"],
            "Rct": row["Rct"],
        }

        try:
            cycle_df = load_cycle_data(filename, cycle_type)
        except FileNotFoundError as e:
            print(f"Warning: Skipping {filename} - {e}")
            continue

        # Add metadata columns to every row
        for key, val in metadata.items():
            cycle_df[key] = val

        all_dfs.append(cycle_df)

    if not all_dfs:
        return pd.DataFrame(columns=ALL_COLUMNS)

    combined = pd.concat(all_dfs, ignore_index=True)

    # Reorder columns: metadata first, then type-specific (fill missing with NaN)
    for col in ALL_COLUMNS:
        if col not in combined.columns:
            combined[col] = pd.NA

    combined = combined[ALL_COLUMNS]

    return combined


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(METADATA_PATH)

    # Group by battery_id
    battery_ids = metadata["battery_id"].unique()

    for battery_id in battery_ids:
        battery_meta = metadata[metadata["battery_id"] == battery_id]
        print(f"Processing {battery_id} ({len(battery_meta)} cycles)...")

        combined = combine_battery_data(battery_id, battery_meta)

        output_path = OUTPUT_DIR / f"{battery_id}.csv"
        combined.to_csv(output_path, index=False)
        print(f"  Saved to {output_path} ({len(combined)} rows)")

    print(f"\nDone. Output saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
