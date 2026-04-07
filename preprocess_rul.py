import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler

# Paths
DATA_DIR = Path(__file__).parent / "data"
METADATA_PATH = DATA_DIR / "metadata.csv"
CYCLES_DIR = DATA_DIR / "data"
OUTPUT_DIR = DATA_DIR / "preprocessed_rul"

# Config
BATTERY_IDS = ["B0005", "B0006", "B0007", "B0018"]
EOL_CAPACITY = 1.4  # 30% fade from 2 Ahr
MIN_CAPACITY_OUTLIER = 0.5
INITIAL_CYCLES_FOR_BASELINE = 5


def ensure_filename(filename) -> str:
    """Ensure filename has correct format (e.g., 04506.csv)."""
    s = str(filename).strip()
    if s.endswith(".csv"):
        base = s[:-4]
        return f"{base.zfill(5)}.csv"
    return f"{s.zfill(5)}.csv"


def extract_discharge_features(filepath: Path) -> dict:
    """Extract features from a discharge cycle CSV."""
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        return {"discharge_duration": np.nan, "avg_temperature": np.nan, "voltage_at_100s": np.nan}

    if df.empty or "Time" not in df.columns:
        return {"discharge_duration": np.nan, "avg_temperature": np.nan, "voltage_at_100s": np.nan}

    features = {}

    # Discharge duration (total time)
    features["discharge_duration"] = df["Time"].max() if "Time" in df.columns else np.nan

    # Average temperature during discharge
    features["avg_temperature"] = df["Temperature_measured"].mean() if "Temperature_measured" in df.columns else np.nan

    # Voltage at fixed time points (curve shape indicators)
    if "Voltage_measured" in df.columns and "Time" in df.columns:
        max_time = df["Time"].max()
        for t in [100, 300, 600]:
            if max_time >= t:
                idx = (df["Time"] - t).abs().idxmin()
                features[f"voltage_at_{t}s"] = df.loc[idx, "Voltage_measured"]
            else:
                features[f"voltage_at_{t}s"] = np.nan
    else:
        features["voltage_at_100s"] = np.nan
        features["voltage_at_300s"] = np.nan
        features["voltage_at_600s"] = np.nan

    return features


def process_battery(battery_id: str, metadata: pd.DataFrame) -> pd.DataFrame:
    """Process one battery: compute RUL, extract features, forward-fill impedance."""
    battery_meta = metadata[metadata["battery_id"] == battery_id].copy()
    battery_meta = battery_meta.sort_values("uid")

    # Get discharge cycles in order
    discharge_meta = battery_meta[battery_meta["type"] == "discharge"].copy()
    discharge_meta = discharge_meta.reset_index(drop=True)
    discharge_meta["original_cycle_index"] = np.arange(len(discharge_meta))

    # Parse capacity (handle numeric)
    discharge_meta["Capacity"] = pd.to_numeric(discharge_meta["Capacity"], errors="coerce")

    # EOL: compute FIRST on raw data (before outlier filtering)
    # First discharge cycle where capacity drops below threshold
    eol_rows = discharge_meta[discharge_meta["Capacity"] < EOL_CAPACITY]
    if len(eol_rows) > 0:
        eol_cycle_index = eol_rows.iloc[0]["original_cycle_index"]
    else:
        eol_cycle_index = discharge_meta["original_cycle_index"].max()

    # Compute RUL using original cycle indices (before filtering)
    discharge_meta["RUL"] = eol_cycle_index - discharge_meta["original_cycle_index"]
    discharge_meta["RUL"] = discharge_meta["RUL"].clip(lower=0)

    # NOW filter outlier cycles (very low capacity) for modeling
    discharge_meta = discharge_meta[
        (discharge_meta["Capacity"] >= MIN_CAPACITY_OUTLIER) | (discharge_meta["Capacity"].isna())
    ].copy()

    # Re-index after filtering (for display/counting)
    discharge_meta = discharge_meta.reset_index(drop=True)
    discharge_meta["discharge_cycle_index"] = np.arange(len(discharge_meta))

    # Forward-fill Re and Rct from impedance cycles (most recent impedance before each discharge)
    impedance_meta = battery_meta[battery_meta["type"] == "impedance"][["uid", "Re", "Rct"]].copy()
    impedance_meta["Re"] = pd.to_numeric(impedance_meta["Re"], errors="coerce")
    impedance_meta["Rct"] = pd.to_numeric(impedance_meta["Rct"], errors="coerce")

    re_values = []
    rct_values = []
    for _, row in discharge_meta.iterrows():
        d_uid = row["uid"]
        imp_before = impedance_meta[impedance_meta["uid"] <= d_uid]
        if len(imp_before) > 0:
            last = imp_before.iloc[-1]
            re_values.append(last["Re"])
            rct_values.append(last["Rct"])
        else:
            re_values.append(np.nan)
            rct_values.append(np.nan)

    discharge_meta["Re"] = re_values
    discharge_meta["Rct"] = rct_values

    # Extract discharge curve features
    curve_features_list = []
    for _, row in discharge_meta.iterrows():
        filename = ensure_filename(row["filename"])
        filepath = CYCLES_DIR / filename
        curve_feats = extract_discharge_features(filepath)
        curve_features_list.append(curve_feats)

    curve_df = pd.DataFrame(curve_features_list)
    discharge_meta = pd.concat([discharge_meta.reset_index(drop=True), curve_df], axis=1)

    # Derived features: capacity fade, capacity derivative
    initial_capacity = discharge_meta["Capacity"].iloc[:INITIAL_CYCLES_FOR_BASELINE].mean()
    discharge_meta["capacity_fade"] = discharge_meta["Capacity"] / initial_capacity
    discharge_meta["capacity_derivative"] = discharge_meta["Capacity"].diff()
    discharge_meta["capacity_derivative"] = discharge_meta["capacity_derivative"].fillna(0)

    return discharge_meta


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(METADATA_PATH)
    metadata = metadata[metadata["battery_id"].isin(BATTERY_IDS)]

    all_batteries = []
    for battery_id in BATTERY_IDS:
        df = process_battery(battery_id, metadata)
        if not df.empty:
            all_batteries.append(df)
            print(f"{battery_id}: {len(df)} discharge cycles, RUL range 0-{df['RUL'].max()}")

    if not all_batteries:
        print("No data processed.")
        return

    combined = pd.concat(all_batteries, ignore_index=True)

    # Feature columns for modeling (exclude metadata)
    feature_cols = [
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
        "ambient_temperature",
        "discharge_cycle_index",
    ]

    # Drop rows with too many missing features
    combined = combined.dropna(subset=["Capacity", "RUL"])
    for col in feature_cols:
        if col not in combined.columns:
            combined[col] = np.nan

    # Train/test split by battery FIRST (before scaling to avoid leakage)
    train_mask = combined["battery_id"].isin(["B0005", "B0006", "B0007"])
    test_mask = combined["battery_id"] == "B0018"
    train_df = combined[train_mask].copy()
    test_df = combined[test_mask].copy()

    # Normalize features (per-battery for capacity-related)
    combined["Capacity_normalized"] = np.nan
    combined["capacity_fade_normalized"] = np.nan
    for battery_id in BATTERY_IDS:
        mask = combined["battery_id"] == battery_id
        b_data = combined.loc[mask]
        cap_mean = b_data["Capacity"].mean()
        cap_std = b_data["Capacity"].std()
        fade_mean = b_data["capacity_fade"].mean()
        fade_std = b_data["capacity_fade"].std()

        if cap_std > 0:
            combined.loc[mask, "Capacity_normalized"] = (combined.loc[mask, "Capacity"] - cap_mean) / cap_std
        if fade_std > 0:
            combined.loc[mask, "capacity_fade_normalized"] = (
                combined.loc[mask, "capacity_fade"] - fade_mean
            ) / fade_std

    # Update train/test with normalized capacity columns
    train_df["Capacity_normalized"] = combined.loc[train_mask, "Capacity_normalized"].values
    train_df["capacity_fade_normalized"] = combined.loc[train_mask, "capacity_fade_normalized"].values
    test_df["Capacity_normalized"] = combined.loc[test_mask, "Capacity_normalized"].values
    test_df["capacity_fade_normalized"] = combined.loc[test_mask, "capacity_fade_normalized"].values

    # Global scaling: fit on TRAIN only to avoid data leakage
    scale_cols = ["Re", "Rct", "discharge_duration", "avg_temperature", "voltage_at_100s", "voltage_at_300s", "voltage_at_600s"]
    scale_cols = [c for c in scale_cols if c in combined.columns]
    if scale_cols:
        train_medians = train_df[scale_cols].median()
        train_df[scale_cols] = train_df[scale_cols].fillna(train_medians)
        test_df[scale_cols] = test_df[scale_cols].fillna(train_medians)  # use train medians for test

        scaler = StandardScaler()
        scaler.fit(train_df[scale_cols])
        train_df[scale_cols] = scaler.transform(train_df[scale_cols])
        test_df[scale_cols] = scaler.transform(test_df[scale_cols])

    # Rebuild combined from scaled train + test for full output
    combined = pd.concat([train_df, test_df], ignore_index=True)

    # Save
    output_path = OUTPUT_DIR / "rul_preprocessed.csv"
    combined.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path} ({len(combined)} rows)")

    train_df.to_csv(OUTPUT_DIR / "rul_train.csv", index=False)
    test_df.to_csv(OUTPUT_DIR / "rul_test.csv", index=False)
    print(f"Train: {len(train_df)} rows (B0005, B0006, B0007)")
    print(f"Test: {len(test_df)} rows (B0018)")

    # Generate preprocessing visualizations
    from preprocess_visualize import generate_all as generate_preprocess_plots
    generate_preprocess_plots()


if __name__ == "__main__":
    main()
