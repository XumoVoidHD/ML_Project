# NASA Battery Dataset RUL Prediction

This project predicts Remaining Useful Life (RUL) for lithium-ion batteries using the NASA Ames battery aging dataset. The repo converts raw cycle files into one row per discharge cycle, engineers health indicators, and trains regression models to estimate how many discharge cycles are left before end of life.

The local dataset in this repo is a Kaggle mirror of the NASA battery aging data. The original NASA description says the cells were cycled through charge, discharge, and impedance measurements under controlled temperatures, and testing stopped at a 30% capacity fade from the rated 2 Ah capacity to 1.4 Ah.

## What is in this dataset?

The raw dataset is organized into:

- `data/metadata.csv`: one row per recorded cycle, with cycle type, battery id, file name, capacity, and impedance metadata.
- `data/data/*.csv`: raw time-series files for individual charge, discharge, or impedance cycles.
- `data/combined_by_battery/*.csv`: battery-level merged tables created by `combine_data_by_battery.py`.
- `data/preprocessed_rul/*.csv`: cycle-level machine learning tables created by `preprocess_rul.py`.

## Project structure

The repo is now organized like this:

- `rul/`: reusable project modules for preprocessing, prediction, utilities, and plotting
- `scripts/training/`: individual model training scripts
- `scripts/data/`: data download helpers
- `notebooks/`: exploratory notebooks
- `docs/`: planning notes such as the dashboard checklist
- `data/`, `artifacts/`, `figures/`: generated datasets, saved models, and plots

The original root commands are still available through small wrapper files, so commands like `python predict_rul.py` still work.

In the full metadata table there are:

- 34 batteries
- 2,815 charge cycles
- 2,794 discharge cycles
- 1,956 impedance cycles
- ambient temperatures: `4`, `22`, `24`, `43`, `44` degrees C

This RUL workflow uses four batteries:

| Battery | Total rows | Discharge cycles | Charge cycles | Impedance cycles | Start capacity (Ah) | Final capacity (Ah) | EOL cycle index used here |
|---|---:|---:|---:|---:|---:|---:|---:|
| B0005 | 616 | 168 | 170 | 278 | 1.8565 | 1.3251 | 124 |
| B0006 | 616 | 168 | 170 | 278 | 2.0353 | 1.1857 | 108 |
| B0007 | 616 | 168 | 170 | 278 | 1.8911 | 1.4325 | 167 |
| B0018 | 319 | 132 | 134 | 53 | 1.8550 | 1.3411 | 96 |

Notes:

- `B0005`, `B0006`, and `B0007` are the training batteries.
- `B0018` is the held-out test battery.
- `B0007` never drops below `1.4 Ah` in the available metadata, so this repo treats its final observed discharge cycle as the effective EOL for labeling.

## Raw cycle data

Each raw CSV contains a time series for one cycle.

Discharge files contain columns like:

- `Voltage_measured`
- `Current_measured`
- `Temperature_measured`
- `Current_load`
- `Voltage_load`
- `Time`

Charge files contain:

- `Voltage_measured`
- `Current_measured`
- `Temperature_measured`
- `Current_charge`
- `Voltage_charge`
- `Time`

Impedance files contain fields related to electrochemical impedance, including:

- `Sense_current`
- `Battery_current`
- `Current_ratio`
- `Battery_impedance`
- `Rectified_Impedance`

The metadata table also stores:

- `Capacity`: discharge capacity in Ah
- `Re`: electrolyte resistance
- `Rct`: charge-transfer resistance

## Scientific meaning of the main variables

These are the important battery-health quantities used in the repo.

### 1. Capacity

Capacity is the amount of charge the battery can deliver during discharge.

Engineering relation:

```text
Q = integral(I(t) dt)
Capacity_Ah = (1 / 3600) * integral(I(t) dt)
```

Where:

- `Q` is charge
- `I(t)` is current
- dividing by `3600` converts coulombs to ampere-hours

As a lithium-ion battery ages, capacity typically decreases.

### 2. State of health / capacity fade

The repo uses normalized capacity as a degradation indicator:

```text
capacity_fade_i = Capacity_i / mean(Capacity_0 ... Capacity_4)
```

This is a simple State of Health style feature:

```text
SOH_i ~= Capacity_i / Capacity_initial
```

### 3. Capacity derivative

This captures the local degradation rate from one discharge cycle to the next:

```text
capacity_derivative_i = Capacity_i - Capacity_(i-1)
```

More negative values usually mean faster degradation.

### 4. End of life (EOL)

NASA defines end of life at 30% fade from rated capacity:

```text
Capacity_EOL = 0.7 * 2.0 Ah = 1.4 Ah
```

In this repo:

```text
EOL cycle = first discharge cycle where Capacity < 1.4 Ah
```

If no such cycle exists in the observed data, the last discharge cycle is used.

### 5. Remaining Useful Life (RUL)

The prediction target is:

```text
RUL_i = cycle_EOL - cycle_i
```

and then clipped at zero:

```text
RUL_i = max(0, cycle_EOL - cycle_i)
```

This means a battery near failure has small RUL, while a healthy early-life cycle has large RUL.

### 6. Internal resistance and impedance features

The repo uses two impedance-derived health markers:

- `Re`: electrolyte resistance
- `Rct`: charge-transfer resistance

At a high level, terminal voltage is affected by internal resistance:

```text
V_terminal(t) ~= OCV(SOC, T) - I(t) * R_internal - eta(t)
```

Where:

- `OCV` is open-circuit voltage
- `SOC` is state of charge
- `T` is temperature
- `eta(t)` represents additional electrochemical overpotentials

As batteries age, `Re` and `Rct` often increase, and usable capacity and power capability tend to decrease.

If you want a simple empirical rule from this dataset: higher `Re` and `Rct` generally correspond to lower RUL.

### 7. Discharge curve shape

The repo extracts voltage at fixed times during discharge:

```text
V_i(100s), V_i(300s), V_i(600s)
```

For each target time `tau`, the code picks the nearest recorded time sample:

```text
t_star = argmin_t |Time(t) - tau|
V_i(tau) = Voltage_measured(t_star)
```

These values summarize the shape of the discharge curve. As a cell degrades, voltage often drops faster.

### 8. Discharge duration

This is computed as:

```text
discharge_duration_i = max(Time_i)
```

Longer discharge duration usually indicates the battery can sustain load longer, which is generally associated with higher capacity and higher RUL.

### 9. Average temperature

The repo also uses:

```text
avg_temperature_i = (1 / N_i) * sum_k Temperature_i,k
```

Temperature matters because battery kinetics and aging both depend strongly on thermal conditions.

## Empirical relationships in this repo's processed data

Using the existing processed table in `data/preprocessed_rul/rul_preprocessed.csv`, the strongest linear correlations with `RUL` are:

| Variable | Correlation with RUL |
|---|---:|
| `Capacity` | `+0.913` |
| `capacity_fade` | `+0.898` |
| `discharge_duration` | `+0.886` |
| `voltage_at_600s` | `+0.831` |
| `discharge_cycle_index` | `-0.834` |
| `Re` | `-0.674` |
| `Rct` | `-0.704` |

Interpretation:

- higher capacity usually means more cycles remaining
- longer discharge duration usually means more cycles remaining
- higher impedance usually means fewer cycles remaining
- later cycle index usually means fewer cycles remaining

These are empirical correlations from this processed subset, not universal battery laws.

## How preprocessing works in this repo

`preprocess_rul.py` converts raw cycle files into a supervised learning table.

Step by step:

1. Filter metadata to batteries `B0005`, `B0006`, `B0007`, `B0018`.
2. Keep discharge cycles as modeling examples.
3. Convert `Capacity` to numeric.
4. Compute EOL and label every discharge cycle with `RUL`.
5. Remove very low-capacity outliers after labeling:

```text
keep if Capacity >= 0.5 Ah
```

6. Forward-fill `Re` and `Rct` from the most recent earlier impedance cycle.
7. Read each discharge CSV and extract:
   - `discharge_duration`
   - `avg_temperature`
   - `voltage_at_100s`
   - `voltage_at_300s`
   - `voltage_at_600s`
8. Add derived features:
   - `capacity_fade`
   - `capacity_derivative`
9. Split by battery:
   - train: `B0005`, `B0006`, `B0007`
   - test: `B0018`
10. Standardize selected features using train-only statistics to avoid leakage.

Important detail: in the saved preprocessed files, some columns such as `Re`, `Rct`, `discharge_duration`, `avg_temperature`, and fixed-time voltages are already standardized. That is why values in `rul_preprocessed.csv` may look different from the raw units.

## Processed feature set

The final table contains:

| Column | Meaning |
|---|---|
| `Capacity` | discharge capacity in Ah |
| `capacity_fade` | capacity divided by baseline capacity |
| `capacity_derivative` | cycle-to-cycle capacity change |
| `Re` | standardized electrolyte resistance |
| `Rct` | standardized charge-transfer resistance |
| `discharge_duration` | standardized discharge length |
| `avg_temperature` | standardized mean discharge temperature |
| `voltage_at_100s` | standardized voltage near 100 s |
| `voltage_at_300s` | standardized voltage near 300 s |
| `voltage_at_600s` | standardized voltage near 600 s |
| `ambient_temperature` | chamber temperature from metadata |
| `discharge_cycle_index` | 0-based discharge cycle counter |
| `Capacity_normalized` | per-battery z-score of capacity |
| `capacity_fade_normalized` | per-battery z-score of normalized capacity |
| `RUL` | target label in cycles |

## Models in this repo

The repo already includes:

- `scripts/training/train_rul_baseline.py`: Random Forest baseline
- `scripts/training/train_rul_gradient_boosting.py`: Gradient Boosting regressor
- `scripts/training/train_rul_extended_features.py`: Random Forest with extended features
- `scripts/training/train_rul_ensemble.py`: ensemble regressor
- `scripts/training/train_rul_tuned.py`: grid-searched Random Forest
- `scripts/training/train_rul_xgboost.py`: XGBoost regressor

## New end-to-end prediction script

This repo now includes `predict_rul.py`, which gives a single command to train and export predictions.

Default behavior:

- loads the preprocessed train and test files
- trains a Random Forest on the extended feature set
- predicts RUL on the held-out battery `B0018`
- saves model, metrics, and prediction CSVs in `artifacts/`

Run it with:

```bash
python predict_rul.py
```

Optional arguments:

```bash
python predict_rul.py --model random_forest
python predict_rul.py --model gradient_boosting
python predict_rul.py --rebuild-preprocessed
```

Saved artifacts:

- `artifacts/rul_random_forest.pkl`
- `artifacts/rul_random_forest_metrics.json`
- `artifacts/rul_random_forest_train_predictions.csv`
- `artifacts/rul_random_forest_test_predictions.csv`

## Recommended workflow

If you want to reproduce everything from raw data:

```bash
python combine_data_by_battery.py
python preprocess_rul.py
python predict_rul.py
```

If you want to compare several models:

```bash
python run_all_rul_models.py
```

You can also run a specific training module directly:

```bash
python -m scripts.training.train_rul_baseline
python -m scripts.training.train_rul_gradient_boosting
```

## Install

```bash
pip install -r requirements.txt
```

## Source notes

Primary dataset description:

- NASA PCoE battery aging dataset repository: https://c3.ndc.nasa.gov/dashlink/resources/133
- NASA PCoE data repository overview: https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/

Local acquisition helper:

- `scripts/data/download_data.py` downloads a Kaggle-hosted mirror via `kagglehub`.
