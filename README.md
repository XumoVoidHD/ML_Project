# Lithium-Ion Battery Remaining Useful Life Prediction

## Project Overview

This project predicts Remaining Useful Life (RUL) for lithium-ion batteries using the NASA Battery Dataset. The pipeline converts cycle-level metadata and per-cycle discharge time series into supervised discharge-cycle features, then compares both row-based tabular regressors and sequence models built with PyTorch.

The current model families in the codebase are:

- Linear Regression
- Ridge Regression
- Random Forest
- XGBoost
- SVR
- MLP
- LSTM
- GRU

## Dataset Explanation

The NASA battery dataset tracks repeated charge, discharge, and impedance cycles for several batteries under controlled laboratory conditions. For this project, the target batteries are:

- Train: `B0005`, `B0006`
- Test: `B0018`
- Censored analysis only: `B0007`

`B0007` never drops below the end-of-life threshold in the provided data, so it is treated as censored and excluded from supervised target training.

## Data Structure Explanation

- `data/metadata.csv`
  - One row per cycle event.
  - Includes `battery_id`, cycle `type`, `filename`, and summary values such as `Capacity`, `Re`, and `Rct`.
- `data/data/`
  - Per-cycle CSV files referenced by `metadata.csv`.
  - Discharge files contain measurements such as `Time`, `Voltage_measured`, and `Temperature_measured`.

## Preprocessing Strategy

### Why Only Discharge Cycles

RUL is defined at the point where a battery's usable discharge capacity falls below a threshold, so discharge cycles are the natural supervised unit. Charge and impedance cycles are still useful indirectly because impedance rows provide degradation indicators that can be forward-filled into later discharge rows.

### Why This RUL Definition

RUL is defined as the number of remaining discharge cycles until `Capacity < 1.4 Ah`. This creates a physically meaningful end-of-life target tied to a clear operational threshold. The threshold-crossing discharge cycle has `RUL = 0`.

### Why Battery-Wise Split

The split is fixed to avoid cross-battery leakage:

- Train: `B0005`, `B0006`
- Validation: last 20% of supervised discharge cycles inside each training battery
- Test: `B0018`
- Censored only: `B0007`

The test battery is never used during fitting, validation, scaling, or early stopping.

### Leakage Avoidance

The preprocessing pipeline avoids leakage in several ways:

- Only discharge cycles are used as supervised samples.
- `Re` and `Rct` are forward-filled only from prior impedance cycles in the same battery.
- Missing feature values are imputed using training-set medians only.
- No future cycle statistics are used.
- No full-life aggregate features are used.
- The scaler is fit only on training rows and then applied to validation, test, and censored rows.
- Sequence windows are built from consecutive cycles of the same battery only.

## Feature Engineering

Each supervised row represents one discharge cycle and includes:

- `Capacity`
- `capacity_fade`
- `capacity_derivative`
- `Re`, `Rct`
- `discharge_duration`
- `avg_temperature`
- `voltage_at_100s`, `voltage_at_300s`, `voltage_at_600s`
- `discharge_cycle_index`

The processed tables also keep raw unscaled copies of these features in `raw_*` columns for inspection and dashboard use.

These features capture battery health from multiple physical perspectives:

- Capacity fade reflects aging because the battery stores less charge relative to its early-life baseline.
- Internal resistance (`Re`, `Rct`) increases with degradation and captures electrochemical wear.
- Temperature reflects thermal stress, which accelerates aging and influences observed performance.
- Voltage trajectory points summarize the discharge curve and encode electrochemical health beyond a single scalar capacity value.

## Models

### Tabular Models

- Linear Regression: simplest baseline for a direct linear relationship.
- Ridge Regression: regularized linear model that often generalizes better on small noisy data.
- Random Forest: tree ensemble for nonlinear interactions.
- XGBoost: boosted tree model with strong tabular performance and feature importance.
- SVR: nonlinear margin-based regressor with optional post-hoc calibration on the validation split.
- MLP: feed-forward neural baseline for tabular features.

### Sequence Models

- LSTM: bidirectional recurrent encoder over consecutive discharge cycles from the same battery.
- GRU: lighter recurrent alternative for temporal degradation patterns.

### Why Sequence Models May Help

Battery degradation is fundamentally temporal. Sequence models can use recent trajectory information, not just the current cycle snapshot, so they may capture degradation pace and trend changes more effectively than row-based methods.

## Current Results Snapshot

Based on the current `experiments/summary.csv`, the strongest recorded runs by test MAE are:

- XGBoost: `xgboost_20260411_095234` with test MAE `10.90`, test RMSE `12.94`, test R2 `0.786`
- Ridge: `ridge_20260409_214544` with test MAE `11.23`, test RMSE `12.26`, test R2 `0.808`
- LSTM: `lstm_20260411_113957` with test MAE `11.28`, test RMSE `13.57`, test R2 `0.744`
- Random Forest: `random_forest_20260411_113657` with test MAE `11.64`, test RMSE `13.97`, test R2 `0.751`

`experiments/summary.csv` compares all recorded runs. The best model should be chosen by unseen-battery test performance together with qualitative plots such as predicted-vs-actual and residual distributions.

## How To Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Preprocess Data

```bash
python preprocessing/pipeline.py
```

This generates:

- `processed/full.csv`
- `processed/train.csv`
- `processed/val.csv`
- `processed/test.csv`
- `processed/feature_columns.json`
- `processed/scaler.pkl`

### 3. Train Models

Examples:

```bash
python train.py --model xgboost
python train.py --model ridge --ridge_alpha 50
python train.py --model random_forest --rf_n_estimators 600 --rf_max_depth 8 --rf_min_samples_leaf 6
python train.py --model lstm --sequence_length 5 --hidden_dim 32 --num_layers 1
python train.py --model gru --sequence_length 8
python train.py --model svr --svr_c 10 --svr_epsilon 0.1 --svr_kernel rbf
python train.py --model mlp --mlp_hidden_dim 64
```

To automatically populate multiple experiment runs from parameter grids:

```bash
python run_sweeps.py --dry-run
python run_sweeps.py --preset recommended --max-runs-per-model 6
python run_sweeps.py --preset focused --max-runs-per-model 8
python run_sweeps.py --models xgboost random_forest ridge lstm --max-runs-per-model 10
```

### 4. Launch Dashboard

```bash
streamlit run app.py
```

The dashboard includes:

- Model comparison ranked by test MAE and test RMSE
- Manual what-if prediction for row-based models
- Sequence prediction from existing battery data or uploaded CSV
- Local feature-influence summaries for row models
- Simple future-trend forecasting for manual row inputs
- Training trigger UI
- Raw battery degradation plots and experiment diagnostics

## Project Structure

```text
project/
|-- app.py
|-- prediction_insights.py
|-- run_sweeps.py
|-- train.py
|-- data/
|-- experiments/
|-- models/
|   |-- baseline_models.py
|   |-- gru_model.py
|   |-- lstm_model.py
|   `-- xgboost_model.py
|-- preprocessing/
|   |-- dataset.py
|   `-- pipeline.py
|-- processed/
|-- training/
|   |-- evaluate.py
|   `-- trainer.py
|-- README.md
|-- requirements.txt
`-- .gitignore
```
