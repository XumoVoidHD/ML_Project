# Lithium-Ion Battery Remaining Useful Life Prediction

## Project Overview

This project predicts Remaining Useful Life (RUL) for lithium-ion batteries using the NASA Battery Dataset. The pipeline converts cycle-level metadata and per-cycle discharge time series into supervised discharge-cycle features, then compares a row-based XGBoost regressor against sequence models built with PyTorch.

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

RUL is defined at the point where a battery’s usable discharge capacity falls below a threshold, so discharge cycles are the natural supervised unit. Charge and impedance cycles are still useful indirectly because impedance rows provide degradation indicators that can be forward-filled into later discharge rows.

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
- No future cycle statistics are used.
- No full-life aggregate features are used.
- The scaler is fit only on training rows and then applied to validation and test rows.
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

These features capture battery health from multiple physical perspectives:

- Capacity fade reflects aging because the battery stores less charge relative to its early-life baseline.
- Internal resistance (`Re`, `Rct`) increases with degradation and captures electrochemical wear.
- Temperature reflects thermal stress, which accelerates aging and influences observed performance.
- Voltage trajectory points summarize the discharge curve and encode electrochemical health beyond a single scalar capacity value.

## Models

### XGBoost

XGBoost treats each discharge cycle independently using the engineered row-level features. It is strong on small tabular datasets and provides direct feature importance for interpretability.

### LSTM / BiLSTM

The LSTM implementation uses a bidirectional recurrent encoder over consecutive discharge cycles from the same battery. It models temporal context that a row-based method cannot observe directly.

### GRU

The GRU is a lighter recurrent alternative that still captures sequential aging patterns and can train efficiently on limited data.

### Why Sequence Models Can Perform Better

Battery degradation is fundamentally temporal. Sequence models can use recent trajectory information, not just the current cycle snapshot, so they often capture degradation pace and trend changes more effectively than row-based methods.

## Results

After training, each run saves:

- model artifact
- `metrics.json`
- `config.json`
- `predictions.csv`
- plots in `experiments/<model_name>_<timestamp>/plots/`

`experiments/summary.csv` compares all recorded runs. The best model should be chosen by test MAE/RMSE together with qualitative plots such as predicted-vs-actual and residual distributions.

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

```bash
python train.py --model xgboost
python train.py --model lstm
python train.py --model gru
```

You can also provide:

```bash
python train.py --model lstm --seed 42 --sequence_length 10
```

To automatically populate multiple experiment runs from parameter grids:

```bash
python run_sweeps.py --dry-run
python run_sweeps.py --preset recommended --max-runs-per-model 6
python run_sweeps.py --models xgboost random_forest --max-runs-per-model 10
```

### 4. Launch Dashboard

```bash
streamlit run app.py
```

The dashboard includes:

- Model comparison
- Manual XGBoost prediction
- Sequence prediction for LSTM/GRU
- Training trigger UI
- Capacity, RUL, and feature-importance visualizations

## Project Structure

```text
project/
├── app.py
├── data/
├── experiments/
├── models/
│   ├── gru_model.py
│   ├── lstm_model.py
│   └── xgboost_model.py
├── preprocessing/
│   ├── dataset.py
│   └── pipeline.py
├── processed/
├── training/
│   ├── evaluate.py
│   └── trainer.py
├── train.py
├── README.md
├── requirements.txt
└── .gitignore
```
