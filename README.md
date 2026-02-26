# Remaining Useful Life (RUL) Prediction – Project Explanation

## 1. Project Objective

The goal of this project is to predict how many discharge cycles remain before a lithium-ion battery reaches failure.

This is known as **RUL – Remaining Useful Life**.

In simple terms: *Given the current condition of a battery, how many cycles are left before it dies?*

---

## 2. Understanding the Dataset

The dataset comes from NASA battery experiments.

**Four batteries were tested:**

- B0005
- B0006
- B0007
- B0018

Each battery was repeatedly:

- Charged
- Discharged
- Measured for internal resistance (impedance)

Over time, batteries degrade. As degradation happens:

- Capacity decreases
- Internal resistance increases
- Voltage drops faster during discharge

---

## 3. What Is End of Life (EOL)?

A battery is considered dead when its capacity drops by 30%.

| Parameter | Value |
|-----------|-------|
| Original capacity | 2.0 Ah |
| End of Life threshold | 1.4 Ah |

**EOL occurs when Capacity ≤ 1.4 Ah**

---

## 4. What Is RUL?

For each discharge cycle:

```
RUL = (EOL cycle) − (current cycle)
```

**Example:** If battery fails at cycle 150:

| Cycle | RUL |
|-------|-----|
| 20 | 130 |
| 80 | 70 |
| 149 | 1 |

RUL is the target variable our model learns to predict.

---

## 5. What Does the Raw Data Look Like?

Each discharge cycle contains time-series data like:

| Time (s) | Voltage (V) | Current (A) | Temperature (°C) |
|----------|-------------|-------------|------------------|
| 0 | 4.2 | 2.0 | 25 |
| 1 | 4.19 | 2.0 | 25 |
| 2 | 4.18 | 2.0 | 25 |
| ... | ... | ... | ... |

Each cycle contains hundreds or thousands of rows. Machine learning models cannot directly use this raw time-series data, so we convert:

**Time-series data → One row per discharge cycle**

This transformation is called **preprocessing**.

---

## 6. Preprocessing Explained

Preprocessing converts messy experimental data into a clean dataset suitable for machine learning.

### Step 1: Keep Only Discharge Cycles

Capacity is measured during discharge. We filter `type == "discharge"`. Each discharge cycle becomes one modeling example.

### Step 2: Compute RUL

For each battery:

1. Find first cycle where Capacity ≤ 1.4 Ah
2. Compute: `RUL = EOL cycle − current cycle`

This creates the label (target variable).

### Step 3: Extract Features from Each Cycle

Instead of using raw time-series, we compute summary features:

| Feature | Description |
|---------|-------------|
| **Capacity** | Remaining charge the battery can store. Decreases as battery ages. |
| **capacity_fade** | `Capacity / initial_capacity` — normalized degradation indicator |
| **capacity_derivative** | Difference in capacity between consecutive cycles — shows degradation speed |
| **discharge_duration** | How long the discharge lasts. Healthy battery → longer; degraded → shorter |
| **avg_temperature** | Heat affects battery degradation |
| **voltage_at_100s, 300s, 600s** | Voltage at fixed times — captures shape of discharge curve |
| **Re, Rct** | Internal resistance from impedance. Increases with aging. Forward-filled from most recent impedance measurement |

### Step 4: Train/Test Split by Battery

- **Train on:** B0005, B0006, B0007
- **Test on:** B0018

*Why?* In real-world scenarios, we train on known batteries and predict on new unseen batteries. This prevents data leakage and memorization.

### Step 5: Scaling Features

Some features have different numeric ranges. We:

- Fit scaler on **training data only**
- Apply same scaling to test data

This prevents data leakage.

---

## 7. Final Processed Dataset

Each row represents **one discharge cycle** with columns:

- Capacity
- capacity_fade
- capacity_derivative
- Re
- Rct
- discharge_duration
- avg_temperature
- voltage_at_100s
- voltage_at_300s
- voltage_at_600s
- **RUL** (target)

This is a structured supervised learning dataset.

---

## 8. Model Training

After preprocessing:

1. Load train/test data
2. Train regression models
3. Predict RUL
4. Evaluate performance

**Models used:**

- Random Forest
- Gradient Boosting
- XGBoost
- Ensemble models
- Hyperparameter-tuned models

**Metrics used:**

- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² Score

---

## 9. Simple Summary

This project:

1. Converts raw battery experiment data into cycle-level health indicators
2. Computes remaining useful life
3. Trains machine learning models to predict battery failure
4. Evaluates performance on an unseen battery

> **In one sentence:** We use battery degradation signals like capacity, voltage behavior, and internal resistance to predict how many cycles remain before failure.
