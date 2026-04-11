# Lithium-Ion Battery RUL Project Notes

## 1. Project Title

**Lithium-Ion Battery Remaining Useful Life (RUL) Prediction using the NASA Battery Dataset**

---

## 2. Project Goal

The goal of this project is to predict the **Remaining Useful Life (RUL)** of lithium-ion batteries.

In this project, RUL means:

> **How many discharge cycles are left before the battery capacity drops below 1.4 Ah**

This is useful because it helps estimate when a battery is approaching failure, which is important for:

- predictive maintenance
- battery health monitoring
- safety
- planning battery replacement
- robotics and energy systems where battery reliability matters

---

## 3. Dataset Used

We use the **NASA Battery Dataset**, which contains repeated battery cycle measurements collected under controlled laboratory conditions.

The dataset includes multiple cycle types:

- charge cycles
- discharge cycles
- impedance cycles

Each cycle can contain:

- battery ID
- cycle type
- filename of the cycle data
- capacity
- resistance values such as `Re` and `Rct`
- time-series sensor readings like voltage, temperature, and time

### Batteries chosen in this project

This project does **not** use every battery in the full NASA dataset. It uses the following batteries:

- `B0005` for training
- `B0006` for training
- `B0018` for testing
- `B0007` only for censored analysis

### Why these batteries were chosen

- `B0005` and `B0006` both reach failure and can be used for supervised learning
- `B0018` is kept fully unseen during training so we can test real generalization
- `B0007` never falls below the failure threshold in the available data, so true RUL is not known in the same way

---

## 4. What Data We Use and What We Do Not Use

### Data we use

The supervised prediction unit in this project is a **discharge cycle**.

For each discharge cycle, we use:

- capacity
- capacity fade
- capacity derivative
- internal resistance indicators `Re` and `Rct`
- discharge duration
- average temperature
- voltage at 100 seconds
- voltage at 300 seconds
- voltage at 600 seconds
- discharge cycle index

We also use impedance-cycle information **indirectly**:

- `Re`
- `Rct`

These are forward-filled from prior impedance measurements into later discharge rows from the same battery.

### What each feature means and what it signifies

The model input features can be explained in a simple teacher-friendly way:

- `Capacity`: the measured discharge capacity of the battery in that cycle. It tells us how much charge the battery can still deliver. Lower capacity usually means the battery is closer to end of life.
- `capacity_fade`: how much capacity has been lost relative to the first discharge cycle. This is an engineered feature created in our pipeline. It signifies overall aging progress.
- `capacity_derivative`: the change in capacity from the previous discharge cycle. This is also engineered in our pipeline. It signifies how fast the battery is degrading from one cycle to the next.
- `Re`: electrolyte resistance from impedance data. Higher `Re` usually indicates more internal degradation and harder current flow.
- `Rct`: charge-transfer resistance from impedance data. Higher `Rct` often means the battery's electrochemical reactions are becoming less efficient as it ages.
- `discharge_duration`: the total time of the discharge cycle, computed from the time-series file. This is a derived summary feature we create. A shorter discharge duration often means the battery cannot sustain discharge as long as before.
- `avg_temperature`: the average measured temperature during discharge, computed from the time-series file. This is a derived summary feature we create. It signifies thermal behavior, and abnormal heat can be linked to stress and aging.
- `voltage_at_100s`: voltage value around 100 seconds during discharge. This is an engineered snapshot feature obtained by interpolation from the raw voltage curve. It helps summarize the early part of the discharge profile.
- `voltage_at_300s`: voltage value around 300 seconds during discharge. This is another engineered voltage-curve summary that captures mid-discharge behavior.
- `voltage_at_600s`: voltage value around 600 seconds during discharge. This is another engineered voltage-curve summary that captures later discharge behavior.
- `discharge_cycle_index`: the count of discharge cycles seen so far for that battery. This is generated in our pipeline. It signifies how far along the battery is in its life.

### Why these features matter to the model

These features were chosen because they describe battery health from different viewpoints:

- `Capacity` gives the most direct signal of how much useful charge is still available.
- `capacity_fade` tells the model how much life has already been lost compared with the beginning.
- `capacity_derivative` tells the model whether degradation is happening slowly or suddenly.
- `Re` and `Rct` help the model capture internal electrochemical degradation that may not be visible from capacity alone.
- `discharge_duration` helps show whether the battery can still sustain discharge for long periods.
- `avg_temperature` helps the model capture thermal stress and abnormal battery behavior.
- `voltage_at_100s`, `voltage_at_300s`, and `voltage_at_600s` summarize the shape of the discharge curve instead of using only one scalar value.
- `discharge_cycle_index` gives the model life-progress context, which is useful because the same capacity value can mean different things at early and late stages.

So overall, the feature set combines:

- direct health information
- degradation trend information
- internal resistance information
- thermal information
- voltage-curve information
- life-stage information

### Which features are directly available and which ones we made

#### Directly available or directly measured values

- `Capacity` comes from the dataset metadata for discharge cycles
- `Re` and `Rct` come from impedance-cycle metadata
- raw time, voltage, and temperature readings come from each cycle CSV file

#### Features we created or transformed in preprocessing

- `capacity_fade`
- `capacity_derivative`
- `discharge_duration`
- `avg_temperature`
- `voltage_at_100s`
- `voltage_at_300s`
- `voltage_at_600s`
- `discharge_cycle_index`

We also transform impedance information by carrying the latest known `Re` and `Rct` from earlier impedance cycles into later discharge-cycle rows, so each discharge sample has the newest available resistance indicators.

### Other processed columns created by the pipeline

Besides the main model-input features, our processed table also contains:

- `ambient_temperature`: lab ambient temperature from metadata; it is stored but not currently used as a model feature
- `failure_cycle_index`: the first discharge cycle where capacity falls below 1.4 Ah
- `is_censored`: whether the battery never reaches failure in the available data
- `RUL`: the prediction target, meaning how many discharge cycles are left until failure
- `split`: whether the row belongs to train, validation, test, or censored data
- `raw_*` columns such as `raw_Capacity` and `raw_voltage_at_300s`: unscaled copies saved after preprocessing so we can still inspect original values even though the model trains on standardized features

### Data we do not use directly

We do **not** directly use:

- charge cycles as supervised samples
- future cycles when predicting the current cycle
- full-life aggregate statistics that would reveal later-life behavior
- censored battery `B0007` as labeled supervised training data

### Why we exclude this data

#### Why not use charge cycles as training samples

RUL is defined around **discharge capacity degradation**, so discharge cycles are the most natural prediction unit.

#### Why not use future information

Using future cycles would create **data leakage**, which would make the model look better than it really is.

#### Why not train on `B0007`

`B0007` does not cross the failure threshold in the provided data. Since the failure point is unknown, supervised RUL labels cannot be assigned in the same way.

#### Why not use full-life summary features

Those features would leak information from later cycles into earlier predictions and would make the evaluation unrealistic.

---

## 5. How RUL Is Defined

This project defines failure as:

> **Capacity < 1.4 Ah**

Then for each discharge cycle:

> **RUL = failure cycle index - current discharge cycle index**

That means:

- the final discharge cycle before failure has low RUL
- the threshold-crossing discharge cycle has `RUL = 0`

This gives a physically meaningful target tied to actual battery degradation.

---

## 6. Preprocessing Done in the Project

The preprocessing pipeline converts raw cycle data into a machine-learning-ready table.

### Step 1: Read metadata

The pipeline reads:

- `data/metadata.csv`

This file tells us:

- which battery each row belongs to
- what type of cycle it is
- which raw CSV file contains the detailed time-series data

### Step 2: Keep only the relevant batteries

The pipeline keeps only:

- `B0005`
- `B0006`
- `B0018`
- `B0007`

### Step 3: Process cycles battery by battery

For each battery:

- rows are sorted in time order
- impedance values are tracked
- only discharge cycles become supervised feature rows

### Step 4: Extract engineered features

For every discharge cycle, the project computes:

- `Capacity`
- `capacity_fade = 1 - current_capacity / first_discharge_capacity`
- `capacity_derivative = current_capacity - previous_capacity`
- `Re`
- `Rct`
- `discharge_duration`
- `avg_temperature`
- `voltage_at_100s`
- `voltage_at_300s`
- `voltage_at_600s`
- `discharge_cycle_index`

What these engineered features are meant to capture:

- degradation level through `capacity_fade`
- short-term degradation trend through `capacity_derivative`
- electrochemical wear through `Re` and `Rct`
- discharge behavior through `discharge_duration`
- thermal stress through `avg_temperature`
- shape of the voltage curve through `voltage_at_100s`, `voltage_at_300s`, and `voltage_at_600s`
- life-progress information through `discharge_cycle_index`

### Step 5: Assign RUL labels

For batteries that cross the failure threshold:

- find the first cycle where `Capacity < 1.4`
- keep discharge cycles up to that point
- assign RUL to each row

For censored batteries like `B0007`:

- mark them as censored
- do not create supervised RUL labels

### Step 6: Split the data

The split is battery-wise:

- train batteries: `B0005`, `B0006`
- validation: last 20% of supervised cycles within each training battery
- test battery: `B0018`
- censored only: `B0007`

### Step 7: Handle missing values

Missing feature values are filled using the **median of the training data only**.

### Step 8: Scale features

A `StandardScaler` is fitted **only on training rows** and then applied to:

- train
- validation
- test
- censored rows

### Processed outputs created

The preprocessing script creates:

- `processed/full.csv`
- `processed/train.csv`
- `processed/val.csv`
- `processed/test.csv`
- `processed/feature_columns.json`
- `processed/scaler.pkl`

### Current processed split sizes

From the current project run:

- train rows: `187`
- validation rows: `47`
- test rows: `97`
- full processed rows: `499`

Battery-wise counts:

- `B0005`: 100 train, 25 validation
- `B0006`: 87 train, 22 validation
- `B0018`: 97 test
- `B0007`: 168 censored

---

## 7. Whole Workflow of the Project

The overall workflow is:

### Stage 1: Raw data

- read metadata
- read per-cycle time-series files

### Stage 2: Feature engineering

- convert raw cycle files into one feature row per discharge cycle
- compute RUL labels

### Stage 3: Data split and scaling

- create train, validation, and test sets
- fit imputation and scaling on train only

### Stage 4: Model training

Train multiple machine learning and deep learning models.

### Stage 5: Evaluation

Evaluate each run using:

- MAE
- RMSE
- R2

### Stage 6: Experiment tracking

Each experiment stores:

- model artifact
- config file
- metrics file
- predictions file
- plots

### Stage 7: Dashboard and comparison

The Streamlit dashboard is used to:

- compare models
- inspect tabular and sequence predictions
- generate simple what-if forecasts and local feature explanations for row models
- view diagnostic plots
- trigger training runs

### Stage 8: Sweep runs

The project also includes:

- `run_sweeps.py`

This script is used to automatically train many parameter combinations and populate the experiment results.

---

## 8. Models Used in This Project

The project currently uses the following models.

### 1. Linear Regression

This is the simplest baseline model.

Purpose:

- check whether a basic linear relationship is enough
- give a simple baseline for comparison

How it uses the features:

- it takes one discharge-cycle feature row at a time
- it assumes the relationship between each input feature and RUL is mostly linear
- it combines all features using a weighted sum to predict RUL

In simple words:

- if `Capacity` goes down, RUL usually goes down
- if `Re` or `Rct` goes up, RUL usually goes down
- if `discharge_duration` is longer, RUL may be higher

This model is easy to explain, but it cannot capture strong nonlinear behavior very well.

### 2. Ridge Regression

This is linear regression with regularization.

Purpose:

- reduce instability
- improve generalization compared with plain linear regression

How it uses the features:

- it uses the same input features as linear regression
- it also learns a weighted linear combination of those features
- but it adds a penalty so that no single feature weight becomes unrealistically large

Why this helps:

- battery data is small and noisy
- regularization makes the model more stable
- it usually generalizes better to an unseen battery

In this project, Ridge is strong because it keeps the model simple while still using all important battery-health features.

### 3. Random Forest Regressor

This is a tree-based ensemble model.

Purpose:

- capture nonlinear relationships
- remain relatively robust on small tabular datasets

How it uses the features:

- it takes one discharge-cycle feature row at a time
- it builds many decision trees
- each tree learns rules such as:
  - if `Capacity` is low and `Rct` is high, RUL is likely low
  - if `capacity_fade` is still small, RUL is likely higher

Why this is useful:

- it can model nonlinear behavior better than linear methods
- it can capture interactions between features
- for example, temperature and resistance together may indicate stronger degradation than either feature alone

Random Forest is often a good middle ground between interpretability and nonlinear learning power.

### 4. XGBoost Regressor

This is a boosted tree model.

Purpose:

- strong performance on tabular data
- good at learning nonlinear feature interactions

How it uses the features:

- it also takes one discharge-cycle feature row at a time
- instead of building independent trees like Random Forest, it builds trees sequentially
- each new tree focuses on correcting the mistakes made by previous trees

Why this helps:

- it usually fits tabular battery-health features very well
- it can learn subtle interactions between `Capacity`, `capacity_fade`, resistance, temperature, voltage points, and cycle index
- it often gives strong performance even when the dataset is not very large

In this project, XGBoost works well because the input is structured tabular data with meaningful engineered features.

### 5. LSTM

This is a recurrent neural network for sequence modeling.

Purpose:

- use several recent discharge cycles together
- learn temporal degradation patterns

How it uses the features:

- instead of looking at only one discharge cycle, it looks at a sequence of recent discharge cycles
- each cycle in the sequence contains the full feature vector:
  - `Capacity`
  - `capacity_fade`
  - `capacity_derivative`
  - `Re`
  - `Rct`
  - `discharge_duration`
  - `avg_temperature`
  - `voltage_at_100s`
  - `voltage_at_300s`
  - `voltage_at_600s`
  - `discharge_cycle_index`
- the LSTM processes these cycle-by-cycle and tries to remember how the battery trend is evolving over time

Why this is useful:

- battery degradation is naturally a time-dependent process
- two batteries with similar current capacity may still have different future RUL if their recent degradation trends are different

So LSTM is useful when recent history matters, not just the current snapshot.

### 6. GRU

This is another recurrent sequence model, lighter than LSTM.

Purpose:

- capture temporal information with fewer parameters

How it uses the features:

- it uses the same sequence-style input as LSTM
- each input step is one discharge-cycle feature vector
- it learns temporal degradation patterns across consecutive cycles

Why GRU is different from LSTM:

- it has a simpler internal structure
- it usually trains faster
- it can work well when we want sequence modeling with fewer parameters

In simple words, GRU tries to do the same job as LSTM but in a lighter and sometimes more efficient way.

### 7. SVR

This is a support vector regression baseline.

Purpose:

- test a nonlinear kernel method on the engineered tabular features
- compare margin-based regression with tree-based and linear models

How it uses the features:

- it takes one discharge-cycle feature row at a time
- it maps the inputs into a nonlinear feature space through the chosen kernel
- in this project, it can also apply a simple post-hoc calibration step using validation predictions

Why this is useful:

- it can model nonlinear relationships without building deep sequence models
- it gives another reference point for how hard this dataset is

### 8. MLP

This is a feed-forward neural network baseline for tabular data.

Purpose:

- test whether a small neural network on engineered features can compete with classical tabular models

How it uses the features:

- it takes one discharge-cycle feature row at a time
- it passes the feature vector through hidden layers with nonlinear activations

Why this is useful:

- it checks whether nonlinear learned combinations of the engineered features help, even without explicit sequence modeling

### Common understanding of feature usage across models

The models in this project use the features in two main ways:

- **Tabular models**: Linear Regression, Ridge Regression, Random Forest, XGBoost, SVR, and MLP use one discharge-cycle feature row at a time.
- **Sequence models**: LSTM and GRU use multiple consecutive discharge-cycle rows together as a short history window.

This means:

- tabular models learn from the current battery snapshot
- sequence models learn from both the current snapshot and the recent trend

---

## 9. Why We Tried Both Simple and Complex Models

This project is not only about getting the lowest error. It is also about understanding:

- whether simple baselines are already strong
- whether more complex sequence models truly add value
- whether a model generalizes to a completely unseen battery

This is why both:

- simple tabular baselines
- deep sequence models

were included.

---

## 10. How Model Accuracy Is Compared

We mainly compare models using:

### MAE

**Mean Absolute Error**

Interpretation:

- average number of cycles the prediction is off by

This is the easiest metric to explain to a teacher.

### RMSE

**Root Mean Squared Error**

Interpretation:

- similar to MAE, but gives more penalty to large mistakes

### R2

**Coefficient of determination**

Interpretation:

- how much of the variation in RUL is explained by the model

### Important evaluation note

For this project, **test battery performance matters more than validation performance**.

Reason:

- validation samples still come from the same training batteries
- test samples come from a completely unseen battery

So the best model should be chosen mostly using:

- test MAE
- test RMSE
- test R2

---

## 11. Current Best Results

The results below are based on the current `experiments/summary.csv`.

### Best current run for each model


| Model             | Best Experiment                 | Test MAE | Test RMSE | Test R2 | Validation MAE | Validation RMSE | Validation R2 |
| ----------------- | ------------------------------- | -------- | --------- | ------- | -------------- | --------------- | ------------- |
| XGBoost           | `xgboost_20260411_095234`       | 10.90    | 12.94     | 0.786   | 13.83          | 15.25           | -3.941        |
| Ridge             | `ridge_20260409_214544`         | 11.23    | 12.26     | 0.808   | 7.19           | 7.83            | -0.301        |
| LSTM              | `lstm_20260411_113957`          | 11.28    | 13.57     | 0.744   | 2.47           | 3.01            | 0.723         |
| Random Forest     | `random_forest_20260411_113657` | 11.64    | 13.97     | 0.751   | 15.40          | 16.77           | -4.978        |
| SVR               | `svr_20260411_110439`           | 20.37    | 25.03     | 0.201   | 19.63          | 21.84           | -9.135        |
| MLP               | `mlp_20260411_110439`           | 23.71    | 30.00     | -0.148  | 28.71          | 30.11           | -18.269       |
| Linear Regression | `linear_20260409_183522`        | 27.11    | 27.12     | 0.062   | 0.58           | 0.82            | 0.986         |
| GRU               | `gru_20260411_095350`           | 34.58    | 42.14     | -1.464  | 4.51           | 5.29            | 0.146         |


### Best-performing models overall right now

At the moment, the strongest models are:

1. **XGBoost**
2. **Ridge Regression**
3. **LSTM / Random Forest are close behind depending on whether you prioritize MAE or R2**

These are the most promising because they currently give the best unseen-battery test performance. The latest sweeps also show that a tuned LSTM can become competitive, even though the strongest overall results are still coming from the engineered-feature pipeline.

---

## 12. Which Models Look Overfitted

A model is suspicious if:

- it is extremely good on validation
- but much worse on the unseen test battery

That usually means it learned battery-specific patterns instead of general battery-aging behavior.

### Models that look strongly overfitted or poorly generalized

#### LSTM

- older runs showed a strong validation-versus-test gap
- after tuning sequence length and hidden size, later runs improved a lot on the unseen battery

This means the earlier LSTM setup was overfitting, but the newer tuned LSTM is much more competitive and should not be described as a failed model anymore.

#### GRU

- validation is much better than test
- test performance is still poor in the current setup

#### Linear Regression

- excellent validation result
- poor test result

This suggests the model fits the seen batteries in a way that does not transfer well to the new battery.

### Models that generalize better

#### XGBoost

- stable and strong on the unseen test battery
- consistently among the best across many runs

#### Random Forest

- also stable on the unseen battery
- slightly worse or similar to the best XGBoost runs depending on the parameter setting

#### Ridge Regression

- currently gives the best recorded test RMSE and test R2, while XGBoost has the best recorded test MAE
- however, it should still be treated carefully and confirmed with more battery-wise testing because the dataset is small

### Main conclusion on overfitting

The most obvious overfitting signs in the current results are from:

- GRU
- plain Linear Regression
- MLP and some SVR settings

The most trustworthy current models are:

- XGBoost
- Ridge
- tuned LSTM
- Random Forest

---

## 13. Main Strengths of This Project

- clear physics-based RUL definition
- careful leakage avoidance
- battery-wise test split
- combination of simple baselines and deep models
- tracked experiments with saved metrics and plots
- dashboard for presentation and analysis
- sweep script to populate many runs automatically

---

## 14. Main Limitations of This Project

- only a small number of batteries are used
- validation still comes from the training batteries, so it is not a perfect measure of generalization
- deep sequence models may need more data or better architecture to perform well
- one held-out battery is useful, but more battery-wise testing would make conclusions stronger

---

## 15. Final Conclusion

This project builds a full machine learning pipeline for **battery Remaining Useful Life prediction** using the NASA battery dataset.

The project:

- selects relevant discharge-cycle data
- engineers physically meaningful battery health features
- avoids leakage
- trains multiple machine learning and deep learning models
- compares them on an unseen battery

### Final project takeaway

The main finding is:

> **More complex models are not automatically better**

In this project, the sequence models:

- LSTM
- GRU

show signs of overfitting and poor generalization.

The most effective current models are:

- XGBoost
- Ridge Regression
- tuned LSTM
- Random Forest

So the present evidence suggests that, for this dataset and feature design, **engineered tabular models remain the strongest overall, but a tuned LSTM can become competitive while GRU still underperforms in the current setup**.

---

## 16. Files Related to This Project

Important files in the repository:

- `README.md`
- `preprocessing/pipeline.py`
- `preprocessing/dataset.py`
- `train.py`
- `run_sweeps.py`
- `training/trainer.py`
- `training/evaluate.py`
- `app.py`
- `experiments/summary.csv`

---

## 17. Short Viva / Presentation Summary

If you need a short version to say in class:

> This project predicts lithium-ion battery remaining useful life using the NASA battery dataset. We only use discharge cycles as supervised samples because RUL is defined by discharge capacity degradation. We engineer battery-health features such as capacity fade, resistance, temperature, and voltage-curve points, then train multiple models including linear regression, ridge regression, random forest, XGBoost, SVR, MLP, LSTM, and GRU. We split the data battery-wise so that one battery remains completely unseen during testing. Our latest results show that XGBoost and Ridge are still the strongest overall models, and a tuned LSTM has become competitive, while GRU and some other baselines still show poor generalization. This means that for this project, careful feature engineering and proper evaluation matter more than model complexity alone.
