# Rossmann CNN-LSTM Replication

Re-implementing the hybrid CNN-LSTM sales forecasters of de Castro Moraes et al. (2024) on Rossmann Store Sales.

## Data

**Paper Dataset**:
The Kaggle "Store Item Demand Forecasting Challenge" data (10 stores × 50 items) that the paper used.
_Avoid_: original dataset, Kaggle data

**New Dataset**:
The Kaggle Rossmann Store Sales `train.csv`, the dataset we apply the models to.
_Avoid_: our data, Rossmann test.csv (it has no sales and is not used)

**Series**:
The daily sales of one store; in the New Dataset a store plays the role the paper gives to a product.
_Avoid_: product, item, time series (ambiguous)

**Complete Store**:
A store with a sales record on every one of the 942 days; only these 934 stores form Series.
_Avoid_: valid store, clean store

**Open Day**:
A store-day with Open = 1 and Sales > 0; the only store-days that are scored.
_Avoid_: trading day, active day

**Exogenous Features**:
The day-level inputs besides sales: one-hot day-of-week and month, plus the share of stores on state holiday, school holiday and promo.
_Avoid_: external variables, auxiliary data, covariates

## Forecasting

**Lag Window**:
The 7 consecutive past days fed to a model as its time steps.
_Avoid_: lookback, sequence length

**Horizon**:
The first 30 or 90 days of the Test Period over which one-step-ahead forecasts are scored.
_Avoid_: forecast length, steps ahead

**Train Period / Validation Period / Test Period**:
2013-01-01–2014-12-31, 2015-01-01–2015-04-30 and 2015-05-01–2015-07-31; the Test Period is scored once.
_Avoid_: train/val/test split (when meaning dates)

## Models

**S-CNN-LSTM**:
The paper's stacked hybrid: an LSTM feeding a three-stage convolutional block.
_Avoid_: stacked model, CNN-LSTM

**P-CNN-LSTM**:
The paper's parallel hybrid: LSTM and convolutional branches concatenated before the output.
_Avoid_: parallel model, hybrid

**LSTM Baseline**:
The paper's LSTM(100)–Dense benchmark, retrained on the New Dataset.
_Avoid_: simple model, vanilla LSTM

**Seasonal Naive**:
The forecast that each store sells what it sold on the same weekday one week earlier.
_Avoid_: naive baseline, persistence
