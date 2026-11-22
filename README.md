# Hybrid CNN-LSTM Sales Forecasting on Rossmann

Re-implementation of the stacked (**S-CNN-LSTM**) and parallel (**P-CNN-LSTM**) hybrid models of

> de Castro Moraes, T., Yuan, X.-M., & Chew, E. P. (2024). Hybrid convolutional long short-term memory
> models for sales forecasting in retail. *Journal of Forecasting*, 43(5), 1278–1293.
> https://doi.org/10.1002/for.3073

applied to a new dataset, **Rossmann Store Sales**, and checked against the paper on its own dataset.

## Datasets

| | Link | Used as |
|---|---|---|
| New Dataset | [Rossmann Store Sales](https://www.kaggle.com/competitions/rossmann-store-sales/data) | `data/train.csv` (`test.csv` has no sales and is not used) |
| Paper Dataset | [Store Item Demand Forecasting Challenge](https://www.kaggle.com/competitions/demand-forecasting-kernels-only/data) | `data/train_paper.csv` (the competition's `train.csv`, renamed) |

**Rossmann after preparation:** 934 stores (the ones with a record on all 942 days), 22 exogenous
features, 7-day input windows.

| Split | Target dates | Samples |
|---|---|---|
| Train | 2013-01-08 → 2014-12-31 | 723 |
| Validation | 2015-01-01 → 2015-04-30 | 120 |
| Test | 2015-05-01 → 2015-07-31 | 92 |

## How to run

```sh
pip install -r requirements.txt
sh run_experiments.sh                # every training run and evaluation (~30 min on CPU)
python app.py                        # Gradio demo
```

Single steps:

```sh
python dataset.py                                   # check the data pipeline (add `paper` for the paper data)
python train.py --model p_cnn_lstm --seed 42        # models: s_cnn_lstm, p_cnn_lstm, lstm
python train.py --model p_cnn_lstm --seed 42 --next-day-features   # the improved model
python evaluate.py --split val                      # use while choosing; --split test once at the end
python evaluate.py --split test --dataset paper
```

## Files

| File | What it does |
|---|---|
| `dataset.py` | Load CSVs → daily matrix (dates × series) + features → split by date → scale → 3D windows |
| `models.py` | S-CNN-LSTM, P-CNN-LSTM and the LSTM baseline, with the paper's hyperparameters |
| `train.py` | Train one model with one seed, with early stopping; save the model and its learning curves |
| `evaluate.py` | MAE, RMSE, MAPE and SMAPE on open days, plus the seasonal-naive baseline; mean ± std over seeds |
| `app.py` | Gradio demo |
| `run_experiments.sh` | Reproduces every result below |
| `GLOSSARY.md`, `docs/adr/` | Project terms and the two key design decisions |

Results land in `results/<dataset>/`: `models/`, `curves/` (training and validation curves of every run),
`*_summary.csv` (mean and std) and `*_per_run.csv`.

## Model settings

| Setting | Paper | Ours |
|---|---|---|
| Input | 7 lags × (500 series + calendar + US holidays) | 7 lags × (934 stores + 22 features) |
| LSTM units | 100, ReLU | same |
| CNN | 3 × [Conv1D → AvgPool], filters 128/256/500, pools 2/7/10 | same, `padding="same"` |
| Kernel size | not given | 3 |
| Dense in the P-CNN-LSTM LSTM branch | not given | 100 |
| Dropout | 0.2 / 0.3 | 0.2 inside the branches, 0.3 before the output |
| Initialisation | Gaussian, σ = 0.05 | same |
| Optimizer, learning rate, batch size | Adam, 0.001, 32 | same |
| Epochs, early stopping | up to 50, patience 10 on MAE | same (validation MAE, best weights restored) |
| Loss | MSE | same |
| Scaling | [0, 1] | MinMax fitted on the training period only |

## Results

All neural models: mean ± std over seeds 42, 43 and 44. Rossmann errors are scored on open days only.
Horizons are one-step-ahead forecasts scored over the first N days of the test period
(see `docs/adr/0002-one-step-ahead-horizons.md`).

### Rossmann (test set)

| Horizon | Model | MAE | RMSE | MAPE % | SMAPE % |
|---|---|---|---|---|---|
| 30 d | Seasonal naive | 1947 | 3002 | 27.37 | 34.56 |
| | LSTM baseline | 1136 ± 26 | 1785 ± 174 | 15.12 ± 0.22 | 16.11 ± 0.80 |
| | S-CNN-LSTM | 1071 ± 115 | 1532 ± 199 | 14.51 ± 1.39 | 15.36 ± 1.62 |
| | P-CNN-LSTM | 1066 ± 34 | 1522 ± 24 | 14.58 ± 0.61 | 15.43 ± 0.71 |
| | **P-CNN-LSTM + next-day features** | **796 ± 52** | **1088 ± 76** | **11.16 ± 0.62** | **11.35 ± 0.79** |
| 90 d | Seasonal naive | 2265 | 3167 | 31.91 | 34.92 |
| | LSTM baseline | 921 ± 24 | 1384 ± 86 | 12.94 ± 0.33 | 13.05 ± 0.50 |
| | S-CNN-LSTM | 876 ± 41 | 1260 ± 79 | 12.28 ± 0.64 | 12.60 ± 0.64 |
| | P-CNN-LSTM | 897 ± 20 | 1280 ± 26 | 12.79 ± 0.27 | 12.87 ± 0.32 |
| | **P-CNN-LSTM + next-day features** | **778 ± 27** | **1103 ± 43** | **11.11 ± 0.42** | **11.18 ± 0.39** |

### Paper Dataset (test year 2017): the paper's numbers vs ours

| Horizon | Model | Paper MAPE | Paper SMAPE | Our MAPE | Our SMAPE |
|---|---|---|---|---|---|
| 30 d | LSTM | 14.62 | 15.01 | 15.04 ± 0.29 | 14.25 ± 0.07 |
| | S-CNN-LSTM | 14.15 | 14.47 | 15.03 ± 0.38 | 14.43 ± 0.38 |
| | P-CNN-LSTM | 14.76 | 14.57 | 15.13 ± 0.26 | 14.27 ± 0.10 |
| | P-CNN-LSTM + next-day features | – | – | 14.90 ± 0.02 | 14.10 ± 0.05 |
| 90 d | LSTM | 15.02 | 16.22 | 14.23 ± 0.15 | 13.64 ± 0.06 |
| | S-CNN-LSTM | 15.41 | 14.65 | 14.40 ± 0.22 | 14.26 ± 0.63 |
| | P-CNN-LSTM | 14.00 | 13.91 | 14.37 ± 0.13 | 13.70 ± 0.11 |
| | P-CNN-LSTM + next-day features | – | – | 14.09 ± 0.01 | 13.55 ± 0.02 |
| 360 d | LSTM | 16.29 | 15.18 | 12.65 ± 0.06 | 12.32 ± 0.12 |
| | S-CNN-LSTM | 16.11 | 14.95 | 13.22 ± 0.48 | 13.38 ± 0.77 |
| | P-CNN-LSTM | 13.42 | 13.37 | 12.79 ± 0.07 | 12.32 ± 0.20 |
| | P-CNN-LSTM + next-day features | – | – | 12.51 ± 0.02 | 12.14 ± 0.04 |

Seasonal naive on the paper data: 21.05 / 19.83 (30 d), 19.77 / 19.34 (90 d), 17.92 / 17.32 (360 d).
Full tables with MAE and RMSE are in `results/paper/test_summary.csv`.

## Bonus: improving the model

Three changes to P-CNN-LSTM were screened on the **Rossmann validation set** (3 seeds each, 30-day MAPE):

| Change | Validation MAPE (30 d) |
|---|---|
| None (original P-CNN-LSTM) | 13.46 ± 0.51 |
| (a) next-day features | **11.30 ± 0.71** |
| (b) 14-day lag window | 16.30 ± 0.05 |
| (c) train on `log1p(sales)` | 40.97 ± 3.06 |

**(a) was kept.** Each step of the input window carries the calendar, holiday and promo features of the
*following* day, so the last step describes the day being forecast. These values are known in advance.
The paper's design only shows the model past days, so it cannot know that tomorrow is a holiday or a
promo day. On Rossmann this cuts test MAPE by 3.4 points (30 d) and 1.7 points (90 d), because
promotions and holidays drive a large share of the sales swings. On the paper data the gain is small but
consistent (−0.2 to −0.3 MAPE), since only the calendar and US holidays are available there.

The rejected candidates' models are kept in `results/rossmann/rejected_candidates/`.

## Notes for the discussion

- **No 360-day horizon on Rossmann:** the data covers only 942 days, so the test period is 92 days.
- **Closed days** (Sales = 0) stay in training but are not scored; MAPE is undefined on them.
- **Fewer training samples than the paper** (723 days vs 1,088).
- **Pooling on a 7-step window:** with `padding="same"` the time axis shrinks 7 → 4 → 1 → 1, so the
  CNN block summarises the whole week.
- **Validation error below training error:** dropout is active only during training, and January–April
  is a calmer sales period.
- **Negative predictions on closed Sundays:** the output layer is linear; these days are not scored.
