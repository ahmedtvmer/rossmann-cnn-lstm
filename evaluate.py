"""Score every saved model (plus the seasonal-naive baseline) on open days.

Use --split val while choosing the bonus change; run --split test once at the end.
    python evaluate.py --split val
    python evaluate.py --split test
    python evaluate.py --split test --dataset paper
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from tensorflow import keras

from dataset import load, make_windows, to_sales

HORIZONS = (30, 90, 360)   # 360 only exists for the paper dataset's one-year test set
OUT_DIR = "results"


def metrics(actual, predicted, scored):
    """MAE, RMSE, MAPE and SMAPE over the store-days where `scored` is True."""
    a, p = actual[scored], predicted[scored]
    error = np.abs(a - p)
    return {
        "MAE": error.mean(),
        "RMSE": np.sqrt((error ** 2).mean()),
        "MAPE": 100 * (error / a).mean(),
        "SMAPE": 100 * (error / ((np.abs(a) + np.abs(p)) / 2)).mean(),
    }


def score(name, dates, predicted, sales, is_open):
    """One row per horizon: metrics over the first `h` days of `dates`."""
    actual = sales.loc[dates].values
    scored = (is_open.loc[dates].values == 1) & (actual > 0)   # Open Days only
    rows = []
    for h in HORIZONS:
        if h <= len(dates):
            rows.append({"run": name, "horizon": h,
                         **metrics(actual[:h], predicted[:h], scored[:h])})
    return rows


def predict_model(config_path, sales, features, ends, split):
    """Rebuild the model's windows, predict, and return real-unit sales."""
    with open(config_path) as f:
        cfg = json.load(f)
    data, scaler = make_windows(sales, features, cfg["lags"], cfg["log_sales"],
                                cfg["next_day_features"], *ends)
    X, _, dates = data[split]
    net = keras.models.load_model(config_path.replace(".json", ".keras"))
    predicted = to_sales(net.predict(X, verbose=0), scaler, cfg["log_sales"])
    return dates, predicted


def seasonal_naive(sales, dates):
    """Each store sells what it sold 7 days earlier."""
    return sales.shift(7).loc[dates].values


def main(dataset, split):
    sales, is_open, features, ends = load(dataset)
    out = f"{OUT_DIR}/{dataset}"
    rows = []
    dates = None
    for config_path in sorted(glob.glob(f"{out}/models/*.json")):
        dates, predicted = predict_model(config_path, sales, features, ends, split)
        name = os.path.basename(config_path).replace(".json", "")
        rows += score(name, dates, predicted, sales, is_open)
    rows += score("seasonal_naive_s0", dates, seasonal_naive(sales, dates), sales, is_open)

    per_run = pd.DataFrame(rows)
    per_run["model"] = per_run["run"].str.rsplit("_s", n=1).str[0]   # drop the seed
    summary = per_run.drop(columns="run").groupby(["horizon", "model"]).agg(["mean", "std"])

    per_run.to_csv(f"{out}/{split}_per_run.csv", index=False)
    summary.to_csv(f"{out}/{split}_summary.csv")
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print(summary.round(2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["rossmann", "paper"], default="rossmann")
    parser.add_argument("--split", choices=["val", "test"], required=True)
    args = parser.parse_args()
    main(args.dataset, args.split)
