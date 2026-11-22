"""Train one model with one seed, save the best weights and the learning curves.

Examples:
    python train.py --model p_cnn_lstm --seed 42
    python train.py --model p_cnn_lstm --seed 42 --next-day-features   # bonus candidate (a)
    python train.py --model p_cnn_lstm --seed 42 --lags 14              # bonus candidate (b)
    python train.py --model p_cnn_lstm --seed 42 --log-sales            # bonus candidate (c)
    python train.py --model p_cnn_lstm --seed 42 --dataset paper        # the paper's own data
"""
import argparse
import json
import os

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

from dataset import load, make_windows
from models import MODELS

EPOCHS = 50
BATCH_SIZE = 32
PATIENCE = 10
OUT_DIR = "results"


def run_name(model, seed, lags=7, log_sales=False, next_day_features=False):
    """e.g. 'p_cnn_lstm_s42' or 'p_cnn_lstm_lags14_s42'."""
    parts = [model]
    if lags != 7:
        parts.append(f"lags{lags}")
    if log_sales:
        parts.append("log")
    if next_day_features:
        parts.append("nextday")
    parts.append(f"s{seed}")
    return "_".join(parts)


def train(model, seed, dataset="rossmann", lags=7, log_sales=False, next_day_features=False):
    keras.utils.set_random_seed(seed)          # fixes Python, NumPy and TF seeds
    tf.config.experimental.enable_op_determinism()

    sales, _, features, (train_end, val_end) = load(dataset)
    data, _ = make_windows(sales, features, lags, log_sales, next_day_features,
                           train_end, val_end)
    X_train, y_train, _ = data["train"]
    X_val, y_val, _ = data["val"]

    net = MODELS[model](X_train.shape[1], X_train.shape[2], y_train.shape[1])
    stop = keras.callbacks.EarlyStopping(monitor="val_mae", patience=PATIENCE,
                                         restore_best_weights=True)
    history = net.fit(X_train, y_train, validation_data=(X_val, y_val),
                      epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[stop], verbose=2)

    name = run_name(model, seed, lags, log_sales, next_day_features)
    out = f"{OUT_DIR}/{dataset}"
    os.makedirs(f"{out}/models", exist_ok=True)
    os.makedirs(f"{out}/curves", exist_ok=True)
    net.save(f"{out}/models/{name}.keras")
    config = dict(model=model, seed=seed, dataset=dataset, lags=lags, log_sales=log_sales,
                  next_day_features=next_day_features)
    with open(f"{out}/models/{name}.json", "w") as f:
        json.dump(config, f)
    plot_curves(history.history, name, out)
    return name


def plot_curves(history, name, out):
    """Training and validation loss (MSE) and MAE per epoch."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric in zip(axes, ["loss", "mae"]):
        ax.plot(history[metric], label="train")
        ax.plot(history[f"val_{metric}"], label="validation")
        ax.set_title(f"{name} - {metric}")
        ax.set_xlabel("epoch")
        ax.legend()
    fig.tight_layout()
    fig.savefig(f"{out}/curves/{name}.png")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", choices=["rossmann", "paper"], default="rossmann")
    parser.add_argument("--lags", type=int, default=7)
    parser.add_argument("--log-sales", action="store_true")
    parser.add_argument("--next-day-features", action="store_true")
    args = parser.parse_args()
    print("saved", train(args.model, args.seed, args.dataset, args.lags, args.log_sales,
                         args.next_day_features))
