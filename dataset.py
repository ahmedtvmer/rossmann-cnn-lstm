"""Data pipeline: raw Rossmann CSVs -> scaled 3D windows [samples, lags, features]."""
import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from sklearn.preprocessing import MinMaxScaler

TRAIN_END = "2014-12-31"
VAL_END = "2015-04-30"


def load(name):
    """Return sales, open flags, features and (train_end, val_end) for a dataset."""
    if name == "rossmann":
        return (*load_rossmann(), (TRAIN_END, VAL_END))
    if name == "paper":
        return (*load_paper(), ("2015-12-31", "2016-12-31"))   # the paper's split
    raise ValueError(name)


def load_paper(path="data/train_paper.csv"):
    """Store Item Demand data used by the paper: 10 stores x 50 items = 500 series."""
    df = pd.read_csv(path, parse_dates=["date"])
    sales = df.pivot_table(index="date", columns=["store", "item"], values="sales").sort_index()
    is_open = pd.DataFrame(1, index=sales.index, columns=sales.columns)   # never closed

    holidays = USFederalHolidayCalendar().holidays(sales.index.min(), sales.index.max())
    features = calendar_features(sales.index)
    features["us_holiday"] = sales.index.isin(holidays).astype(float)
    return sales, is_open, features


def load_rossmann(path="data/train.csv"):
    """Return daily sales and open flags (dates x stores) plus daily exogenous features."""
    df = pd.read_csv(path, parse_dates=["Date"], dtype={"StateHoliday": str})

    # keep only stores that have a record on every day
    days_per_store = df.groupby("Store").size()
    complete = days_per_store[days_per_store == df["Date"].nunique()].index
    df = df[df["Store"].isin(complete)]

    sales = df.pivot(index="Date", columns="Store", values="Sales").sort_index()
    is_open = df.pivot(index="Date", columns="Store", values="Open").sort_index()

    shares = df.groupby("Date").agg(
        state_holiday=("StateHoliday", lambda s: (s != "0").mean()),
        school_holiday=("SchoolHoliday", "mean"),
        promo=("Promo", "mean"),
    )
    features = calendar_features(sales.index).join(shares)
    return sales, is_open, features


def calendar_features(dates):
    """One-hot day of week (7) and month (12)."""
    dow = pd.get_dummies(dates.day_of_week, prefix="dow")
    month = pd.get_dummies(dates.month, prefix="month")
    return pd.concat([dow, month], axis=1).set_index(dates).astype(float)


def make_windows(sales, features, lags=7, log_sales=False, next_day_features=False,
                 train_end=TRAIN_END, val_end=VAL_END):
    """Build (X, y) windows for train / val / test, split by the date of the target day.

    X[i] holds `lags` past days of [scaled sales of every store, features].
    y[i] is the scaled sales of every store on the target day.
    """
    values = np.log1p(sales.values) if log_sales else sales.values.astype(float)

    # fit the scaler on training days only, to avoid leaking the future
    train_rows = sales.index <= train_end
    scaler = MinMaxScaler().fit(values[train_rows])
    scaled = scaler.transform(values)

    feats = features.values
    if next_day_features:
        # each step carries the calendar of the following day,
        # so the last step tells the model about the target day
        feats = np.roll(feats, -1, axis=0)
    inputs = np.hstack([scaled, feats])

    X, y = [], []
    for t in range(lags, len(sales)):
        X.append(inputs[t - lags:t])
        y.append(scaled[t])
    X, y = np.array(X, dtype="float32"), np.array(y, dtype="float32")
    target_dates = sales.index[lags:]

    splits = {
        "train": target_dates <= train_end,
        "val": (target_dates > train_end) & (target_dates <= val_end),
        "test": target_dates > val_end,
    }
    data = {name: (X[mask], y[mask], target_dates[mask]) for name, mask in splits.items()}
    return data, scaler


def to_sales(scaled, scaler, log_sales=False):
    """Undo scaling (and log) to get predictions back in real sales units."""
    values = scaler.inverse_transform(scaled)
    return np.expm1(values) if log_sales else values


if __name__ == "__main__":
    import sys
    sales, is_open, features, (train_end, val_end) = load(sys.argv[1] if len(sys.argv) > 1 else "rossmann")
    print("series:", sales.shape[1], "days:", sales.shape[0], "features:", features.shape[1])
    data, _ = make_windows(sales, features, train_end=train_end, val_end=val_end)
    for name, (X, y, dates) in data.items():
        print(f"{name:5s} X{X.shape} y{y.shape} {dates[0].date()} -> {dates[-1].date()}")
