"""Gradio demo: pick a Rossmann store and compare the models' test-period forecasts.

    python app.py      (then open the printed local URL)
"""
import gradio as gr
import matplotlib.pyplot as plt
import pandas as pd

from dataset import load
from evaluate import metrics, predict_model

MODEL_DIR = "results/rossmann/models"
RUNS = {
    "S-CNN-LSTM": "s_cnn_lstm_s42",
    "P-CNN-LSTM": "p_cnn_lstm_s42",
    "P-CNN-LSTM + next-day features": "p_cnn_lstm_nextday_s42",
}

# predict the whole test period once at start-up
sales, is_open, features, ends = load("rossmann")
predictions = {}
for label, run in RUNS.items():
    dates, predicted = predict_model(f"{MODEL_DIR}/{run}.json", sales, features, ends, "test")
    predictions[label] = pd.DataFrame(predicted, index=dates, columns=sales.columns)


def forecast(store):
    actual = sales.loc[dates, store]
    scored = ((is_open.loc[dates, store] == 1) & (actual > 0)).values   # Open Days only

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(dates, actual, color="black", linewidth=2, label="actual")
    rows = []
    for label, pred in predictions.items():
        ax.plot(dates, pred[store], label=label)
        scores = metrics(actual.values, pred[store].values, scored)
        rows.append({"model": label, **{k: round(v, 2) for k, v in scores.items()}})
    ax.set_title(f"Store {store}: daily sales, May-July 2015")
    ax.set_ylabel("sales")
    ax.legend()
    fig.tight_layout()
    return fig, pd.DataFrame(rows)


demo = gr.Interface(
    fn=forecast,
    inputs=gr.Dropdown(list(sales.columns), value=sales.columns[0], label="Store"),
    outputs=[gr.Plot(label="Forecast vs actual"), gr.Dataframe(label="Errors on open days")],
    title="Rossmann sales forecasting with hybrid CNN-LSTM",
    description="One-step-ahead forecasts on the test period (seed 42 models).",
)

if __name__ == "__main__":
    demo.launch()
