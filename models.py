"""The paper's models: S-CNN-LSTM, P-CNN-LSTM and the LSTM baseline (Keras)."""
from tensorflow import keras
from tensorflow.keras import layers

# paper Table 2 defaults
LSTM_UNITS = 100
FILTERS = (128, 256, 500)
POOL_SIZES = (2, 7, 10)
KERNEL_SIZE = 3          # not given in the paper (our choice)
DENSE_UNITS = 100        # parallel LSTM branch, not given in the paper (our choice)
INNER_DROPOUT = 0.2
FINAL_DROPOUT = 0.3
LEARNING_RATE = 0.001


def init():
    """Gaussian weight initialisation, mean 0 and std 0.05 (paper)."""
    return keras.initializers.RandomNormal(mean=0.0, stddev=0.05)


def conv_block(x):
    """Three Conv1D + AveragePooling stages, then Flatten."""
    for filters, pool in zip(FILTERS, POOL_SIZES):
        x = layers.Conv1D(filters, KERNEL_SIZE, padding="same", activation="relu",
                          kernel_initializer=init())(x)
        x = layers.AveragePooling1D(pool, padding="same")(x)
    return layers.Flatten()(x)


def stacked_cnn_lstm(n_lags, n_features, n_outputs):
    """S-CNN-LSTM: LSTM -> Dropout -> CNN block -> Dropout -> Dense."""
    inputs = keras.Input(shape=(n_lags, n_features))
    x = layers.LSTM(LSTM_UNITS, activation="relu", return_sequences=True,
                    kernel_initializer=init())(inputs)
    x = layers.Dropout(INNER_DROPOUT)(x)
    x = conv_block(x)
    x = layers.Dropout(FINAL_DROPOUT)(x)
    outputs = layers.Dense(n_outputs, kernel_initializer=init())(x)
    return compile_model(keras.Model(inputs, outputs, name="S-CNN-LSTM"))


def parallel_cnn_lstm(n_lags, n_features, n_outputs):
    """P-CNN-LSTM: an LSTM branch and a CNN branch, concatenated, then Dense."""
    inputs = keras.Input(shape=(n_lags, n_features))

    lstm = layers.LSTM(LSTM_UNITS, activation="relu", kernel_initializer=init())(inputs)
    lstm = layers.Dense(DENSE_UNITS, activation="relu", kernel_initializer=init())(lstm)
    lstm = layers.Dropout(INNER_DROPOUT)(lstm)

    cnn = conv_block(inputs)
    cnn = layers.Dropout(INNER_DROPOUT)(cnn)

    x = layers.Concatenate()([lstm, cnn])
    x = layers.Dropout(FINAL_DROPOUT)(x)
    outputs = layers.Dense(n_outputs, kernel_initializer=init())(x)
    return compile_model(keras.Model(inputs, outputs, name="P-CNN-LSTM"))


def lstm_baseline(n_lags, n_features, n_outputs):
    """Paper's LSTM benchmark: LSTM(100) -> Dense."""
    inputs = keras.Input(shape=(n_lags, n_features))
    x = layers.LSTM(LSTM_UNITS, activation="relu", kernel_initializer=init())(inputs)
    outputs = layers.Dense(n_outputs, kernel_initializer=init())(x)
    return compile_model(keras.Model(inputs, outputs, name="LSTM"))


def compile_model(model):
    """Adam, MSE loss, MAE metric (paper)."""
    model.compile(optimizer=keras.optimizers.Adam(LEARNING_RATE), loss="mse", metrics=["mae"])
    return model


MODELS = {
    "s_cnn_lstm": stacked_cnn_lstm,
    "p_cnn_lstm": parallel_cnn_lstm,
    "lstm": lstm_baseline,
}
