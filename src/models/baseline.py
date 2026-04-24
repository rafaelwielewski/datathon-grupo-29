from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape = float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-9))) * 100)
    return {'mae': mae, 'rmse': rmse, 'mape': mape}


def directional_accuracy(true_price: np.ndarray, pred_price: np.ndarray, close_t: np.ndarray) -> float:
    true_sign = np.sign((true_price - close_t).reshape(-1))
    pred_sign = np.sign((pred_price - close_t).reshape(-1))
    return float((true_sign == pred_sign).mean() * 100)


class NaiveBaseline:
    """Prevê close(t+H) = close(t) — delta zero."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'NaiveBaseline':  # noqa: ARG002
        return self

    def predict(self, close_t: np.ndarray) -> np.ndarray:
        return close_t.reshape(-1, 1)

    def evaluate(self, true_price: np.ndarray, close_t: np.ndarray) -> dict[str, float]:
        pred = self.predict(close_t)
        m = _metrics(true_price.reshape(-1), pred.reshape(-1))
        m['directional_accuracy'] = directional_accuracy(true_price, pred, close_t)
        return m


class SMABaseline:
    """Prevê usando a SMA de `window` dias como estimativa do preço futuro."""

    def __init__(self, window: int = 60) -> None:
        self.window = window

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SMABaseline':  # noqa: ARG002
        return self

    def predict(self, close_series: np.ndarray) -> np.ndarray:
        import pandas as pd
        s = pd.Series(close_series.reshape(-1))
        return s.rolling(self.window).mean().values.reshape(-1, 1)

    def evaluate(
        self,
        true_price: np.ndarray,
        close_t: np.ndarray,
        close_series: np.ndarray,
    ) -> dict[str, float]:
        pred = self.predict(close_series)
        mask = ~np.isnan(pred.reshape(-1))
        m = _metrics(true_price.reshape(-1)[mask], pred.reshape(-1)[mask])
        m['directional_accuracy'] = directional_accuracy(
            true_price[mask], pred.reshape(-1, 1)[mask], close_t[mask]
        )
        return m


def build_lstm_model(
    lookback: int = 60,
    n_features: int = 16,
    lstm_units: list[int] | None = None,
    dense_units: int = 16,
    dropout: float = 0.2,
    recurrent_dropout: float = 0.05,
    learning_rate: float = 5e-4,
    clipnorm: float = 1.0,
    huber_delta: float = 1.0,
):
    """Constrói o modelo LSTM de 2 camadas usado no datathon."""
    import os
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

    from tensorflow import keras
    from tensorflow.keras import layers

    if lstm_units is None:
        lstm_units = [64, 32]

    model = keras.Sequential([
        layers.Input(shape=(lookback, n_features)),
        layers.LSTM(lstm_units[0], return_sequences=True, recurrent_dropout=recurrent_dropout),
        layers.Dropout(dropout),
        layers.LSTM(lstm_units[1], recurrent_dropout=recurrent_dropout),
        layers.Dropout(dropout),
        layers.Dense(dense_units, activation='relu'),
        layers.Dense(1),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=clipnorm),
        loss=keras.losses.Huber(delta=huber_delta),
    )
    return model
