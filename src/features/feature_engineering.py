from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLS = [
    'close', 'high', 'low', 'open', 'volume',
    'ret_1', 'log_ret_1',
    'sma_7', 'sma_21', 'ema_12', 'ema_26',
    'macd', 'macd_signal',
    'rsi_14',
    'vol_7', 'vol_21',
]

HORIZON = 5
LOOKBACK = 60


def _to_series(x: pd.DataFrame | pd.Series) -> pd.Series:
    if isinstance(x, pd.DataFrame):
        return x.iloc[:, 0]
    return x


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    rs = up.rolling(period).mean() / (down.rolling(period).mean() + 1e-9)
    return 100 - (100 / (1 + rs))


def build_features(df: pd.DataFrame, horizon: int = HORIZON) -> pd.DataFrame:
    close = _to_series(df['Close']).astype(float)
    feats = pd.DataFrame({
        'close': close,
        'high': _to_series(df['High']).astype(float),
        'low': _to_series(df['Low']).astype(float),
        'open': _to_series(df['Open']).astype(float),
        'volume': _to_series(df['Volume']).astype(float),
    })

    feats['ret_1'] = feats['close'].pct_change()
    feats['log_ret_1'] = np.log(feats['close']).diff()
    feats['sma_7'] = feats['close'].rolling(7).mean()
    feats['sma_21'] = feats['close'].rolling(21).mean()
    feats['ema_12'] = feats['close'].ewm(span=12, adjust=False).mean()
    feats['ema_26'] = feats['close'].ewm(span=26, adjust=False).mean()
    feats['macd'] = feats['ema_12'] - feats['ema_26']
    feats['macd_signal'] = feats['macd'].ewm(span=9, adjust=False).mean()
    feats['rsi_14'] = rsi(feats['close'], 14)
    feats['vol_7'] = feats['ret_1'].rolling(7).std()
    feats['vol_21'] = feats['ret_1'].rolling(21).std()

    feats = feats.dropna().copy()
    feats['close_t'] = feats['close']
    feats['close_t_h'] = feats['close'].shift(-horizon)
    feats['y_delta_h'] = feats['close_t_h'] - feats['close_t']

    return feats.dropna().copy()


def create_sequences(
    X: np.ndarray,
    y: np.ndarray,
    close_t: np.ndarray,
    close_t_h: np.ndarray,
    dates: np.ndarray,
    lookback: int = LOOKBACK,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    Xw, yw, ct, cth, dt = [], [], [], [], []
    for i in range(lookback - 1, len(X)):
        Xw.append(X[i - lookback + 1 : i + 1])
        yw.append(y[i])
        ct.append(close_t[i])
        cth.append(close_t_h[i])
        dt.append(dates[i])
    return (
        np.array(Xw, dtype=np.float32),
        np.array(yw, dtype=np.float32).reshape(-1, 1),
        np.array(ct, dtype=np.float32).reshape(-1, 1),
        np.array(cth, dtype=np.float32).reshape(-1, 1),
        pd.to_datetime(np.array(dt)),
    )


def temporal_split(
    dates: pd.DatetimeIndex,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    n = len(dates)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    n_train = n - n_val - n_test
    train_end = dates[n_train - 1]
    val_end = dates[n_train + n_val - 1]
    return train_end, val_end
