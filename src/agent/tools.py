from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yfinance as yf
from langchain.tools import tool

from src.features.feature_engineering import FEATURE_COLS, LOOKBACK, build_features

ARTIFACTS_DIR = Path('data/processed/artifacts')
METRICS_PATH = ARTIFACTS_DIR / 'metrics.json'


@tool
def predict_price_delta(symbol: str = 'AAPL') -> str:
    """Predicts the D+5 closing price delta for AAPL using the LSTM ONNX model.

    Args:
        symbol: Stock ticker symbol. Currently only AAPL is supported.

    Returns:
        JSON with current price, predicted delta, predicted D+5 price, and direction.
    """
    import joblib
    import onnxruntime as ort

    sess = ort.InferenceSession(str(ARTIFACTS_DIR / 'final_model.onnx'))
    scaler_X = joblib.load(ARTIFACTS_DIR / 'scaler_X.joblib')
    scaler_y = joblib.load(ARTIFACTS_DIR / 'scaler_y.joblib')

    df = yf.download('AAPL', period=f'{LOOKBACK + 30}d', progress=False)
    if df is None or df.empty:
        return json.dumps({'error': 'Failed to download market data.'})

    # Preço atual = último fechamento real do df (antes do dropna do build_features)
    import pandas as pd

    current_price = float(pd.Series(df['Close'].values.flatten()).dropna().iloc[-1])

    feats = build_features(df)

    if len(feats) < LOOKBACK:
        return json.dumps({'error': 'Insufficient data for prediction.'})

    X = feats[FEATURE_COLS].values[-LOOKBACK:]
    X_scaled = scaler_X.transform(X).reshape(1, LOOKBACK, len(FEATURE_COLS)).astype(np.float32)

    pred_scaled = sess.run(None, {sess.get_inputs()[0].name: X_scaled})[0]
    pred_delta = float(scaler_y.inverse_transform(pred_scaled)[0, 0])

    return json.dumps(
        {
            'symbol': 'AAPL',
            'current_price_usd': round(current_price, 2),
            'predicted_delta_d5_usd': round(pred_delta, 2),
            'predicted_price_d5_usd': round(current_price + pred_delta, 2),
            'direction': 'UP' if pred_delta > 0 else 'DOWN',
            'model': 'LSTM ONNX (lookback=60d, horizon=5d)',
        }
    )


@tool
def get_technical_indicators(symbol: str = 'AAPL') -> str:
    """Returns the latest technical indicators for a stock: RSI, MACD, SMAs, volatility.

    Args:
        symbol: Stock ticker symbol (e.g. AAPL, MSFT).

    Returns:
        JSON with RSI-14, MACD, MACD signal, SMA-7, SMA-21, vol-21 and current price.
    """
    from src.features.feature_engineering import rsi as compute_rsi  # noqa: I001
    import pandas as pd

    df = yf.download(symbol, period='90d', progress=False)
    if df is None or df.empty:
        return json.dumps({'error': f'Failed to download data for {symbol}.'})

    # Computa indicadores direto no df bruto — sem o shift(-5) do target que dropa os últimos 5 dias
    close = pd.Series(df['Close'].values.flatten(), index=df.index, dtype=float)
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    macd_signal = macd.ewm(span=9, adjust=False).mean()

    indicators = pd.DataFrame(
        {
            'close': close,
            'sma_7': close.rolling(7).mean(),
            'sma_21': close.rolling(21).mean(),
            'macd': macd,
            'macd_signal': macd_signal,
            'rsi_14': compute_rsi(close, 14),
            'vol_21': close.pct_change().rolling(21).std(),
        }
    ).dropna()

    latest = indicators.iloc[-1]
    last_date = str(pd.DatetimeIndex(indicators.index)[-1].date())
    rsi_val = float(latest['rsi_14'])

    return json.dumps(
        {
            'symbol': symbol,
            'date': last_date,
            'close_usd': round(float(latest['close']), 2),
            'rsi_14': round(rsi_val, 2),
            'rsi_signal': 'OVERBOUGHT' if rsi_val > 70 else 'OVERSOLD' if rsi_val < 30 else 'NEUTRAL',
            'macd': round(float(latest['macd']), 4),
            'macd_signal_line': round(float(latest['macd_signal']), 4),
            'macd_crossover': 'BULLISH' if latest['macd'] > latest['macd_signal'] else 'BEARISH',
            'sma_7': round(float(latest['sma_7']), 2),
            'sma_21': round(float(latest['sma_21']), 2),
            'trend': 'UP' if latest['sma_7'] > latest['sma_21'] else 'DOWN',
            'vol_21_daily': round(float(latest['vol_21']), 6),
        }
    )


@tool
def calculate_position_risk(position_size: str = '100') -> str:
    """Calculates risk metrics for an AAPL stock position based on model error statistics.

    Args:
        position_size: Number of AAPL shares as a string (e.g. '100', '500').

    Returns:
        JSON with MAE, VaR at 95% and 99%, total exposure, and risk classification.
    """
    size = int(position_size)

    with open(METRICS_PATH) as f:
        metrics = json.load(f)

    mae = metrics['model']['mae_price']
    rmse = metrics['model']['rmse_price']
    dir_acc = metrics['model']['directional_accuracy_pct']

    return json.dumps(
        {
            'position_size': size,
            'mae_per_share_usd': round(mae, 2),
            'expected_error_usd': round(mae * size, 2),
            'var_95_per_share_usd': round(rmse * 2, 2),
            'var_95_total_usd': round(rmse * 2 * size, 2),
            'var_99_per_share_usd': round(rmse * 3, 2),
            'var_99_total_usd': round(rmse * 3 * size, 2),
            'directional_accuracy_pct': round(dir_acc, 2),
            'risk_level': 'HIGH' if size > 500 else 'MEDIUM' if size > 100 else 'LOW',
            'requires_human_review': size > 500,
        }
    )


def get_all_tools() -> list:
    return [predict_price_delta, get_technical_indicators, calculate_position_risk]
