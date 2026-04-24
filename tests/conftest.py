from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def raw_ohlcv() -> pd.DataFrame:
    """DataFrame com colunas OHLCV no formato yfinance (200 pregões sintéticos)."""
    rng = np.random.default_rng(42)
    n = 200
    close = 150.0 + np.cumsum(rng.normal(0, 1, n))
    high = close + rng.uniform(0.5, 2.0, n)
    low = close - rng.uniform(0.5, 2.0, n)
    open_ = close + rng.normal(0, 0.5, n)
    volume = rng.integers(1_000_000, 10_000_000, n).astype(float)
    dates = pd.date_range("2022-01-03", periods=n, freq="B")
    return pd.DataFrame(
        {"Close": close, "High": high, "Low": low, "Open": open_, "Volume": volume},
        index=dates,
    )


@pytest.fixture
def feature_df(raw_ohlcv: pd.DataFrame) -> pd.DataFrame:
    from src.features.feature_engineering import build_features
    return build_features(raw_ohlcv)
