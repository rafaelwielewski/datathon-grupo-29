from __future__ import annotations

import pandas as pd

from src.features.feature_engineering import (
    FEATURE_COLS,
    LOOKBACK,
    build_features,
    create_sequences,
    rsi,
    temporal_split,
)


def test_rsi_range(raw_ohlcv: pd.DataFrame) -> None:
    values = rsi(raw_ohlcv["Close"])
    valid = values.dropna()
    assert (valid >= 0).all() and (valid <= 100).all()


def test_build_features_columns(raw_ohlcv: pd.DataFrame) -> None:
    df = build_features(raw_ohlcv)
    for col in FEATURE_COLS:
        assert col in df.columns, f"feature '{col}' ausente"
    assert "y_delta_h" in df.columns
    assert "close_t" in df.columns
    assert "close_t_h" in df.columns


def test_build_features_no_nan(raw_ohlcv: pd.DataFrame) -> None:
    df = build_features(raw_ohlcv)
    assert not df[FEATURE_COLS + ["y_delta_h"]].isna().any().any()


def test_build_features_target_consistency(raw_ohlcv: pd.DataFrame) -> None:
    df = build_features(raw_ohlcv)
    expected = df["close_t_h"] - df["close_t"]
    pd.testing.assert_series_equal(df["y_delta_h"], expected, check_names=False)


def test_create_sequences_shape(feature_df: pd.DataFrame) -> None:
    X = feature_df[FEATURE_COLS].values.astype("float32")
    y = feature_df["y_delta_h"].values.astype("float32")
    ct = feature_df["close_t"].values.astype("float32")
    cth = feature_df["close_t_h"].values.astype("float32")
    dates = feature_df.index.values

    Xw, yw, ctw, cthw, dw = create_sequences(X, y, ct, cth, dates, lookback=LOOKBACK)

    expected_samples = len(X) - LOOKBACK + 1
    assert Xw.shape == (expected_samples, LOOKBACK, len(FEATURE_COLS))
    assert yw.shape == (expected_samples, 1)
    assert len(dw) == expected_samples


def test_temporal_split_ordering(feature_df: pd.DataFrame) -> None:
    dates = feature_df.index
    train_end, val_end = temporal_split(dates)
    assert train_end < val_end
    assert val_end <= dates[-1]


def test_temporal_split_ratios(feature_df: pd.DataFrame) -> None:
    dates = feature_df.index
    n = len(dates)
    train_end, val_end = temporal_split(dates, val_ratio=0.15, test_ratio=0.15)

    n_train = (dates <= train_end).sum()
    n_val = ((dates > train_end) & (dates <= val_end)).sum()
    n_test = (dates > val_end).sum()

    assert n_train + n_val + n_test == n
    assert abs(n_test / n - 0.15) < 0.02
    assert abs(n_val / n - 0.15) < 0.02
