from __future__ import annotations

import numpy as np
import pytest

from src.models.baseline import NaiveBaseline, SMABaseline, _metrics, directional_accuracy


def _sample_arrays(n: int = 100) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    close_t = np.full(n, 150.0)
    close_t_h = close_t + rng.normal(0, 5, n)
    return close_t.reshape(-1, 1), close_t_h.reshape(-1, 1)


def test_metrics_perfect_prediction() -> None:
    y = np.array([1.0, 2.0, 3.0])
    m = _metrics(y, y)
    assert m['mae'] == pytest.approx(0.0)
    assert m['rmse'] == pytest.approx(0.0)
    assert m['mape'] == pytest.approx(0.0)


def test_metrics_keys() -> None:
    y = np.array([1.0, 2.0])
    m = _metrics(y, y * 1.1)
    assert {'mae', 'rmse', 'mape'} == set(m.keys())


def test_directional_accuracy_perfect() -> None:
    close_t = np.array([100.0, 100.0, 100.0])
    true_price = np.array([105.0, 95.0, 102.0])
    pred_price = true_price.copy()
    assert directional_accuracy(true_price, pred_price, close_t) == pytest.approx(100.0)


def test_directional_accuracy_zero() -> None:
    close_t = np.array([100.0, 100.0])
    true_price = np.array([105.0, 95.0])
    pred_price = np.array([95.0, 105.0])
    assert directional_accuracy(true_price, pred_price, close_t) == pytest.approx(0.0)


def test_naive_baseline_predict_shape() -> None:
    close_t, _ = _sample_arrays(50)
    model = NaiveBaseline().fit(np.zeros((50, 60, 16)), np.zeros(50))
    pred = model.predict(close_t)
    assert pred.shape == close_t.shape


def test_naive_baseline_predict_equals_close_t() -> None:
    close_t = np.array([[100.0], [200.0], [150.0]])
    model = NaiveBaseline()
    np.testing.assert_array_equal(model.predict(close_t), close_t)


def test_naive_baseline_evaluate_keys() -> None:
    close_t, close_t_h = _sample_arrays(80)
    model = NaiveBaseline()
    result = model.evaluate(close_t_h, close_t)
    assert {'mae', 'rmse', 'mape', 'directional_accuracy'} == set(result.keys())


def test_sma_baseline_predict_shape() -> None:
    close_t, _ = _sample_arrays(100)
    model = SMABaseline(window=10)
    pred = model.predict(close_t)
    assert pred.shape == close_t.shape


def test_sma_baseline_evaluate_keys() -> None:
    close_t, close_t_h = _sample_arrays(100)
    model = SMABaseline(window=10)
    result = model.evaluate(close_t_h, close_t, close_t)
    assert {'mae', 'rmse', 'mape', 'directional_accuracy'} == set(result.keys())


def test_build_lstm_model_output_shape() -> None:
    from src.models.baseline import build_lstm_model

    model = build_lstm_model(lookback=10, n_features=4, lstm_units=[8, 4], dense_units=4)
    X = np.random.default_rng(1).random((5, 10, 4)).astype('float32')
    out = model.predict(X, verbose=0)
    assert out.shape == (5, 1)
