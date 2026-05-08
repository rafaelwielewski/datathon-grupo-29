from __future__ import annotations

import numpy as np
import pytest

from src.models.baseline import (
    LogisticRegressionBaseline,
    MajorityClassBaseline,
    PriorRateBaseline,
    classification_metrics,
)
from src.models.predictor import (
    FlightParams,
    _flatten_feature_store_result,
    _normalize_feature_key,
    _params_from_feature_store,
)


def test_classification_metrics_perfect():
    y = np.array([0, 1, 0, 1, 1])
    m = classification_metrics(y, y, y_proba=y.astype(float))
    assert m['precision'] == pytest.approx(1.0)
    assert m['recall'] == pytest.approx(1.0)
    assert m['f1'] == pytest.approx(1.0)
    assert m['auc'] == pytest.approx(1.0)


def test_classification_metrics_no_proba_excludes_auc():
    y = np.array([0, 1, 0, 1])
    pred = np.array([1, 1, 0, 0])
    m = classification_metrics(y, pred)
    assert {'precision', 'recall', 'f1'} <= set(m.keys())
    assert 'auc' not in m


def test_majority_baseline_predicts_most_frequent():
    y_train = np.array([0] * 70 + [1] * 30)
    rng = np.random.default_rng(0)
    X = rng.random((100, 3))
    model = MajorityClassBaseline().fit(X, y_train)
    preds = model.predict(rng.random((20, 3)))
    assert set(preds.tolist()) == {0}


def test_majority_baseline_evaluate_keys():
    rng = np.random.default_rng(0)
    y = np.array([0] * 60 + [1] * 40)
    X = rng.random((100, 3))
    result = MajorityClassBaseline().fit(X, y).evaluate(X, y)
    assert {'precision', 'recall', 'f1'} <= set(result.keys())


def test_prior_rate_baseline_below_threshold_predicts_zero():
    y_train = np.array([0] * 80 + [1] * 20)  # prior = 0.20
    rng = np.random.default_rng(0)
    X = rng.random((100, 3))
    # threshold=0.5 > prior=0.20 → predict all 0
    preds = PriorRateBaseline(threshold=0.5).fit(X, y_train).predict(rng.random((5, 3)))
    assert set(preds.tolist()) == {0}


def test_logistic_regression_baseline_learns_simple_pattern():
    rng = np.random.default_rng(0)
    X = rng.random((200, 1))
    y = (X[:, 0] > 0.5).astype(int)
    model = LogisticRegressionBaseline().fit(X, y)
    preds = model.predict(np.array([[0.1], [0.9]]))
    assert preds[0] == 0
    assert preds[1] == 1
    m = model.evaluate(X, y)
    assert m['auc'] > 0.9


# --- predictor pure functions ---


def test_normalize_feature_key_colon():
    assert _normalize_feature_key('flight_features:YEAR') == 'YEAR'


def test_normalize_feature_key_double_underscore():
    assert _normalize_feature_key('view__MONTH') == 'MONTH'


def test_normalize_feature_key_plain():
    assert _normalize_feature_key('AIRLINE') == 'AIRLINE'


def test_flatten_feature_store_result_skips_flight_id():
    result = {'flight_id': [0], 'flight_features:YEAR': [2015]}
    flat = _flatten_feature_store_result(result)
    assert 'flight_id' not in flat
    assert flat['YEAR'] == 2015


def test_flatten_feature_store_result_empty_list():
    result = {'flight_features:MONTH': []}
    flat = _flatten_feature_store_result(result)
    assert flat['MONTH'] is None


def test_flatten_feature_store_result_scalar():
    result = {'flight_features:AIRLINE': 'AA'}
    flat = _flatten_feature_store_result(result)
    assert flat['AIRLINE'] == 'AA'


def _valid_features() -> dict:
    return {
        'YEAR': 2015,
        'MONTH': 6,
        'DAY': 15,
        'DAY_OF_WEEK': 2,
        'sched_dep_hour': 8,
        'sched_dep_minute': 30,
        'sched_arr_hour': 11,
        'sched_arr_minute': 0,
        'DISTANCE': 1000.0,
        'SCHEDULED_TIME': 150.0,
        'AIRLINE': 'AA',
        'ORIGIN_AIRPORT': 'ATL',
        'DESTINATION_AIRPORT': 'LAX',
    }


def test_params_from_feature_store_ok():
    params = _params_from_feature_store(_valid_features())
    assert isinstance(params, FlightParams)
    assert params.airline == 'AA'
    assert params.scheduled_departure == 830
    assert params.scheduled_arrival == 1100


def test_params_from_feature_store_missing_raises():
    features = _valid_features()
    del features['AIRLINE']
    with pytest.raises(ValueError, match='Missing required features'):
        _params_from_feature_store(features)
