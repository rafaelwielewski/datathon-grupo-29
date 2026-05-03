from __future__ import annotations

import numpy as np
import pytest

from src.models.baseline import (
    LogisticRegressionBaseline,
    MajorityClassBaseline,
    PriorRateBaseline,
    classification_metrics,
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
