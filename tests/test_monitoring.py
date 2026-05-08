from __future__ import annotations

import logging


def test_prometheus_metrics_are_defined():
    from src.monitoring.metrics import (
        ACTIVE_REQUESTS,
        FLIGHT_PREDICTION,
        REQUEST_COUNT,
        REQUEST_LATENCY,
    )

    assert REQUEST_COUNT is not None
    assert REQUEST_LATENCY is not None
    assert ACTIVE_REQUESTS is not None
    assert FLIGHT_PREDICTION is not None


def test_request_count_increments():
    from src.monitoring.metrics import REQUEST_COUNT

    before = REQUEST_COUNT.labels(method='GET', endpoint='/health', status_code='200')._value.get()
    REQUEST_COUNT.labels(method='GET', endpoint='/health', status_code='200').inc()
    after = REQUEST_COUNT.labels(method='GET', endpoint='/health', status_code='200')._value.get()
    assert after == before + 1


def test_active_requests_gauge():
    from src.monitoring.metrics import ACTIVE_REQUESTS

    ACTIVE_REQUESTS.inc()
    ACTIVE_REQUESTS.dec()
    assert ACTIVE_REQUESTS._value.get() >= 0


def test_business_metric_flight_prediction():
    from src.monitoring.metrics import FLIGHT_PREDICTION

    before = FLIGHT_PREDICTION.labels(prediction='DELAYED')._value.get()
    FLIGHT_PREDICTION.labels(prediction='DELAYED').inc()
    after = FLIGHT_PREDICTION.labels(prediction='DELAYED')._value.get()
    assert after == before + 1


def test_metrics_endpoint_returns_prometheus_format():
    from unittest.mock import MagicMock, patch

    from fastapi.testclient import TestClient

    with patch('src.serving.app._get_agent') as mock_agent:
        mock_agent.return_value = MagicMock()
        from src.serving.app import app

        client = TestClient(app)
        response = client.get('/metrics')

    assert response.status_code == 200
    assert 'api_requests_total' in response.text or 'text/plain' in response.headers.get('content-type', '')


def test_run_drift_report_returns_required_keys():
    from unittest.mock import MagicMock, patch

    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(0)
    cols = ['DISTANCE', 'SCHEDULED_TIME']
    ref_df = pd.DataFrame({c: rng.normal(0, 1, 100) for c in cols})
    cur_df = pd.DataFrame({c: rng.normal(0, 1, 50) for c in cols})

    mock_run = MagicMock()
    mock_run.dict.return_value = {
        'metrics': [
            {'metric_name': 'DriftedColumnsCount', 'value': {'share': 0.05}},
            {
                'metric_name': 'ValueDrift',
                'config': {'column': 'DISTANCE', 'threshold': 0.05},
                'value': 0.5,  # p-value > 0.05 means no drift
            },
        ]
    }
    mock_report = MagicMock()
    mock_report.run.return_value = mock_run

    with patch('evidently.Report', return_value=mock_report):
        from src.monitoring.drift import run_drift_report

        result = run_drift_report(ref_df, cur_df)

    assert 'drift_detected' in result
    assert 'drift_share' in result
    assert 'per_feature' in result
    assert isinstance(result['drift_share'], float)
    assert isinstance(result['drift_detected'], bool)
    assert result['drift_detected'] is False


def test_run_drift_report_detects_drift_above_threshold():
    from unittest.mock import MagicMock, patch

    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(1)
    cols = ['DISTANCE']
    ref_df = pd.DataFrame({c: rng.normal(0, 1, 100) for c in cols})
    cur_df = pd.DataFrame({c: rng.normal(5, 1, 50) for c in cols})

    mock_run = MagicMock()
    # 2 features, both with p-value well below Bonferroni alpha (0.05/2=0.025)
    mock_run.dict.return_value = {
        'metrics': [
            {'metric_name': 'ValueDrift', 'config': {'column': 'DISTANCE'}, 'value': 0.001},
            {'metric_name': 'ValueDrift', 'config': {'column': 'SCHEDULED_TIME'}, 'value': 0.002},
        ]
    }
    mock_report = MagicMock()
    mock_report.run.return_value = mock_run

    with patch('evidently.Report', return_value=mock_report):
        from src.monitoring.drift import run_drift_report

        result = run_drift_report(ref_df, cur_df, warning_threshold=0.1)

    assert result['drift_detected'] is True
    assert result['drift_share'] == 1.0  # 2/2 features drifted


def test_detect_and_log_drift_handles_missing_parquet(tmp_path):
    from unittest.mock import patch

    import yaml

    config = {
        'drift': {
            'warning_threshold': 0.1,
            'retrain_threshold': 0.2,
            'reference_parquet': str(tmp_path / 'nonexistent.parquet'),
            'reference_split': 'train',
            'current_split': 'test',
        }
    }
    cfg_file = tmp_path / 'monitoring_config.yaml'
    cfg_file.write_text(yaml.dump(config))

    with patch('src.monitoring.drift.CONFIG_PATH', cfg_file):
        from src.monitoring.drift import detect_and_log_drift

        result = detect_and_log_drift()

    assert 'error' in result


def test_drift_endpoint_returns_result():
    from unittest.mock import MagicMock, patch

    from fastapi.testclient import TestClient

    mock_drift_result = {
        'drift_detected': False,
        'drift_share': 0.05,
        'per_feature': {},
        'warning_threshold': 0.1,
        'retrain_threshold': 0.2,
    }

    with (
        patch('src.serving.app._get_agent', return_value=MagicMock()),
        patch('src.serving.app.detect_and_log_drift', return_value=mock_drift_result),
    ):
        from src.serving.app import app

        client = TestClient(app)
        response = client.get('/drift')

    assert response.status_code == 200
    data = response.json()
    assert 'drift_detected' in data
    assert 'drift_share' in data


def test_request_id_filter_injects_request_id():
    from src.serving.logging_config import RequestIDFilter

    f = RequestIDFilter()
    record = logging.LogRecord('test', logging.INFO, '', 0, 'msg', (), None)
    assert f.filter(record) is True
    assert hasattr(record, 'request_id')


def test_configure_logging_sets_handler():
    from src.serving.logging_config import configure_logging

    configure_logging()
    assert len(logging.root.handlers) >= 1
    assert any(isinstance(h, logging.StreamHandler) for h in logging.root.handlers)
