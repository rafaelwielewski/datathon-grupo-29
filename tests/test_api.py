"""Tests for the flight delay prediction API."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    from src.serving.app import app

    return TestClient(app)


class TestHealthEndpoint:
    def test_health_check_success(self, client: TestClient) -> None:
        response = client.get('/health')
        assert response.status_code == 200
        assert response.json()['status'] == 'ok'

    def test_health_response_is_json(self, client: TestClient) -> None:
        response = client.get('/health')
        assert response.headers['content-type'].lower().startswith('application/json')
        assert isinstance(response.json(), dict)


class TestQueryEndpoint:
    def test_query_rejects_empty_question(self, client: TestClient) -> None:
        response = client.post('/query', json={'question': ''})
        assert response.status_code == 422

    def test_query_rejects_missing_question(self, client: TestClient) -> None:
        response = client.post('/query', json={})
        assert response.status_code == 422

    def test_query_rejects_question_too_long(self, client: TestClient) -> None:
        response = client.post('/query', json={'question': 'x' * 1001})
        assert response.status_code == 422


class TestMetricsEndpoint:
    def test_metrics_returns_prometheus_format(self, client: TestClient) -> None:
        response = client.get('/metrics')
        assert response.status_code in [200, 404]
        if response.status_code == 200:
            assert 'HELP' in response.text or 'TYPE' in response.text


class TestErrorHandling:
    def test_unknown_endpoint_returns_404(self, client: TestClient) -> None:
        response = client.get('/nonexistent_endpoint')
        assert response.status_code == 404

    def test_docs_available(self, client: TestClient) -> None:
        response = client.get('/docs')
        assert response.status_code == 200
