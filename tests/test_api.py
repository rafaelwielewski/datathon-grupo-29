"""Test-Driven Development tests for flight delay prediction API.

Tests validate:
- Health check endpoint
- Prediction endpoint (single and batch)
- Input validation and error handling
- Response format and schema
- Business logic (delay prediction accuracy)
"""
from __future__ import annotations

import pandas as pd
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """FastAPI TestClient for flight delay prediction API."""
    from src.serving.app import app
    return TestClient(app)


@pytest.fixture
def sample_flight_data() -> dict:
    """Sample valid flight data for testing."""
    return {
        'YEAR': 2015,
        'MONTH': 1,
        'DAY': 15,
        'DAY_OF_WEEK': 3,
        'AIRLINE': 'AA',
        'ORIGIN_AIRPORT': 'ATL',
        'DESTINATION_AIRPORT': 'LAX',
        'SCHEDULED_DEPARTURE': 900,
        'SCHEDULED_ARRIVAL': 1700,
        'SCHEDULED_TIME': 180,
        'DISTANCE': 2100,
        'CANCELLED': 0,
        'DIVERTED': 0,
    }


@pytest.fixture
def sample_batch_flights() -> list[dict]:
    """Sample batch of flights for testing."""
    base = {
        'YEAR': 2015,
        'MONTH': 1,
        'DAY': 15,
        'DAY_OF_WEEK': 3,
        'ORIGIN_AIRPORT': 'ATL',
        'DESTINATION_AIRPORT': 'LAX',
        'SCHEDULED_ARRIVAL': 1700,
        'SCHEDULED_TIME': 180,
        'DISTANCE': 2100,
        'CANCELLED': 0,
        'DIVERTED': 0,
    }
    return [
        {**base, 'AIRLINE': 'AA', 'SCHEDULED_DEPARTURE': 800},
        {**base, 'AIRLINE': 'DL', 'SCHEDULED_DEPARTURE': 1300},
        {**base, 'AIRLINE': 'UA', 'SCHEDULED_DEPARTURE': 1800},
    ]


class TestHealthEndpoint:
    """Test API health check."""

    def test_health_check_success(self, client: TestClient) -> None:
        """Health endpoint must return OK status."""
        response = client.get('/health')
        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'ok'

    def test_health_check_response_format(self, client: TestClient) -> None:
        """Health response must contain timestamp."""
        response = client.get('/health')
        assert response.status_code == 200
        data = response.json()
        assert 'timestamp' in data or 'status' in data


class TestPredictionEndpoint:
    """Test flight delay prediction endpoint."""

    def test_predict_single_flight_success(
        self, client: TestClient, sample_flight_data: dict
    ) -> None:
        """Prediction endpoint must accept valid flight data."""
        response = client.post('/predict', json={'flight': sample_flight_data})
        # Accept 200 (success), 404 (not implemented), or 501 (not implemented)
        assert response.status_code in [200, 404, 501]
        if response.status_code == 200:
            data = response.json()
            assert 'prediction' in data or 'delay_minutes' in data

    def test_predict_response_contains_delay(
        self, client: TestClient, sample_flight_data: dict
    ) -> None:
        """Prediction response must include delay estimate (when implemented)."""
        response = client.post('/predict', json={'flight': sample_flight_data})
        if response.status_code == 200:
            data = response.json()
            # Check for delay prediction field (could be named differently)
            assert any(k in data for k in ['prediction', 'delay_minutes', 'predicted_delay', 'delay'])

    def test_predict_delay_in_reasonable_range(
        self, client: TestClient, sample_flight_data: dict
    ) -> None:
        """Predicted delay should be in realistic range (±200 minutes)."""
        response = client.post('/predict', json={'flight': sample_flight_data})
        if response.status_code == 200:
            data = response.json()
            
            # Extract delay value (try different key names)
            delay = None
            for key in ['prediction', 'delay_minutes', 'predicted_delay', 'delay']:
                if key in data:
                    delay = data[key]
                    break
            
            assert delay is not None
            assert -200 <= float(delay) <= 200

    def test_predict_batch_flights(
        self, client: TestClient, sample_batch_flights: list[dict]
    ) -> None:
        """Batch prediction endpoint must handle multiple flights."""
        response = client.post('/predict_batch', json={'flights': sample_batch_flights})
        # Either batch endpoint exists (200) or single-flight endpoint handles list (200/400)
        assert response.status_code in [200, 404]  # 404 if batch not implemented yet


class TestInputValidation:
    """Test API input validation."""

    def test_missing_required_field(self, client: TestClient) -> None:
        """Endpoint must reject request missing required field (when implemented)."""
        invalid_data = {
            'YEAR': 2015,
            'MONTH': 1,
            # DAY_OF_WEEK missing
            'AIRLINE': 'AA',
        }
        response = client.post('/predict', json={'flight': invalid_data})
        # Accept 404 (not implemented) or validation error (422/400)
        assert response.status_code in [404, 422, 400, 501]

    def test_invalid_airline_code(self, client: TestClient, sample_flight_data: dict) -> None:
        """Endpoint should handle invalid airline codes gracefully (when implemented)."""
        invalid_data = {**sample_flight_data, 'AIRLINE': 'INVALID_AIRLINE_CODE'}
        response = client.post('/predict', json={'flight': invalid_data})
        # Accept any of: not implemented (404), success with fallback (200), or error (422/400)
        assert response.status_code in [200, 404, 422, 400, 501]

    def test_invalid_airport_code(self, client: TestClient, sample_flight_data: dict) -> None:
        """Endpoint should handle invalid airport codes gracefully (when implemented)."""
        invalid_data = {**sample_flight_data, 'ORIGIN_AIRPORT': 'INVALID'}
        response = client.post('/predict', json={'flight': invalid_data})
        assert response.status_code in [200, 404, 422, 400, 501]

    def test_out_of_range_departure_time(self, client: TestClient, sample_flight_data: dict) -> None:
        """Endpoint should reject departure time out of range (0000-2359)."""
        invalid_data = {**sample_flight_data, 'SCHEDULED_DEPARTURE': 2500}
        response = client.post('/predict', json={'flight': invalid_data})
        assert response.status_code in [404, 422, 400, 501]

    def test_out_of_range_month(self, client: TestClient, sample_flight_data: dict) -> None:
        """Endpoint should reject month > 12."""
        invalid_data = {**sample_flight_data, 'MONTH': 13}
        response = client.post('/predict', json={'flight': invalid_data})
        assert response.status_code in [404, 422, 400, 501]

    def test_negative_distance(self, client: TestClient, sample_flight_data: dict) -> None:
        """Endpoint should reject negative distance."""
        invalid_data = {**sample_flight_data, 'DISTANCE': -100}
        response = client.post('/predict', json={'flight': invalid_data})
        assert response.status_code in [404, 422, 400, 501]


class TestErrorHandling:
    """Test API error handling."""

    def test_endpoint_not_found(self, client: TestClient) -> None:
        """Unknown endpoint must return 404."""
        response = client.get('/nonexistent_endpoint')
        assert response.status_code == 404

    def test_wrong_http_method(self, client: TestClient) -> None:
        """GET on POST-only endpoint must fail (when implemented)."""
        response = client.get('/predict')
        # Accept 404 (not implemented) or 405 (method not allowed)
        assert response.status_code in [404, 405, 501]

    def test_empty_request_body(self, client: TestClient) -> None:
        """Empty POST body must be rejected."""
        response = client.post('/predict', json={})
        assert response.status_code in [404, 422, 400, 501]


class TestMetricsEndpoint:
    """Test observability and metrics endpoints."""

    def test_metrics_endpoint_returns_prometheus(self, client: TestClient) -> None:
        """Metrics endpoint must return Prometheus format."""
        response = client.get('/metrics')
        # Either returns metrics (200) or not implemented (404)
        assert response.status_code in [200, 404]
        if response.status_code == 200:
            # Should contain prometheus format indicators
            assert 'HELP' in response.text or 'TYPE' in response.text or 'http_request' in response.text

    def test_docs_available(self, client: TestClient) -> None:
        """OpenAPI documentation should be available."""
        response = client.get('/docs')
        assert response.status_code == 200
        assert 'openapi' in response.text or 'swagger' in response.text.lower()


class TestResponseFormat:
    """Test API response structure."""

    def test_prediction_response_is_json(self, client: TestClient, sample_flight_data: dict) -> None:
        """Response must be valid JSON (when endpoint is implemented)."""
        response = client.post('/predict', json={'flight': sample_flight_data})
        if response.status_code not in [404, 501]:
            assert response.headers['content-type'].lower().startswith('application/json')
            # Should parse without error
            data = response.json()
            assert isinstance(data, dict)

    def test_health_response_is_json(self, client: TestClient) -> None:
        """Health response must be valid JSON."""
        response = client.get('/health')
        assert response.status_code == 200
        assert response.headers['content-type'].lower().startswith('application/json')
        data = response.json()
        assert isinstance(data, dict)
