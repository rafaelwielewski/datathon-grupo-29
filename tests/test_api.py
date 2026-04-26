from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """TestClient com agente mockado — sem chamadas reais ao LLM."""
    from src.serving import app as app_module

    mock_executor = MagicMock()
    mock_executor.invoke.return_value = {
        'messages': [
            MagicMock(content='test question', type='human'),
            MagicMock(content='AAPL está com RSI neutro. Previsão D+5: alta de $3.20.', type='ai'),
        ]
    }

    app_module._get_agent.cache_clear()
    with patch.object(app_module, '_get_agent', return_value=mock_executor):
        yield TestClient(app_module.app)


def test_health_ok(client: TestClient):
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json()['status'] == 'ok'


def test_query_returns_answer(client: TestClient):
    response = client.post('/query', json={'question': 'Qual a previsão para AAPL?'})
    assert response.status_code == 200
    data = response.json()
    assert 'answer' in data
    assert len(data['answer']) > 0


def test_query_empty_rejected(client: TestClient):
    response = client.post('/query', json={'question': ''})
    assert response.status_code == 422


def test_query_missing_field_rejected(client: TestClient):
    response = client.post('/query', json={})
    assert response.status_code == 422


def test_query_too_long_rejected(client: TestClient):
    response = client.post('/query', json={'question': 'x' * 1001})
    assert response.status_code == 422


def test_docs_available(client: TestClient):
    response = client.get('/docs')
    assert response.status_code == 200
