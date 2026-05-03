from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# --- Tools ---


def test_predict_flight_delay_returns_expected_keys():
    payload = json.dumps(
        {
            'airline': 'AA',
            'origin': 'ATL',
            'destination': 'LAX',
            'month': 6,
            'day': 15,
            'day_of_week': 2,
            'scheduled_departure': 800,
            'scheduled_arrival': 1100,
            'distance': 1950,
            'scheduled_time': 340,
        }
    )

    with patch('src.models.predictor.run_prediction', return_value=(0.45, 0.607)):
        from src.agent.tools import predict_flight_delay

        result = json.loads(predict_flight_delay.invoke(payload))

    assert 'delayed' in result
    assert 'delayed_probability' in result
    assert 'threshold' in result
    assert isinstance(result['delayed'], bool)
    assert 0.0 <= result['delayed_probability'] <= 1.0


def test_predict_flight_delay_invalid_json_returns_error():
    from src.agent.tools import predict_flight_delay

    result = json.loads(predict_flight_delay.invoke('not json'))
    assert 'error' in result


def test_predict_flight_delay_above_threshold_is_delayed():
    with patch('src.models.predictor.run_prediction', return_value=(0.80, 0.607)):
        from src.agent.tools import predict_flight_delay

        result = json.loads(predict_flight_delay.invoke('{"airline":"AA","origin":"ATL","destination":"LAX"}'))
    assert result['delayed'] is True
    assert result['delayed_probability'] == pytest.approx(0.80)


def test_get_airport_delay_stats_returns_data(tmp_path: Path):
    stats = {'ATL': {'delay_rate': 0.21, 'avg_delay_minutes': 8.5, 'total_flights': 50000}}
    (tmp_path / 'airport_stats.json').write_text(json.dumps(stats))

    with patch('src.agent.tools.ARTIFACTS_DIR', tmp_path):
        from src.agent.tools import get_airport_delay_stats

        result = json.loads(get_airport_delay_stats.invoke('ATL'))

    assert result['airport'] == 'ATL'
    assert result['delay_rate'] == 0.21


def test_get_airport_delay_stats_unknown_code_returns_error(tmp_path: Path):
    (tmp_path / 'airport_stats.json').write_text(json.dumps({'ATL': {}}))

    with patch('src.agent.tools.ARTIFACTS_DIR', tmp_path):
        from src.agent.tools import get_airport_delay_stats

        result = json.loads(get_airport_delay_stats.invoke('ZZZ'))

    assert 'error' in result


def test_get_airline_delay_stats_returns_data(tmp_path: Path):
    stats = {'DL': {'delay_rate': 0.17, 'avg_delay_minutes': 5.8, 'name': 'Delta Air Lines'}}
    (tmp_path / 'airline_stats.json').write_text(json.dumps(stats))

    with patch('src.agent.tools.ARTIFACTS_DIR', tmp_path):
        from src.agent.tools import get_airline_delay_stats

        result = json.loads(get_airline_delay_stats.invoke('DL'))

    assert result['airline'] == 'DL'
    assert result['delay_rate'] == 0.17


def test_get_airline_delay_stats_unknown_code_returns_error(tmp_path: Path):
    (tmp_path / 'airline_stats.json').write_text(json.dumps({'AA': {}}))

    with patch('src.agent.tools.ARTIFACTS_DIR', tmp_path):
        from src.agent.tools import get_airline_delay_stats

        result = json.loads(get_airline_delay_stats.invoke('ZZ'))

    assert 'error' in result


def test_get_all_tools_has_three_tools():
    from src.agent.tools import get_all_tools

    tools = get_all_tools()
    names = [t.name for t in tools]
    assert 'predict_flight_delay' in names
    assert 'get_airport_delay_stats' in names
    assert 'get_airline_delay_stats' in names


# --- RAG ---


def test_build_knowledge_base_loads_markdown(tmp_path: Path):
    kb_dir = tmp_path / 'kb'
    kb_dir.mkdir()
    (kb_dir / 'test.md').write_text('Flight delay risk factors weather congestion aircraft aviation')

    cfg = {'rag': {'knowledge_base_dirs': [str(kb_dir)], 'chunk_size': 200, 'chunk_overlap': 20}}
    from src.agent.rag_pipeline import build_knowledge_base_docs

    docs = build_knowledge_base_docs(cfg)
    assert len(docs) >= 1


def test_build_knowledge_base_missing_dir_returns_empty(tmp_path: Path):
    cfg = {'rag': {'knowledge_base_dirs': [str(tmp_path / 'nonexistent')], 'chunk_size': 200, 'chunk_overlap': 20}}
    from src.agent.rag_pipeline import build_knowledge_base_docs

    docs = build_knowledge_base_docs(cfg)
    assert docs == []


# --- Agent ---


def test_build_agent_executor_accepts_mock_llm():
    from src.agent.react_agent import build_agent_executor
    from src.agent.tools import get_airline_delay_stats, get_airport_delay_stats

    mock_llm = MagicMock()
    mock_llm.bind_tools = MagicMock(return_value=mock_llm)
    cfg = {'agent': {'verbose': False, 'max_iterations': 3, 'handle_parsing_errors': True}}
    executor = build_agent_executor(llm=mock_llm, tools=[get_airport_delay_stats, get_airline_delay_stats], cfg=cfg)
    assert executor is not None


def test_invoke_agent_normalizes_output():
    from langchain_core.messages import AIMessage, HumanMessage

    from src.agent.react_agent import invoke_agent

    mock_agent = MagicMock()
    mock_agent.invoke.return_value = {
        'messages': [
            HumanMessage(content='Will flight AA123 be delayed?'),
            AIMessage(content='Based on historical data, AA has a 20% delay rate on this route.'),
        ]
    }
    result = invoke_agent(mock_agent, 'Will flight AA123 be delayed?')
    assert 'delay' in result['output'].lower()
    assert isinstance(result['intermediate_steps'], list)


def test_load_config_returns_expected_keys(tmp_path: Path):
    from src.agent.react_agent import _load_config

    cfg_content = """
llm:
  provider: github
  model: gpt-4.1
  base_url: https://models.inference.ai.azure.com
  temperature: 0.0
  max_tokens: 2048
rag:
  embedding_model: all-MiniLM-L6-v2
  chunk_size: 256
  chunk_overlap: 25
  k_results: 2
  knowledge_base_dirs: []
agent:
  max_iterations: 5
  verbose: false
  handle_parsing_errors: true
"""
    cfg_file = tmp_path / 'agent_config.yaml'
    cfg_file.write_text(cfg_content)
    cfg = _load_config(cfg_file)
    assert {'llm', 'rag', 'agent'} <= set(cfg.keys())


def test_build_llm_returns_chat_openai():
    from langchain_openai import ChatOpenAI

    from src.agent.react_agent import build_llm

    os.environ.setdefault('GITHUB_TOKEN', 'github_pat_test')
    cfg = {
        'llm': {
            'provider': 'github',
            'model': 'gpt-4.1',
            'base_url': 'https://models.inference.ai.azure.com',
            'temperature': 0.0,
            'max_tokens': 512,
        }
    }
    llm = build_llm(cfg)
    assert isinstance(llm, ChatOpenAI)
    assert llm.model_name == 'gpt-4.1'
