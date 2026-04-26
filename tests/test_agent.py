from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd


def _make_mock_feats(n: int = 70) -> pd.DataFrame:
    """DataFrame mínimo simulando saída de build_features."""
    rng = np.random.default_rng(0)
    close = 150.0 + np.cumsum(rng.normal(0, 1, n))
    dates = pd.date_range('2024-01-01', periods=n, freq='B')
    return pd.DataFrame(
        {
            'close': close,
            'high': close + 1,
            'low': close - 1,
            'open': close,
            'volume': 1e6,
            'ret_1': 0.001,
            'log_ret_1': 0.001,
            'sma_7': close,
            'sma_21': close,
            'ema_12': close,
            'ema_26': close,
            'macd': 0.5,
            'macd_signal': 0.3,
            'rsi_14': 55.0,
            'vol_7': 0.01,
            'vol_21': 0.012,
            'close_t': close,
            'close_t_h': close + 3.0,
            'y_delta_h': 3.0,
        },
        index=dates,
    )


def _make_mock_ohlcv(n: int = 90, rsi_value: float = 55.0) -> pd.DataFrame:
    """DataFrame OHLCV no formato yfinance para testar get_technical_indicators."""
    rng = np.random.default_rng(1)
    # Gerar preços com trend para RSI controlável
    close_vals = 150.0 + np.cumsum(rng.normal(0.05 if rsi_value > 50 else -0.05, 0.5, n))
    dates = pd.date_range('2024-01-01', periods=n, freq='B')
    # yfinance retorna MultiIndex columns — simular com colunas simples
    return pd.DataFrame(
        {
            'Close': close_vals,
            'High': close_vals + 1,
            'Low': close_vals - 1,
            'Open': close_vals + rng.normal(0, 0.3, n),
            'Volume': rng.integers(1_000_000, 10_000_000, n).astype(float),
        },
        index=dates,
    )


# --- Tools ---


def test_get_technical_indicators_expected_keys():
    with patch('src.agent.tools.yf.download') as mock_dl:
        mock_dl.return_value = _make_mock_ohlcv()
        from src.agent.tools import get_technical_indicators

        result = json.loads(get_technical_indicators.invoke('AAPL'))
        assert 'rsi_14' in result
        assert 'macd' in result
        assert 'close_usd' in result
        assert 'rsi_signal' in result
        assert result['symbol'] == 'AAPL'


def test_get_technical_indicators_date_is_latest():
    """Verifica que a data retornada é a mais recente (não 5 dias atrás)."""
    df = _make_mock_ohlcv(n=90)
    expected_date = str(df.index[-1].date())
    with patch('src.agent.tools.yf.download') as mock_dl:
        mock_dl.return_value = df
        from src.agent.tools import get_technical_indicators

        result = json.loads(get_technical_indicators.invoke('AAPL'))
        assert result['date'] == expected_date


def test_get_technical_indicators_rsi_neutral():
    with patch('src.agent.tools.yf.download') as mock_dl:
        mock_dl.return_value = _make_mock_ohlcv(rsi_value=55.0)
        from src.agent.tools import get_technical_indicators

        result = json.loads(get_technical_indicators.invoke('AAPL'))
        assert result['rsi_signal'] in ('NEUTRAL', 'OVERBOUGHT', 'OVERSOLD')


def test_get_technical_indicators_rsi_overbought():
    """RSI acima de 70 com preços consistentemente subindo."""
    rng = np.random.default_rng(42)
    n = 90
    close = 150.0 + np.cumsum(np.abs(rng.normal(1.5, 0.1, n)))  # só sobe
    dates = pd.date_range('2024-01-01', periods=n, freq='B')
    df = pd.DataFrame({'Close': close, 'High': close + 1, 'Low': close - 1, 'Open': close, 'Volume': 1e6}, index=dates)
    with patch('src.agent.tools.yf.download') as mock_dl:
        mock_dl.return_value = df
        from src.agent.tools import get_technical_indicators

        result = json.loads(get_technical_indicators.invoke('AAPL'))
        assert result['rsi_signal'] == 'OVERBOUGHT'


def test_calculate_position_risk_low(tmp_path: Path):
    metrics = {
        'model': {
            'mae_price': 7.36,
            'rmse_price': 9.86,
            'mape_price_pct': 3.27,
            'directional_accuracy_pct': 52.5,
        },
        'baselines': {},
    }
    (tmp_path / 'metrics.json').write_text(json.dumps(metrics))
    with patch('src.agent.tools.METRICS_PATH', tmp_path / 'metrics.json'):
        from src.agent.tools import calculate_position_risk

        result = json.loads(calculate_position_risk.invoke('50'))
        assert result['risk_level'] == 'LOW'
        assert result['position_size'] == 50
        assert 'var_95_total_usd' in result
        assert not result['requires_human_review']


def test_calculate_position_risk_high(tmp_path: Path):
    metrics = {
        'model': {
            'mae_price': 7.36,
            'rmse_price': 9.86,
            'mape_price_pct': 3.27,
            'directional_accuracy_pct': 52.5,
        },
        'baselines': {},
    }
    (tmp_path / 'metrics.json').write_text(json.dumps(metrics))
    with patch('src.agent.tools.METRICS_PATH', tmp_path / 'metrics.json'):
        from src.agent.tools import calculate_position_risk

        result = json.loads(calculate_position_risk.invoke('1000'))
        assert result['risk_level'] == 'HIGH'
        assert result['requires_human_review']


def test_get_all_tools_has_three():
    from src.agent.tools import get_all_tools

    tools = get_all_tools()
    assert len(tools) >= 3


# --- RAG ---


def test_build_knowledge_base_docs_loads_markdown(tmp_path: Path):
    kb_dir = tmp_path / 'kb'
    kb_dir.mkdir()
    (kb_dir / 'test.md').write_text('AAPL stock RSI indicator momentum financial risk analysis')

    cfg = {'rag': {'knowledge_base_dirs': [str(kb_dir)], 'chunk_size': 200, 'chunk_overlap': 20}}
    from src.agent.rag_pipeline import build_knowledge_base_docs

    docs = build_knowledge_base_docs(cfg)
    assert len(docs) >= 1
    assert any('AAPL' in d.page_content or 'RSI' in d.page_content for d in docs)


def test_build_knowledge_base_docs_missing_dir_returns_empty(tmp_path: Path):
    cfg = {'rag': {'knowledge_base_dirs': [str(tmp_path / 'nonexistent')], 'chunk_size': 200, 'chunk_overlap': 20}}
    from src.agent.rag_pipeline import build_knowledge_base_docs

    docs = build_knowledge_base_docs(cfg)
    assert docs == []


# --- Agent ---


def test_build_agent_executor_accepts_mock_llm():
    from src.agent.react_agent import build_agent_executor
    from src.agent.tools import calculate_position_risk, get_technical_indicators

    mock_llm = MagicMock()
    mock_llm.bind_tools = MagicMock(return_value=mock_llm)

    cfg = {'agent': {'verbose': False, 'max_iterations': 3, 'handle_parsing_errors': True}}
    executor = build_agent_executor(
        llm=mock_llm,
        tools=[get_technical_indicators, calculate_position_risk],
        cfg=cfg,
    )
    assert executor is not None


def test_invoke_agent_normalizes_output():
    from langchain_core.messages import AIMessage, HumanMessage  # noqa: I001
    from src.agent.react_agent import invoke_agent

    mock_agent = MagicMock()
    mock_agent.invoke.return_value = {
        'messages': [
            HumanMessage(content='test question'),
            AIMessage(content='Final answer about AAPL.'),
        ]
    }
    result = invoke_agent(mock_agent, 'test question')
    assert result['output'] == 'Final answer about AAPL.'
    assert isinstance(result['intermediate_steps'], list)


def test_load_config_returns_expected_keys(tmp_path: Path):
    from src.agent.react_agent import _load_config

    cfg_content = """
llm:
  provider: github
  model: gpt-4o-mini
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
    assert 'llm' in cfg
    assert 'rag' in cfg
    assert 'agent' in cfg
    assert cfg['llm']['model'] == 'gpt-4o-mini'


def test_build_llm_returns_chat_openai():
    import os

    from langchain_openai import ChatOpenAI  # noqa: I001

    from src.agent.react_agent import build_llm

    cfg = {
        'llm': {
            'provider': 'github',
            'model': 'gpt-4o-mini',
            'base_url': 'https://models.inference.ai.azure.com',
            'temperature': 0.0,
            'max_tokens': 512,
        }
    }
    os.environ.setdefault('GITHUB_TOKEN', 'github_pat_test')
    llm = build_llm(cfg)
    assert isinstance(llm, ChatOpenAI)
    assert llm.model_name == 'gpt-4o-mini'


def test_predict_price_delta_insufficient_data():
    with (
        patch('src.agent.tools.yf.download') as mock_dl,
        patch('src.agent.tools.build_features') as mock_bf,
        patch('onnxruntime.InferenceSession'),
        patch('joblib.load'),
    ):
        mock_dl.return_value = _make_mock_ohlcv(n=20)  # df com Close válido
        mock_bf.return_value = _make_mock_feats(n=10)  # menos que LOOKBACK=60
        from src.agent.tools import predict_price_delta

        result = json.loads(predict_price_delta.invoke('AAPL'))
        assert 'error' in result


def test_predict_price_delta_download_failure():
    with (
        patch('src.agent.tools.yf.download') as mock_dl,
        patch('onnxruntime.InferenceSession'),
        patch('joblib.load'),
    ):
        mock_dl.return_value = MagicMock(empty=True)
        from src.agent.tools import predict_price_delta

        result = json.loads(predict_price_delta.invoke('AAPL'))
        assert 'error' in result


def test_get_technical_indicators_download_failure():
    with patch('src.agent.tools.yf.download') as mock_dl:
        mock_dl.return_value = MagicMock(empty=True)

        from src.agent.tools import get_technical_indicators

        result = json.loads(get_technical_indicators.invoke('AAPL'))
        assert 'error' in result
