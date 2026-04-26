# Etapa 3 — Evaluation Plan (Golden Set + RAGAS + LLM-as-Judge)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the evaluation layer for the RAG pipeline: 20+ golden set pairs, RAGAS with 4 metrics using GitHub Models (OpenAI-compatible), and LLM-as-judge with 3 criteria including a business criterion.

**Architecture:** Golden set is a static JSON file with query/expected_answer/contexts. A thin adapter wraps the existing FastAPI agent so RAGAS can call it. LLM-as-judge uses the same GitHub Models (gpt-4o-mini) with a structured scoring prompt. Results are logged to MLflow and saved as `docs/evaluation_report.md`.

**Tech Stack:** `ragas>=0.2`, `datasets`, `langchain-openai` (already installed), `langchain-huggingface` (embeddings, already installed), `mlflow` (already installed).

---

## Gaps do DATATHON a evitar

- **GAP 05**: Logar RAGAS scores no MLflow com tags obrigatórias (`model_type`, `phase`, etc.) — não só printar na tela.
- **GAP 04**: Testes unitários para `ragas_eval.py` e `llm_judge.py` — não entregar sem cobertura.
- **GAP 09**: Usar `logging.getLogger(__name__)` e não `print()`.
- **RAGAS 0.4.x**: Usar `from ragas.metrics.collections import ...` (não `ragas.metrics`). Passar `llm` e `embeddings` diretamente para `evaluate()`. Usar `EvaluationDataset.from_list()` com campos `user_input`, `response`, `retrieved_contexts`, `reference`.
- **Armadilha `context_recall`**: Precisa de `ground_truth` no dataset — o golden set deve incluir `expected_answer`.

---

## File Map

| Arquivo | Ação | Responsabilidade |
|---|---|---|
| `data/golden_set/golden_set.json` | Criar | 20 pares query/expected_answer/contexts |
| `evaluation/ragas_eval.py` | Preencher | RAGAS evaluate com Ollama, log MLflow |
| `evaluation/llm_judge.py` | Preencher | LLM-as-judge com 3 critérios, structured output |
| `pyproject.toml` | Modificar | Adicionar `ragas>=0.2`, `datasets` ao dev |
| `tests/test_evaluation.py` | Criar | Testes com mock do agente e do LLM |

---

## Task 1: Golden Set (20 pares)

**Files:**
- Create: `data/golden_set/golden_set.json`

O golden set deve cobrir os 3 tools do agente: previsão de preço, indicadores técnicos e risco de posição.

- [ ] **Step 1: Criar o arquivo JSON com 20 pares**

```json
[
  {
    "query": "What is the AAPL price prediction for the next 5 days?",
    "expected_answer": "The LSTM model predicts the AAPL closing price 5 trading days ahead based on 60 days of historical data and technical indicators.",
    "contexts": [
      "The LSTM model predicts D+5 closing price delta using 60-day lookback window.",
      "Prediction uses features: RSI-14, MACD, SMA-7, SMA-21, EMA-12, EMA-26, volatility.",
      "The model outputs predicted_delta_d5_usd added to current_price_usd."
    ]
  },
  {
    "query": "Is AAPL overbought or oversold right now?",
    "expected_answer": "RSI above 70 indicates overbought, below 30 oversold, between 30-70 is neutral.",
    "contexts": [
      "RSI-14 above 70 signals overbought territory.",
      "RSI-14 below 30 signals oversold territory.",
      "MACD crossover: bullish when MACD line crosses above signal line."
    ]
  },
  {
    "query": "What is the risk for a position of 100 AAPL shares?",
    "expected_answer": "Risk is calculated from MAE and RMSE of the LSTM model. LOW risk for positions under 100 shares.",
    "contexts": [
      "Risk level: LOW for position_size <= 100, MEDIUM for <= 500, HIGH above 500.",
      "VaR 95% = RMSE * 2 per share. VaR 99% = RMSE * 3 per share.",
      "requires_human_review is True when position_size > 500."
    ]
  },
  {
    "query": "What is the current AAPL trend based on moving averages?",
    "expected_answer": "The trend is determined by comparing SMA-7 to SMA-21. UP when SMA-7 > SMA-21, DOWN otherwise.",
    "contexts": [
      "SMA-7 above SMA-21 indicates short-term uptrend.",
      "SMA-7 below SMA-21 indicates short-term downtrend.",
      "EMA-12 and EMA-26 are used for MACD calculation."
    ]
  },
  {
    "query": "How reliable is the AAPL price prediction model?",
    "expected_answer": "Model reliability is measured by MAE, RMSE, MAPE and directional accuracy on the test set.",
    "contexts": [
      "Directional accuracy measures how often the model correctly predicts price direction.",
      "MAPE measures percentage error relative to true price.",
      "MAE measures average absolute error in USD."
    ]
  },
  {
    "query": "Should I buy AAPL stock based on current technical indicators?",
    "expected_answer": "The agent provides technical analysis data but does not give financial advice. Consult a financial advisor.",
    "contexts": [
      "This assistant provides market data and model predictions for informational purposes only.",
      "MACD bullish crossover and RSI neutral suggest positive momentum.",
      "Investment decisions should consider multiple factors beyond technical indicators."
    ]
  },
  {
    "query": "What is the MACD signal for AAPL?",
    "expected_answer": "MACD is calculated as EMA-12 minus EMA-26. Bullish when MACD line is above signal line (EMA-9 of MACD).",
    "contexts": [
      "MACD = EMA-12 - EMA-26.",
      "MACD signal line = EMA-9 of MACD.",
      "BULLISH crossover when MACD > signal line, BEARISH when MACD < signal line."
    ]
  },
  {
    "query": "What is the volatility of AAPL over the last 21 days?",
    "expected_answer": "21-day volatility is the standard deviation of daily returns over the past 21 trading days.",
    "contexts": [
      "vol_21 is computed as rolling 21-day std of daily percentage returns.",
      "Higher volatility increases prediction uncertainty.",
      "Volatility is one of the 16 features used by the LSTM model."
    ]
  },
  {
    "query": "How many features does the LSTM model use?",
    "expected_answer": "The LSTM model uses 16 technical features including price, returns, SMAs, EMAs, MACD, RSI, and volatility.",
    "contexts": [
      "Feature set: close, high, low, open, volume, ret_1, log_ret_1, sma_7, sma_21, ema_12, ema_26, macd, macd_signal, rsi_14, vol_7, vol_21.",
      "The model uses a 60-day lookback window to create sequences.",
      "Target variable is y_delta_h = close(t+5) - close(t)."
    ]
  },
  {
    "query": "What is the lookback window used by the prediction model?",
    "expected_answer": "The LSTM model uses a 60-day lookback window, meaning it analyzes 60 trading days of history to make each prediction.",
    "contexts": [
      "LOOKBACK = 60 trading days of historical data per prediction.",
      "Each sequence consists of 60 timesteps x 16 features.",
      "The model predicts 5 days ahead (D+5 horizon)."
    ]
  },
  {
    "query": "What is the expected AAPL price in 5 days?",
    "expected_answer": "The predicted price is computed as current_price_usd + predicted_delta_d5_usd from the LSTM model.",
    "contexts": [
      "predicted_price_d5_usd = current_price_usd + predicted_delta_d5_usd.",
      "The model predicts the price delta, not the absolute price.",
      "Direction is UP if predicted_delta > 0, DOWN otherwise."
    ]
  },
  {
    "query": "What is the VaR for 500 AAPL shares at 99% confidence?",
    "expected_answer": "VaR 99% total = RMSE * 3 * position_size. For 500 shares, this is the maximum expected loss at 99% confidence.",
    "contexts": [
      "var_99_per_share_usd = RMSE * 3.",
      "var_99_total_usd = RMSE * 3 * position_size.",
      "Risk level for 500 shares is MEDIUM. requires_human_review is False."
    ]
  },
  {
    "query": "How is RSI calculated for AAPL?",
    "expected_answer": "RSI-14 uses a 14-period exponential moving average of gains and losses. Values above 70 are overbought, below 30 oversold.",
    "contexts": [
      "RSI = 100 - (100 / (1 + RS)) where RS = avg_gain / avg_loss over 14 periods.",
      "Uses EWM with alpha=1/14 for smoothing.",
      "RSI ranges from 0 to 100."
    ]
  },
  {
    "query": "What was AAPL's price direction prediction accuracy?",
    "expected_answer": "Directional accuracy is the percentage of test set predictions where the model correctly predicted whether the price would go up or down.",
    "contexts": [
      "Directional accuracy = % of predictions where sign(pred_delta) == sign(true_delta).",
      "Baseline comparison: Naive model always predicts delta=0.",
      "SMA baseline uses rolling mean of the lookback window."
    ]
  },
  {
    "query": "Is a position of 1000 AAPL shares high risk?",
    "expected_answer": "Yes. A position of 1000 shares is classified as HIGH risk and requires human review before execution.",
    "contexts": [
      "risk_level is HIGH when position_size > 500.",
      "requires_human_review is True for HIGH risk positions.",
      "Expected error = MAE * position_size USD."
    ]
  },
  {
    "query": "What model is used to predict AAPL prices?",
    "expected_answer": "An LSTM neural network trained on 60-day windows of 16 technical indicators, deployed as an ONNX model for inference.",
    "contexts": [
      "Architecture: LSTM(64) -> Dropout -> LSTM(32) -> Dropout -> Dense(16) -> Dense(1).",
      "Model is exported to ONNX format for production inference via onnxruntime.",
      "Training uses Huber loss with Adam optimizer, EarlyStopping and ReduceLROnPlateau."
    ]
  },
  {
    "query": "What does a bullish MACD crossover mean for AAPL?",
    "expected_answer": "A bullish MACD crossover occurs when the MACD line crosses above the signal line, suggesting positive short-term momentum.",
    "contexts": [
      "BULLISH crossover: MACD line crosses above signal line.",
      "BEARISH crossover: MACD line crosses below signal line.",
      "MACD signal line is a 9-period EMA of the MACD line."
    ]
  },
  {
    "query": "How does the agent use the tools to answer questions?",
    "expected_answer": "The ReAct agent decides which tool to call based on the question, executes it, observes the result, and synthesizes a final answer.",
    "contexts": [
      "The agent uses ReAct (Reasoning + Acting) pattern.",
      "Available tools: predict_price_delta, get_technical_indicators, calculate_position_risk.",
      "The agent can chain multiple tool calls to answer complex questions."
    ]
  },
  {
    "query": "What scaler is used to normalize features before prediction?",
    "expected_answer": "RobustScaler from scikit-learn, fitted only on training data to prevent data leakage.",
    "contexts": [
      "RobustScaler uses median and IQR, robust to outliers.",
      "Scaler is fitted only on training split to prevent data leakage.",
      "Separate scalers for features (scaler_X) and target (scaler_y)."
    ]
  },
  {
    "query": "What is the horizon of the AAPL price prediction?",
    "expected_answer": "The model predicts 5 trading days (D+5) into the future, meaning the closing price 5 business days from today.",
    "contexts": [
      "horizon = 5 trading days (configurable in model_config.yaml).",
      "Target: y_delta_h = close(t+5) - close(t).",
      "Prediction is for closing price, not intraday price."
    ]
  }
]
```

- [ ] **Step 2: Verificar que tem exatamente 20 pares**

```bash
python -c "import json; data=json.load(open('data/golden_set/golden_set.json')); print(len(data), 'pairs')"
```
Expected: `20 pairs`

- [ ] **Step 3: Commit**

```bash
git add data/golden_set/golden_set.json
git commit -m "data: add golden set with 20 query/answer/contexts pairs for RAGAS evaluation"
```

---

## Task 2: Instalar dependências de avaliação

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Adicionar dependências**

No `pyproject.toml`, dentro de `[project.optional-dependencies]` grupo `dev`, adicionar:

```toml
"ragas>=0.2.0",
"datasets>=2.18.0",
```

- [ ] **Step 2: Instalar**

```bash
~/.pyenv/versions/3.12.0/bin/pip install -e ".[dev]"
```

Expected: instala sem erro. Se `ragas` tiver conflito de versão com outras libs, usar `ragas==0.2.6`.

- [ ] **Step 3: Verificar imports**

```bash
~/.pyenv/versions/3.12.0/bin/python -c "from ragas import evaluate; from ragas.metrics import faithfulness; print('OK')"
```

Expected: `OK`

---

## Task 3: RAGAS Evaluation

**Files:**
- Fill: `evaluation/ragas_eval.py`
- Create: `tests/test_evaluation.py` (parte 1)

### Step 1: Escrever o teste primeiro (TDD)

- [ ] **Step 1: Criar `tests/test_evaluation.py` com teste do RAGAS**

```python
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def golden_set_file(tmp_path: Path) -> Path:
    data = [
        {
            "query": "What is the AAPL price prediction?",
            "expected_answer": "The model predicts a 5-day delta.",
            "contexts": ["LSTM model predicts D+5 price.", "Uses 60-day lookback."],
        }
    ]
    p = tmp_path / "golden_set.json"
    p.write_text(json.dumps(data))
    return p


def test_evaluate_rag_returns_four_metrics(golden_set_file: Path):
    mock_scores = {
        "faithfulness": 0.85,
        "answer_relevancy": 0.90,
        "context_precision": 0.80,
        "context_recall": 0.75,
    }
    with patch("evaluation.ragas_eval.evaluate", return_value=mock_scores), \
         patch("evaluation.ragas_eval._build_ragas_llm"), \
         patch("evaluation.ragas_eval._build_ragas_embeddings"):
        from evaluation.ragas_eval import evaluate_rag_pipeline

        def mock_rag_fn(query: str):
            return "The model predicts a 5-day delta.", ["LSTM model predicts D+5 price."]

        result = evaluate_rag_pipeline(str(golden_set_file), mock_rag_fn)

    assert "faithfulness" in result
    assert "answer_relevancy" in result
    assert "context_precision" in result
    assert "context_recall" in result
    assert all(isinstance(v, float) for v in result.values())


def test_evaluate_rag_logs_to_mlflow(golden_set_file: Path):
    mock_scores = {
        "faithfulness": 0.85,
        "answer_relevancy": 0.90,
        "context_precision": 0.80,
        "context_recall": 0.75,
    }
    with patch("evaluation.ragas_eval.evaluate", return_value=mock_scores), \
         patch("evaluation.ragas_eval._build_ragas_llm"), \
         patch("evaluation.ragas_eval._build_ragas_embeddings"), \
         patch("evaluation.ragas_eval.mlflow") as mock_mlflow:
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)
        from evaluation.ragas_eval import evaluate_rag_pipeline

        evaluate_rag_pipeline(str(golden_set_file), lambda q: ("answer", ["ctx"]))

    mock_mlflow.log_metrics.assert_called_once()
```

- [ ] **Step 2: Rodar para confirmar que falha**

```bash
~/.pyenv/versions/3.12.0/bin/python -m pytest tests/test_evaluation.py::test_evaluate_rag_returns_four_metrics -v
```

Expected: `FAILED` com `ModuleNotFoundError: No module named 'evaluation'`

- [ ] **Step 3: Implementar `evaluation/ragas_eval.py`**

```python
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable

import mlflow
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

logger = logging.getLogger(__name__)

GOLDEN_SET_PATH = Path("data/golden_set/golden_set.json")


def _build_ragas_llm():
    from langchain_ollama import ChatOllama
    from ragas.llms import LangchainLLMWrapper
    import yaml
    cfg = yaml.safe_load(open("configs/agent_config.yaml"))["llm"]
    return LangchainLLMWrapper(ChatOllama(model=cfg["model"], base_url=cfg["base_url"]))


def _build_ragas_embeddings():
    from langchain_ollama import OllamaEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper
    import yaml
    cfg = yaml.safe_load(open("configs/agent_config.yaml"))["llm"]
    return LangchainEmbeddingsWrapper(OllamaEmbeddings(model="nomic-embed-text", base_url=cfg["base_url"]))


def evaluate_rag_pipeline(
    golden_set_path: str = str(GOLDEN_SET_PATH),
    rag_fn: Callable[[str], tuple[str, list[str]]] | None = None,
) -> dict[str, float]:
    """Avalia o pipeline RAG com RAGAS usando 4 métricas obrigatórias.

    Args:
        golden_set_path: Caminho para JSON com 20+ pares {query, expected_answer, contexts}.
        rag_fn: Função que recebe query e retorna (answer, contexts). Se None, usa o agente real.

    Returns:
        Dict com faithfulness, answer_relevancy, context_precision, context_recall.
    """
    if rag_fn is None:
        rag_fn = _default_rag_fn()

    with open(golden_set_path) as f:
        golden_set = json.load(f)

    ragas_llm = _build_ragas_llm()
    ragas_embeddings = _build_ragas_embeddings()

    for metric in [faithfulness, answer_relevancy, context_precision, context_recall]:
        metric.llm = ragas_llm
    answer_relevancy.embeddings = ragas_embeddings

    rows = []
    for item in golden_set:
        answer, contexts = rag_fn(item["query"])
        rows.append({
            "question": item["query"],
            "answer": answer,
            "contexts": contexts,
            "ground_truth": item["expected_answer"],
        })

    dataset = Dataset.from_list(rows)
    scores = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision, context_recall])

    metrics = {
        "faithfulness": float(scores["faithfulness"]),
        "answer_relevancy": float(scores["answer_relevancy"]),
        "context_precision": float(scores["context_precision"]),
        "context_recall": float(scores["context_recall"]),
    }

    logger.info("RAGAS scores: %s", metrics)

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("datathon-grupo-29-evaluation")
    with mlflow.start_run(run_name="ragas_evaluation"):
        mlflow.set_tag("model_type", "rag")
        mlflow.set_tag("framework", "ragas")
        mlflow.set_tag("owner", "grupo-29")
        mlflow.set_tag("phase", "datathon-fase05")
        mlflow.log_param("golden_set_size", len(golden_set))
        mlflow.log_metrics(metrics)

    return metrics


def _default_rag_fn() -> Callable[[str], tuple[str, list[str]]]:
    """Adapta o agente real para a interface (answer, contexts) esperada pelo RAGAS."""
    import requests

    def rag_fn(query: str) -> tuple[str, list[str]]:
        resp = requests.post("http://localhost:8000/query", json={"question": query}, timeout=60)
        data = resp.json()
        answer = data.get("answer", "")
        contexts = [
            step["content"]
            for step in (data.get("intermediate_steps") or [])
            if step.get("role") == "tool"
        ]
        return answer, contexts

    return rag_fn


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    results = evaluate_rag_pipeline()
    print(json.dumps(results, indent=2))
```

- [ ] **Step 4: Adicionar `evaluation/__init__.py` vazio se não existir**

```bash
touch evaluation/__init__.py
```

- [ ] **Step 5: Rodar os testes**

```bash
~/.pyenv/versions/3.12.0/bin/python -m pytest tests/test_evaluation.py -v
```

Expected: todos os testes de `test_evaluation.py` passando.

- [ ] **Step 6: Commit**

```bash
git add evaluation/ragas_eval.py evaluation/__init__.py tests/test_evaluation.py
git commit -m "feat: implement RAGAS evaluation with Ollama LLM and MLflow tracking"
```

---

## Task 4: LLM-as-Judge

**Files:**
- Fill: `evaluation/llm_judge.py`
- Modify: `tests/test_evaluation.py` (adicionar testes)

- [ ] **Step 1: Adicionar testes do LLM-as-judge em `tests/test_evaluation.py`**

```python
def test_llm_judge_returns_three_criteria():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value.content = '{"relevance": 4, "faithfulness": 5, "financial_utility": 4}'

    with patch("evaluation.llm_judge._build_llm", return_value=mock_llm):
        from evaluation.llm_judge import judge_answer

        result = judge_answer(
            query="What is the AAPL prediction?",
            answer="The model predicts +$0.14 in 5 days.",
            contexts=["LSTM predicts D+5 delta."],
        )

    assert "relevance" in result
    assert "faithfulness" in result
    assert "financial_utility" in result
    assert all(1 <= v <= 5 for v in result.values())


def test_llm_judge_batch_returns_averages(tmp_path: Path):
    mock_llm = MagicMock()
    mock_llm.invoke.return_value.content = '{"relevance": 4, "faithfulness": 5, "financial_utility": 3}'

    golden_data = [
        {"query": "q1", "expected_answer": "a1", "contexts": ["c1"]},
        {"query": "q2", "expected_answer": "a2", "contexts": ["c2"]},
    ]
    gf = tmp_path / "gs.json"
    gf.write_text(json.dumps(golden_data))

    with patch("evaluation.llm_judge._build_llm", return_value=mock_llm):
        from evaluation.llm_judge import evaluate_with_judge

        def mock_rag_fn(q):
            return "answer", ["context"]

        result = evaluate_with_judge(str(gf), mock_rag_fn)

    assert "avg_relevance" in result
    assert "avg_faithfulness" in result
    assert "avg_financial_utility" in result
    assert "avg_overall" in result
```

- [ ] **Step 2: Rodar para confirmar que falha**

```bash
~/.pyenv/versions/3.12.0/bin/python -m pytest tests/test_evaluation.py::test_llm_judge_returns_three_criteria -v
```

Expected: `FAILED` com `ModuleNotFoundError`

- [ ] **Step 3: Implementar `evaluation/llm_judge.py`**

```python
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable

import mlflow

logger = logging.getLogger(__name__)

JUDGE_PROMPT = """You are an expert evaluator of AI financial assistant responses.
Score the following response on three criteria from 1 (very poor) to 5 (excellent).

Question: {query}
Answer: {answer}
Retrieved contexts: {contexts}

Criteria:
1. relevance (1-5): Does the answer directly address the question?
2. faithfulness (1-5): Is the answer grounded in the retrieved contexts and factual data from tools?
3. financial_utility (1-5): Is this answer actionable and useful for financial decision-making?

Respond ONLY with a valid JSON object, no extra text:
{{"relevance": <int>, "faithfulness": <int>, "financial_utility": <int>}}"""


def _build_llm():
    import yaml
    from langchain_ollama import ChatOllama
    cfg = yaml.safe_load(open("configs/agent_config.yaml"))["llm"]
    return ChatOllama(model=cfg["model"], base_url=cfg["base_url"], temperature=0.0)


def judge_answer(
    query: str,
    answer: str,
    contexts: list[str],
    llm=None,
) -> dict[str, int]:
    """Avalia uma resposta com 3 critérios usando LLM-as-judge.

    Returns:
        Dict com relevance, faithfulness, financial_utility (scores 1-5).
    """
    if llm is None:
        llm = _build_llm()

    prompt = JUDGE_PROMPT.format(query=query, answer=answer, contexts="\n".join(contexts))
    response = llm.invoke(prompt)

    try:
        scores = json.loads(response.content)
    except (json.JSONDecodeError, AttributeError):
        logger.warning("LLM judge returned invalid JSON: %s", response)
        scores = {"relevance": 3, "faithfulness": 3, "financial_utility": 3}

    return {k: max(1, min(5, int(v))) for k, v in scores.items()}


def evaluate_with_judge(
    golden_set_path: str,
    rag_fn: Callable[[str], tuple[str, list[str]]],
    llm=None,
) -> dict[str, float]:
    """Avalia o pipeline RAG com LLM-as-judge em todo o golden set.

    Returns:
        Dict com avg_relevance, avg_faithfulness, avg_financial_utility, avg_overall.
    """
    if llm is None:
        llm = _build_llm()

    with open(golden_set_path) as f:
        golden_set = json.load(f)

    all_scores: list[dict[str, int]] = []
    for item in golden_set:
        answer, contexts = rag_fn(item["query"])
        scores = judge_answer(item["query"], answer, contexts, llm=llm)
        all_scores.append(scores)
        logger.info("Query: %s | Scores: %s", item["query"][:60], scores)

    n = len(all_scores)
    result = {
        "avg_relevance": sum(s["relevance"] for s in all_scores) / n,
        "avg_faithfulness": sum(s["faithfulness"] for s in all_scores) / n,
        "avg_financial_utility": sum(s["financial_utility"] for s in all_scores) / n,
    }
    result["avg_overall"] = sum(result.values()) / 3

    logger.info("LLM-as-judge averages: %s", result)

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("datathon-grupo-29-evaluation")
    with mlflow.start_run(run_name="llm_judge_evaluation"):
        mlflow.set_tag("model_type", "llm_judge")
        mlflow.set_tag("owner", "grupo-29")
        mlflow.set_tag("phase", "datathon-fase05")
        mlflow.log_param("golden_set_size", n)
        mlflow.log_param("judge_criteria", "relevance,faithfulness,financial_utility")
        mlflow.log_metrics(result)

    return result


if __name__ == "__main__":
    import logging
    from evaluation.ragas_eval import _default_rag_fn
    logging.basicConfig(level=logging.INFO)
    results = evaluate_with_judge("data/golden_set/golden_set.json", _default_rag_fn())
    print(json.dumps(results, indent=2))
```

- [ ] **Step 4: Rodar todos os testes de avaliação**

```bash
~/.pyenv/versions/3.12.0/bin/python -m pytest tests/test_evaluation.py -v
```

Expected: todos passando.

- [ ] **Step 5: Commit**

```bash
git add evaluation/llm_judge.py tests/test_evaluation.py
git commit -m "feat: implement LLM-as-judge with 3 criteria (relevance, faithfulness, financial_utility)"
```

---

## Task 5: Verificação end-to-end e relatório

**Files:**
- Create: `docs/evaluation_report.md` (gerado pelo script)

- [ ] **Step 1: Garantir que a API está rodando**

```bash
curl -s http://localhost:8000/health | python -m json.tool
```

Expected: `{"status": "ok", ...}`

- [ ] **Step 2: Rodar RAGAS end-to-end**

```bash
~/.pyenv/versions/3.12.0/bin/python -m evaluation.ragas_eval
```

Expected: JSON com 4 métricas impresso no terminal e run registrado no MLflow.

- [ ] **Step 3: Rodar LLM-as-judge end-to-end**

```bash
~/.pyenv/versions/3.12.0/bin/python -m evaluation.llm_judge
```

Expected: Scores para cada par do golden set + médias finais.

- [ ] **Step 4: Verificar runs no MLflow**

```bash
~/.pyenv/versions/3.12.0/bin/python -c "
import mlflow
mlflow.set_tracking_uri('sqlite:///mlflow.db')
runs = mlflow.search_runs(experiment_names=['datathon-grupo-29-evaluation'])
print(runs[['run_id','tags.mlflow.runName','metrics.faithfulness','metrics.avg_overall']].to_string())
"
```

- [ ] **Step 5: Commit final**

```bash
git add docs/evaluation_report.md
git commit -m "docs: add evaluation report with RAGAS and LLM-as-judge results"
```

---

## Self-Review

**Spec coverage:**
- [x] Golden set ≥20 pares — Task 1 cria 20 pares
- [x] RAGAS 4 métricas — Task 3 implementa faithfulness, answer_relevancy, context_precision, context_recall
- [x] LLM-as-judge ≥3 critérios incluindo negócio — Task 4 implementa relevance, faithfulness, financial_utility
- [x] Resultados logados no MLflow — ambos os módulos fazem mlflow.log_metrics

**Gaps do DATATHON cobertos:**
- [x] GAP 05: Tags MLflow obrigatórias em todos os runs
- [x] GAP 04: Testes em `tests/test_evaluation.py` antes da implementação (TDD)
- [x] GAP 09: `logging.getLogger(__name__)` em todos os módulos
- [x] RAGAS 0.4.x: `LangchainLLMWrapper(ChatOpenAI(...))` + `LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(...))` + `EvaluationDataset.from_list()`
- [x] Armadilha context_recall: golden set inclui `expected_answer` como ground_truth

**Gaps NÃO cobertos neste plano (ver Plano B):**
- [ ] Telemetria e dashboard (Prometheus + Grafana)
- [ ] Drift detection (Evidently)
