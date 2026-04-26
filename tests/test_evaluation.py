from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch


@staticmethod
def _mock_rag_fn(query: str) -> tuple[str, list[str]]:
    return 'The model predicts a 5-day delta.', ['LSTM model predicts D+5 price.']


# --- RAGAS ---


def test_evaluate_rag_returns_four_metrics(tmp_path: Path):
    data = [
        {
            'query': 'What is the AAPL price prediction?',
            'expected_answer': 'The model predicts a 5-day delta.',
            'contexts': ['LSTM model predicts D+5 price.', 'Uses 60-day lookback.'],
        }
    ]
    golden_set_file = tmp_path / 'golden_set.json'
    golden_set_file.write_text(json.dumps(data))

    mock_result = MagicMock()
    import pandas as pd

    mock_result.to_pandas.return_value = pd.DataFrame(
        {
            'faithfulness': [0.85],
            'answer_relevancy': [0.90],
            'context_precision': [0.80],
            'context_recall': [0.75],
        }
    )

    with (
        patch('evaluation.ragas_eval.evaluate', return_value=mock_result),
        patch('evaluation.ragas_eval._build_ragas_llm'),
        patch('evaluation.ragas_eval._build_ragas_embeddings'),
        patch('evaluation.ragas_eval.mlflow'),
    ):
        from evaluation.ragas_eval import evaluate_rag_pipeline

        result = evaluate_rag_pipeline(str(golden_set_file), _mock_rag_fn)

    assert 'faithfulness' in result
    assert 'answer_relevancy' in result
    assert 'context_precision' in result
    assert 'context_recall' in result
    assert all(isinstance(v, float) for v in result.values())


def test_evaluate_rag_logs_to_mlflow(tmp_path: Path):
    data = [
        {
            'query': 'What is the AAPL price prediction?',
            'expected_answer': 'The model predicts a 5-day delta.',
            'contexts': ['LSTM model predicts D+5 price.'],
        }
    ]
    golden_set_file = tmp_path / 'golden_set.json'
    golden_set_file.write_text(json.dumps(data))

    mock_result = MagicMock()
    import pandas as pd

    mock_result.to_pandas.return_value = pd.DataFrame(
        {
            'faithfulness': [0.85],
            'answer_relevancy': [0.90],
            'context_precision': [0.80],
            'context_recall': [0.75],
        }
    )

    with (
        patch('evaluation.ragas_eval.evaluate', return_value=mock_result),
        patch('evaluation.ragas_eval._build_ragas_llm'),
        patch('evaluation.ragas_eval._build_ragas_embeddings'),
        patch('evaluation.ragas_eval.mlflow') as mock_mlflow,
    ):
        mock_run = MagicMock()
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        from evaluation.ragas_eval import evaluate_rag_pipeline

        evaluate_rag_pipeline(str(golden_set_file), _mock_rag_fn)

    mock_mlflow.log_metrics.assert_called_once()


# --- LLM-as-Judge ---


def test_llm_judge_returns_three_criteria():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value.content = '{"relevance": 4, "faithfulness": 5, "financial_utility": 4}'

    with patch('evaluation.llm_judge._build_llm', return_value=mock_llm):
        from evaluation.llm_judge import judge_answer

        result = judge_answer(
            query='What is the AAPL prediction?',
            answer='The model predicts +$0.14 in 5 days.',
            contexts=['LSTM predicts D+5 delta.'],
        )

    assert 'relevance' in result
    assert 'faithfulness' in result
    assert 'financial_utility' in result
    assert all(1 <= v <= 5 for v in result.values())


def test_llm_judge_clamps_scores():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value.content = '{"relevance": 10, "faithfulness": 0, "financial_utility": 3}'

    with patch('evaluation.llm_judge._build_llm', return_value=mock_llm):
        from evaluation.llm_judge import judge_answer

        result = judge_answer(query='q', answer='a', contexts=['c'])

    assert result['relevance'] == 5
    assert result['faithfulness'] == 1
    assert result['financial_utility'] == 3


def test_llm_judge_invalid_json_fallback():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value.content = 'not json at all'

    with patch('evaluation.llm_judge._build_llm', return_value=mock_llm):
        from evaluation.llm_judge import judge_answer

        result = judge_answer(query='q', answer='a', contexts=['c'])

    assert result == {'relevance': 3, 'faithfulness': 3, 'financial_utility': 3}


def test_llm_judge_batch_returns_averages(tmp_path: Path):
    mock_llm = MagicMock()
    mock_llm.invoke.return_value.content = '{"relevance": 4, "faithfulness": 5, "financial_utility": 3}'

    golden_data = [
        {'query': 'q1', 'expected_answer': 'a1', 'contexts': ['c1']},
        {'query': 'q2', 'expected_answer': 'a2', 'contexts': ['c2']},
    ]
    gf = tmp_path / 'gs.json'
    gf.write_text(json.dumps(golden_data))

    with (
        patch('evaluation.llm_judge._build_llm', return_value=mock_llm),
        patch('evaluation.llm_judge.mlflow'),
    ):
        from evaluation.llm_judge import evaluate_with_judge

        result = evaluate_with_judge(str(gf), _mock_rag_fn)

    assert 'avg_relevance' in result
    assert 'avg_faithfulness' in result
    assert 'avg_financial_utility' in result
    assert 'avg_overall' in result
    assert result['avg_relevance'] == 4.0
    assert result['avg_faithfulness'] == 5.0
    assert result['avg_financial_utility'] == 3.0
