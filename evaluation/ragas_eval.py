from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Callable

import mlflow
from ragas import EvaluationDataset, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics.collections import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

logger = logging.getLogger(__name__)

GOLDEN_SET_PATH = Path('data/golden_set/golden_set.json')


def _load_config() -> dict:
    import yaml

    with open('configs/agent_config.yaml', encoding='utf-8') as f:
        return yaml.safe_load(f)


def _build_ragas_llm() -> LangchainLLMWrapper:
    from langchain_openai import ChatOpenAI

    cfg = _load_config()['llm']
    llm = ChatOpenAI(
        model=cfg['model'],
        base_url=cfg.get('base_url'),
        api_key=os.environ.get('GITHUB_TOKEN') or os.environ.get('OPENAI_API_KEY', ''),
        temperature=0.0,
    )
    return LangchainLLMWrapper(llm)


def _build_ragas_embeddings() -> LangchainEmbeddingsWrapper:
    from langchain_huggingface import HuggingFaceEmbeddings

    embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
    return LangchainEmbeddingsWrapper(embeddings)


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

    with open(golden_set_path, encoding='utf-8') as f:
        golden_set = json.load(f)

    ragas_llm = _build_ragas_llm()
    ragas_embeddings = _build_ragas_embeddings()

    rows = []
    for item in golden_set:
        answer, contexts = rag_fn(item['query'])
        rows.append(
            {
                'user_input': item['query'],
                'response': answer,
                'retrieved_contexts': contexts,
                'reference': item['expected_answer'],
            }
        )

    dataset = EvaluationDataset.from_list(rows)
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        raise_exceptions=False,
    )

    df = result.to_pandas()
    metrics: dict[str, float] = {
        'faithfulness': float(df['faithfulness'].mean()),
        'answer_relevancy': float(df['answer_relevancy'].mean()),
        'context_precision': float(df['context_precision'].mean()),
        'context_recall': float(df['context_recall'].mean()),
    }

    logger.info('RAGAS scores: %s', metrics)

    mlflow.set_experiment('datathon-grupo-29-evaluation')
    with mlflow.start_run(run_name='ragas_evaluation'):
        mlflow.set_tag('model_type', 'rag')
        mlflow.set_tag('framework', 'ragas')
        mlflow.set_tag('owner', 'grupo-29')
        mlflow.set_tag('phase', 'datathon-fase05')
        mlflow.log_param('golden_set_size', len(golden_set))
        mlflow.log_metrics(metrics)

    return metrics


def _default_rag_fn() -> Callable[[str], tuple[str, list[str]]]:
    """Adapta o agente real para a interface (answer, contexts) esperada pelo RAGAS."""
    import requests

    def rag_fn(query: str) -> tuple[str, list[str]]:
        resp = requests.post('http://localhost:8000/query', json={'question': query}, timeout=60)  # noqa: S113
        data = resp.json()
        answer = data.get('answer', '')
        contexts = [
            step['content']
            for step in (data.get('intermediate_steps') or [])
            if step.get('role') == 'tool'
        ]
        return answer, contexts

    return rag_fn


if __name__ == '__main__':
    import logging

    logging.basicConfig(level=logging.INFO)
    results = evaluate_rag_pipeline()
    print(json.dumps(results, indent=2))
