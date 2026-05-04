from __future__ import annotations

import json
import logging
import os
import time
from typing import Callable

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import mlflow

logger = logging.getLogger(__name__)

JUDGE_PROMPT = """You are an expert evaluator of AI aviation assistant responses.
Score the following response on three criteria from 1 (very poor) to 5 (excellent).

Question: {query}
Answer: {answer}
Retrieved contexts: {contexts}

Criteria:
1. relevance (1-5): Does the answer directly address the question?
2. faithfulness (1-5): Is the answer grounded in the retrieved contexts and tool data?
3. aviation_utility (1-5): Is this answer actionable and useful for aviation operational decisions?

Respond ONLY with a valid JSON object, no extra text:
{{"relevance": <int>, "faithfulness": <int>, "aviation_utility": <int>}}"""


def _load_config() -> dict:
    import yaml

    with open('configs/agent_config.yaml', encoding='utf-8') as f:
        return yaml.safe_load(f)


def _build_llm():
    from langchain_openai import ChatOpenAI

    cfg = _load_config()['llm']
    return ChatOpenAI(
        model=cfg['model'],
        base_url=cfg.get('base_url'),
        api_key=os.environ.get('GITHUB_TOKEN') or os.environ.get('OPENAI_API_KEY', ''),
        temperature=0.0,
        max_tokens=256,
    )


def judge_answer(
    query: str,
    answer: str,
    contexts: list[str],
    llm=None,
) -> dict[str, int]:
    """Avalia uma resposta com 3 critérios usando LLM-as-judge.

    Returns:
        Dict com relevance, faithfulness, aviation_utility (scores 1-5).
    """
    if llm is None:
        llm = _build_llm()

    prompt = JUDGE_PROMPT.format(query=query, answer=answer, contexts='\n'.join(contexts))
    response = llm.invoke(prompt)

    try:
        scores = json.loads(response.content)
    except (json.JSONDecodeError, AttributeError):
        logger.warning('LLM judge returned invalid JSON: %s', response)
        scores = {'relevance': 3, 'faithfulness': 3, 'aviation_utility': 3}

    return {k: max(1, min(5, int(v))) for k, v in scores.items()}


def evaluate_with_judge(
    golden_set_path: str,
    rag_fn: Callable[[str], tuple[str, list[str]]],
    llm=None,
) -> dict[str, float]:
    """Avalia o pipeline RAG com LLM-as-judge em todo o golden set.

    Returns:
        Dict com avg_relevance, avg_faithfulness, avg_aviation_utility, avg_overall.
    """
    if llm is None:
        llm = _build_llm()

    with open(golden_set_path, encoding='utf-8') as f:
        golden_set = json.load(f)

    all_scores: list[dict[str, int]] = []
    for item in golden_set:
        answer, contexts = rag_fn(item['query'])
        scores = judge_answer(item['query'], answer, contexts, llm=llm)
        all_scores.append(scores)
        logger.info('Query: %s | Scores: %s', item['query'][:60], scores)
        time.sleep(2)

    n = len(all_scores)
    result: dict[str, float] = {
        'avg_relevance': sum(s['relevance'] for s in all_scores) / n,
        'avg_faithfulness': sum(s['faithfulness'] for s in all_scores) / n,
        'avg_aviation_utility': sum(s['aviation_utility'] for s in all_scores) / n,
    }
    result['avg_overall'] = sum(result.values()) / 3

    logger.info('LLM-as-judge averages: %s', result)

    mlflow.set_experiment('datathon-grupo-29-evaluation')
    with mlflow.start_run(run_name='llm_judge_evaluation'):
        mlflow.set_tag('model_type', 'llm_judge')
        mlflow.set_tag('framework', 'langchain-openai')
        mlflow.set_tag('owner', 'grupo-29')
        mlflow.set_tag('phase', 'datathon-fase05')
        mlflow.log_param('golden_set_size', n)
        mlflow.log_param('judge_criteria', 'relevance,faithfulness,aviation_utility')
        mlflow.log_metrics(result)

    return result


if __name__ == '__main__':
    import logging

    logging.basicConfig(level=logging.INFO)

    def _local_agent_fn():
        from src.agent.react_agent import build_agent_executor, invoke_agent

        agent = build_agent_executor()

        def rag_fn(query: str) -> tuple[str, list[str]]:
            res = invoke_agent(agent, query)
            answer = res.get('output', '')
            contexts = [step['content'] for step in res.get('intermediate_steps', []) if step.get('role') == 'tool']
            return answer, contexts

        return rag_fn

    results = evaluate_with_judge('data/golden_set/golden_set.json', _local_agent_fn())
    print(json.dumps(results, indent=2))
