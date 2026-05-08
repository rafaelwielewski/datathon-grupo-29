from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import mlflow

logger = logging.getLogger(__name__)

GOLDEN_SET_PATH = Path('data/golden_set/golden_set.json')


@dataclass(frozen=True)
class PromptVariant:
	name: str
	system_prompt: str


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
		max_tokens=512,
	)


def _generate_answer(llm, system_prompt: str, question: str) -> str:
	messages = [
		{'role': 'system', 'content': system_prompt},
		{'role': 'user', 'content': question},
	]
	response = llm.invoke(messages)
	return getattr(response, 'content', str(response))


def run_ab_test(
	prompt_a: PromptVariant,
	prompt_b: PromptVariant,
	golden_set_path: str = str(GOLDEN_SET_PATH),
	llm=None,
	judge_llm=None,
) -> dict[str, float]:
	"""Compare two prompts using LLM-as-judge on the golden set.

	Returns:
		Dict with average scores and win rates for A and B.
	"""
	from evaluation.llm_judge import judge_answer

	if llm is None:
		llm = _build_llm()

	with open(golden_set_path, encoding='utf-8') as f:
		golden_set = json.load(f)

	scores_a = []
	scores_b = []

	for item in golden_set:
		question = item['query']
		ans_a = _generate_answer(llm, prompt_a.system_prompt, question)
		ans_b = _generate_answer(llm, prompt_b.system_prompt, question)

		score_a = judge_answer(question, ans_a, contexts=[], llm=judge_llm)
		score_b = judge_answer(question, ans_b, contexts=[], llm=judge_llm)
		scores_a.append(score_a)
		scores_b.append(score_b)

	def _avg(scores: list[dict[str, int]]) -> dict[str, float]:
		n = max(len(scores), 1)
		return {
			'relevance': sum(s['relevance'] for s in scores) / n,
			'faithfulness': sum(s['faithfulness'] for s in scores) / n,
			'aviation_utility': sum(s['aviation_utility'] for s in scores) / n,
		}

	avg_a = _avg(scores_a)
	avg_b = _avg(scores_b)

	def _overall(avg: dict[str, float]) -> float:
		return (avg['relevance'] + avg['faithfulness'] + avg['aviation_utility']) / 3

	overall_a = _overall(avg_a)
	overall_b = _overall(avg_b)

	wins_a = sum(1 for a, b in zip(scores_a, scores_b) if _overall(a) > _overall(b))
	wins_b = sum(1 for a, b in zip(scores_a, scores_b) if _overall(b) > _overall(a))
	ties = max(len(scores_a) - wins_a - wins_b, 0)

	results = {
		'prompt_a_overall': overall_a,
		'prompt_b_overall': overall_b,
		'prompt_a_wins': float(wins_a),
		'prompt_b_wins': float(wins_b),
		'prompt_ties': float(ties),
		'prompt_a_relevance': avg_a['relevance'],
		'prompt_a_faithfulness': avg_a['faithfulness'],
		'prompt_a_aviation_utility': avg_a['aviation_utility'],
		'prompt_b_relevance': avg_b['relevance'],
		'prompt_b_faithfulness': avg_b['faithfulness'],
		'prompt_b_aviation_utility': avg_b['aviation_utility'],
	}

	logger.info('A/B prompt results: %s', results)

	mlflow.set_experiment('datathon-grupo-29-evaluation')
	with mlflow.start_run(run_name='ab_prompt_test'):
		mlflow.log_param('golden_set_size', len(scores_a))
		mlflow.log_param('prompt_a_name', prompt_a.name)
		mlflow.log_param('prompt_b_name', prompt_b.name)
		mlflow.log_metrics(results)

	return results


if __name__ == '__main__':
	logging.basicConfig(level=logging.INFO)

	prompt_a = PromptVariant(
		name='baseline',
		system_prompt='You are an aviation assistant. Answer clearly and concisely.',
	)
	prompt_b = PromptVariant(
		name='tool_focused',
		system_prompt='You are an aviation assistant. Use structured, step-by-step reasoning and cite key inputs.',
	)

	results = run_ab_test(prompt_a, prompt_b)
	print(json.dumps(results, indent=2))
