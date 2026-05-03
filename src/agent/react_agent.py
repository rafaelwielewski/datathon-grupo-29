from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import yaml
from langchain.agents import create_agent
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.outputs import LLMResult
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

CONFIG_PATH = Path('configs/agent_config.yaml')

SYSTEM_PROMPT = """You are an aviation assistant specialized in US domestic flight delay prediction.

Available tools:
- predict_flight_delay: predicts delay probability for a specific flight (airline, route, date, time).
- get_airport_delay_stats: returns historical delay statistics for an airport (IATA code).
- get_airline_delay_stats: returns historical delay statistics for an airline (IATA code).

How to answer common questions:
- Route delay (e.g. ATL-LAX): call get_airport_delay_stats for both origin AND destination airports.
- Best time to fly a route: call predict_flight_delay multiple times with different scheduled_departure
  values (e.g. 600, 900, 1200, 1500, 1800, 2100) and compare the delayed_probability results.
- Airline comparison: call get_airline_delay_stats for each airline.
- Specific flight prediction: call predict_flight_delay with the given parameters.

Rules:
- Always use tools before answering — never guess statistics.
- Always cite the exact values returned by tools.
- Stop reasoning as soon as you have enough data to answer.
- Be concise and professional."""


class _LLMLogger(BaseCallbackHandler):
    """Logs every message sent to and received from the LLM."""

    def on_chat_model_start(self, serialized: dict, messages: list, **_: Any) -> None:
        for batch in messages:
            for msg in batch:
                content = msg.content if isinstance(msg.content, str) else json.dumps(msg.content, ensure_ascii=False)
                logger.info('LLM ← [%s] %s', getattr(msg, 'type', 'msg'), content)

    def on_llm_end(self, response: LLMResult, **_: Any) -> None:
        for batch in response.generations:
            for gen in batch:
                # text is empty when model responds with tool calls only
                text = gen.text if hasattr(gen, 'text') else str(gen)
                if text:
                    logger.info('LLM → %s', text)
                # log tool calls from additional_kwargs
                tool_calls = getattr(gen, 'message', None)
                if tool_calls is not None:
                    for tc in getattr(tool_calls, 'tool_calls', []):
                        logger.info('LLM → tool_call: %s(%s)', tc.get('name'), tc.get('args'))

    def on_llm_error(self, error: BaseException, **_: Any) -> None:
        logger.error('LLM error: %s', error, exc_info=True)

    def on_tool_start(self, serialized: dict, input_str: str, **_: Any) -> None:
        logger.info('TOOL → %s | input: %s', serialized.get('name', '?'), input_str)

    def on_tool_end(self, output: Any, **_: Any) -> None:
        content = getattr(output, 'content', output)
        logger.info('TOOL ← %s', content)


def _load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, encoding='utf-8') as f:
        return yaml.safe_load(f)


def build_llm(cfg: dict | None = None) -> BaseChatModel:
    from pydantic import SecretStr

    if cfg is None:
        cfg = _load_config()
    llm_cfg = cfg['llm']
    return ChatOpenAI(
        model=llm_cfg['model'],
        base_url=llm_cfg.get('base_url'),
        api_key=SecretStr(_resolve_api_key(llm_cfg['provider'])),
        temperature=llm_cfg['temperature'],
        max_completion_tokens=llm_cfg['max_tokens'],
        callbacks=[],
    )


def _resolve_api_key(provider: str) -> str:
    import os

    if provider == 'github':
        return os.environ['GITHUB_TOKEN']
    return os.environ['OPENAI_API_KEY']


def build_agent_executor(
    llm: BaseChatModel | None = None,
    tools: list | None = None,
    cfg: dict | None = None,
):
    """Constrói agente LangGraph ReAct com as tools do domínio de aviação."""
    if cfg is None:
        cfg = _load_config()
    if llm is None:
        llm = build_llm(cfg)
    if tools is None:
        from src.agent.tools import get_all_tools

        tools = get_all_tools()

    if len(tools) < 3:
        logger.warning('Datathon exige >= 3 tools. Fornecidas: %d', len(tools))

    logger.info('Agent built with %d tools: %s', len(tools), [t.name for t in tools])
    return create_agent(model=llm, tools=tools, system_prompt=SYSTEM_PROMPT)


_CALLBACKS = [_LLMLogger()]


def invoke_agent(agent, question: str) -> dict:
    """Invoca o agente e retorna output normalizado."""
    from datetime import date

    today = date.today().isoformat()
    content = f"Today's date: {today}.\n\n{question}"
    logger.info('Agent question: %s', question)
    try:
        result = agent.invoke(
            {'messages': [{'role': 'user', 'content': content}]},
            config={'callbacks': _CALLBACKS},
        )
    except Exception:
        logger.exception('Agent invocation failed for question: %s', question)
        raise
    messages = result.get('messages', [])
    answer = messages[-1].content if messages else ''
    steps = [
        {'role': str(m.type), 'content': str(m.content)}
        for m in messages[1:]  # skip human message
    ]
    logger.debug('Agent used %d intermediate steps', len(steps))
    logger.info('Agent answer: %s', answer)
    return {'output': answer, 'intermediate_steps': steps}
