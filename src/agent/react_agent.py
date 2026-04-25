from __future__ import annotations

import logging
from pathlib import Path

import yaml
from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

CONFIG_PATH = Path('configs/agent_config.yaml')

SYSTEM_PROMPT = """You are a financial assistant specialized in Apple Inc. (AAPL) stock analysis.
Use the available tools to answer accurately. Always cite the data returned by the tools.
Be concise and professional. When asked about predictions or risk, always use the tools first."""


def _load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, encoding='utf-8') as f:
        return yaml.safe_load(f)


def build_llm(cfg: dict | None = None) -> BaseChatModel:
    """Constrói o LLM a partir do config (GitHub Models ou OpenAI)."""
    if cfg is None:
        cfg = _load_config()
    llm_cfg = cfg['llm']
    return ChatOpenAI(
        model=llm_cfg['model'],
        base_url=llm_cfg.get('base_url'),
        api_key=_resolve_api_key(llm_cfg['provider']),
        temperature=llm_cfg['temperature'],
        max_tokens=llm_cfg['max_tokens'],
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
    """Constrói agente LangGraph ReAct com as tools do domínio financeiro."""
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


def invoke_agent(agent, question: str) -> dict:
    """Invoca o agente e retorna output normalizado."""
    try:
        result = agent.invoke({'messages': [{'role': 'user', 'content': question}]})
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
    return {'output': answer, 'intermediate_steps': steps}
