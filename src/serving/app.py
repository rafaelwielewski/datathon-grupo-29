from __future__ import annotations

import logging
from functools import lru_cache

from fastapi import FastAPI
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

app = FastAPI(
    title='AAPL Financial Assistant',
    description='Agente ReAct para análise de ações AAPL com previsão LSTM D+5.',
    version='1.0.0',
)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)


class QueryResponse(BaseModel):
    answer: str
    intermediate_steps: list | None = None


@lru_cache(maxsize=1)
def _get_agent():
    """Inicializa o agente uma única vez (singleton por processo)."""
    from src.agent.react_agent import build_agent_executor
    return build_agent_executor()


@app.get('/health')
def health() -> dict:
    """Verifica se a API está operacional."""
    return {'status': 'ok', 'model': 'LSTM ONNX D+5', 'agent': 'ReAct + Ollama'}


@app.post('/query', response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    """Processa uma pergunta sobre AAPL usando o agente ReAct."""
    from src.agent.react_agent import invoke_agent
    agent = _get_agent()
    result = invoke_agent(agent, request.question)
    return QueryResponse(
        answer=result['output'],
        intermediate_steps=result.get('intermediate_steps'),
    )
