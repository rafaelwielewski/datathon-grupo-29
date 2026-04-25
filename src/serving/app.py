from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Callable

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from pydantic import BaseModel, Field
from starlette.requests import Request
from starlette.responses import Response

from src.serving.context import set_request_id
from src.serving.logging_config import configure_logging

logger = logging.getLogger(__name__)


class LoggedRoute(APIRoute):
    def get_route_handler(self) -> Callable:
        original = super().get_route_handler()

        async def handler(request: Request) -> Response:
            set_request_id(request.headers.get('x-request-id', uuid.uuid4().hex[:8]))
            start = time.perf_counter()
            try:
                try:
                    body = await request.json()
                    logger.info('%s %s | req: %s', request.method, request.url.path, body)
                except Exception:
                    logger.info('%s %s', request.method, request.url.path)

                response: Response = await original(request)
                elapsed = (time.perf_counter() - start) * 1000

                try:
                    resp_data = json.loads(bytes(response.body).decode())
                    resp_log = resp_data.get('answer', resp_data) if isinstance(resp_data, dict) else resp_data
                except Exception:
                    resp_log = bytes(response.body).decode(errors='replace')

                logger.info('%s %s %d (%.0fms) | resp: %s', request.method, request.url.path, response.status_code, elapsed, resp_log)
            except BaseException as exc:
                elapsed = (time.perf_counter() - start) * 1000
                logger.error('%s %s (%.0fms) | error: %s', request.method, request.url.path, elapsed, exc, exc_info=True)
                raise

            return response

        return handler


@asynccontextmanager
async def lifespan(_app: FastAPI):
    from dotenv import load_dotenv
    load_dotenv()
    configure_logging()
    logger.info('Application starting up')
    yield
    logger.info('Application shutting down')


app = FastAPI(
    title='AAPL Financial Assistant',
    description='Agente ReAct para análise de ações AAPL com previsão LSTM D+5.',
    version='1.0.0',
    lifespan=lifespan,
)
app.router.route_class = LoggedRoute


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.error('Unhandled error: %s', exc, exc_info=True)
    return JSONResponse(status_code=500, content={'detail': 'Internal server error'})


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
