from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Callable

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from prometheus_client import make_asgi_app
from pydantic import BaseModel, Field
from starlette.requests import Request
from starlette.responses import Response

from src.monitoring.drift import detect_and_log_drift
from src.monitoring.metrics import (
    ACTIVE_REQUESTS,
    FLIGHT_PREDICTION,
    REQUEST_COUNT,
    REQUEST_LATENCY,
)
from src.security.guardrails import InputGuardrail, OutputGuardrail
from src.serving.context import set_request_id
from src.serving.logging_config import configure_logging

logger = logging.getLogger(__name__)


class LoggedRoute(APIRoute):
    def get_route_handler(self) -> Callable:
        original = super().get_route_handler()

        async def handler(request: Request) -> Response:
            set_request_id(request.headers.get('x-request-id', uuid.uuid4().hex[:8]))
            start = time.perf_counter()
            ACTIVE_REQUESTS.inc()
            try:
                logger.info('%s %s | req_body=omitted', request.method, request.url.path)

                response = await original(request)
                elapsed = (time.perf_counter() - start) * 1000

                resp_len = len(response.body or b'')
                logger.info(
                    '%s %s %d (%.0fms) | resp_bytes=%d',
                    request.method,
                    request.url.path,
                    response.status_code,
                    elapsed,
                    resp_len,
                )
                REQUEST_COUNT.labels(method=request.method, endpoint=request.url.path, status_code=str(response.status_code)).inc()
                REQUEST_LATENCY.labels(method=request.method, endpoint=request.url.path).observe(elapsed / 1000)
            except BaseException as exc:
                elapsed = (time.perf_counter() - start) * 1000
                logger.error('%s %s (%.0fms) | error: %s', request.method, request.url.path, elapsed, exc, exc_info=True)
                REQUEST_COUNT.labels(method=request.method, endpoint=request.url.path, status_code='500').inc()
                raise
            finally:
                ACTIVE_REQUESTS.dec()

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
    title='Flight Delay Prediction Assistant',
    description='Agente ReAct para previsão de atrasos de voos com CatBoost + Platt calibration.',
    version='1.0.0',
    lifespan=lifespan,
)
app.router.route_class = LoggedRoute

_metrics_app = make_asgi_app()
app.mount('/metrics', _metrics_app)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.error('Unhandled error: %s', exc, exc_info=True)
    return JSONResponse(status_code=500, content={'detail': 'Internal server error'})


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)


class QueryResponse(BaseModel):
    answer: str
    intermediate_steps: list | None = None


_INPUT_GUARD = InputGuardrail(max_length=1000)
_OUTPUT_GUARD = OutputGuardrail()


@app.get('/health')
def health() -> dict:
    return {'status': 'ok', 'model': 'CatBoost + Platt', 'agent': 'ReAct + gpt-4.1'}


@app.get('/drift')
def drift_report() -> dict:
    return detect_and_log_drift()


@app.post('/query')
def query(request: QueryRequest) -> QueryResponse:
    """Processa uma pergunta sobre atrasos de voo usando o agente ReAct."""
    from src.agent.react_agent import invoke_agent

    validation = _INPUT_GUARD.validate(request.question)
    if not validation.is_valid:
        raise HTTPException(status_code=400, detail=validation.reason)

    agent = _get_agent()
    result = invoke_agent(agent, request.question)
    safe_output = _OUTPUT_GUARD.sanitize(result['output'])
    raw_steps = result.get('intermediate_steps') or []
    safe_steps = [{'role': s['role'], 'content': _OUTPUT_GUARD.sanitize(s['content'])} for s in raw_steps]
    delayed_label = 'delayed' if 'delayed: true' in safe_output.lower() or 'atrasado' in safe_output.lower() else 'on_time'
    FLIGHT_PREDICTION.labels(prediction=delayed_label).inc()
    return QueryResponse(answer=safe_output, intermediate_steps=safe_steps)


@app.get('/predict-from-store/{flight_id}')
def predict_from_store(flight_id: int) -> dict:
    from src.models.predictor import predict_from_feature_store

    try:
        result = predict_from_feature_store(flight_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    delayed_label = 'delayed' if result.delayed else 'on_time'
    FLIGHT_PREDICTION.labels(prediction=delayed_label).inc()
    return {
        'flight_id': flight_id,
        'delayed_probability': result.delayed_probability,
        'delayed': result.delayed,
        'threshold': result.threshold,
    }


@lru_cache(maxsize=1)
def _get_agent():
    from src.agent.react_agent import build_agent_executor

    return build_agent_executor()
