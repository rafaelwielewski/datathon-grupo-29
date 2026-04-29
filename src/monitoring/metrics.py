from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

REQUEST_COUNT = Counter(
    'api_requests_total',
    'Total de requisições',
    ['method', 'endpoint', 'status_code'],
)

REQUEST_LATENCY = Histogram(
    'api_request_duration_seconds',
    'Latência',
    ['method', 'endpoint'],
)

ACTIVE_REQUESTS = Gauge(
    'api_active_requests',
    'Requisições ativas',
)

# Métrica de Negócio: Monitora o viés das previsões do modelo em produção
MODEL_PREDICTION_DIRECTION = Counter(
    'model_prediction_direction_total',
    'Total de direções previstas pelo modelo',
    ['direction'],  # UP ou DOWN
)

# Métrica de Drift: Monitora a parcela de features com drift
DRIFT_SHARE = Gauge(
    'model_drift_share',
    'Parcela de features com drift detectado (0 a 1)',
)
