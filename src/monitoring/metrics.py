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

FLIGHT_PREDICTION = Counter(
    'flight_prediction_total',
    'Total de predições de atraso de voos',
    ['prediction'],
)

# Métrica de Drift: Monitora a parcela de features com drift
DRIFT_SHARE = Gauge(
    'model_drift_share',
    'Parcela de features com drift detectado (0 a 1)',
)
