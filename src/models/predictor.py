from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path('data/processed/artifacts')
TRAIN_PRIOR = 0.18
MISSING_TOKEN = 'MISSING'  # nosec B105

_airport_state_cache: dict[str, str] | None = None
_route_stats_cache: dict[str, dict] | None = None


def _get_airport_state(iata_code: str) -> str:
    global _airport_state_cache
    if _airport_state_cache is None:
        map_path = ARTIFACTS_DIR / 'airport_state_map.json'
        _airport_state_cache = json.loads(map_path.read_text()) if map_path.exists() else {}
    return _airport_state_cache.get(iata_code.upper(), MISSING_TOKEN)


def _get_route_stats(origin: str, destination: str) -> dict:
    global _route_stats_cache
    if _route_stats_cache is None:
        stats_path = ARTIFACTS_DIR / 'route_stats.json'
        _route_stats_cache = json.loads(stats_path.read_text()) if stats_path.exists() else {}
    return _route_stats_cache.get(f'{origin.upper()}_{destination.upper()}', {})


@dataclass
class FlightParams:
    airline: str = 'AA'
    origin: str = 'ATL'
    destination: str = 'LAX'
    month: int = 6
    day: int = 15
    day_of_week: int = 3
    scheduled_departure: float = 800
    scheduled_arrival: float = 1100
    distance: float = 1000
    scheduled_time: float = 150
    year: int = 2015


@dataclass
class PredictionResult:
    delayed_probability: float
    delayed: bool
    threshold: float


_FEATURE_STORE_FEATURES = [
    'flight_features:YEAR',
    'flight_features:MONTH',
    'flight_features:DAY',
    'flight_features:DAY_OF_WEEK',
    'flight_features:sched_dep_hour',
    'flight_features:sched_dep_minute',
    'flight_features:sched_arr_hour',
    'flight_features:sched_arr_minute',
    'flight_features:DISTANCE',
    'flight_features:SCHEDULED_TIME',
    'flight_features:is_weekend',
    'flight_features:distance_bucket',
    'flight_features:AIRLINE',
    'flight_features:ORIGIN_AIRPORT',
    'flight_features:DESTINATION_AIRPORT',
    'flight_features:ROUTE',
]


def _normalize_feature_key(key: str) -> str:
    if ':' in key:
        return key.split(':')[-1]
    if '__' in key:
        return key.split('__')[-1]
    return key


def _flatten_feature_store_result(result: dict) -> dict:
    flattened: dict[str, object] = {}
    for key, values in result.items():
        if key == 'flight_id':
            continue
        name = _normalize_feature_key(key)
        if isinstance(values, list):
            flattened[name] = values[0] if values else None
        else:
            flattened[name] = values
    return flattened


def _params_from_feature_store(features: dict) -> FlightParams:
    required = [
        'YEAR',
        'MONTH',
        'DAY',
        'DAY_OF_WEEK',
        'sched_dep_hour',
        'sched_dep_minute',
        'sched_arr_hour',
        'sched_arr_minute',
        'DISTANCE',
        'SCHEDULED_TIME',
        'AIRLINE',
        'ORIGIN_AIRPORT',
        'DESTINATION_AIRPORT',
    ]
    missing = [k for k in required if features.get(k) is None]
    if missing:
        raise ValueError(f'Missing required features from store: {missing}')

    dep = int(features['sched_dep_hour']) * 100 + int(features['sched_dep_minute'])
    arr = int(features['sched_arr_hour']) * 100 + int(features['sched_arr_minute'])

    return FlightParams(
        airline=str(features['AIRLINE']),
        origin=str(features['ORIGIN_AIRPORT']),
        destination=str(features['DESTINATION_AIRPORT']),
        month=int(features['MONTH']),
        day=int(features['DAY']),
        day_of_week=int(features['DAY_OF_WEEK']),
        scheduled_departure=dep,
        scheduled_arrival=arr,
        distance=float(features['DISTANCE']),
        scheduled_time=float(features['SCHEDULED_TIME']),
        year=int(features['YEAR']),
    )


def _period(hour: int) -> str:
    if 5 <= hour <= 11:
        return 'morning'
    if 12 <= hour <= 17:
        return 'afternoon'
    if 18 <= hour <= 22:
        return 'evening'
    return 'night'


# Exact feature order the model was trained on (from model.feature_names_)
_MODEL_FEATURES = [
    'YEAR',
    'MONTH',
    'DAY',
    'DAY_OF_WEEK',
    'AIRLINE',
    'FLIGHT_NUMBER',
    'TAIL_NUMBER',
    'ORIGIN_AIRPORT',
    'ORIGIN_STATE',
    'DESTINATION_AIRPORT',
    'DEST_STATE',
    'ROUTE',
    'DISTANCE',
    'SCHEDULED_TIME',
    'sched_dep_hour',
    'sched_dep_minute',
    'sched_dep_period',
    'sched_arr_hour',
    'sched_arr_minute',
    'sched_arr_period',
    'is_weekend',
    'distance_bucket',
    'day_of_year',
    'doy_sin',
    'doy_cos',
    'origin_day_hour_logvol',
    'dest_day_hour_logvol',
    'te_airline_w7',
    'te_airline_w30',
    'te_origin_w7',
    'te_origin_w30',
    'te_dest_w7',
    'te_dest_w30',
    'te_route_w7',
    'te_route_w30',
    'te_origin_hour_w7',
    'te_origin_hour_w30',
    'te_origin_dow_w30',
    'tail_dep_delay_mean_w3',
    'tail_dep_delay_mean_w5',
    'tail_dep_delay_mean_w10',
    'tail_delayed_rate_w5',
    'tail_delayed_rate_w10',
    'origin_dep_delay_mean_w30',
    'origin_dep_delay_mean_w90',
    'origin_weather_rate_w90',
    'origin_weather_rate_w180',
    'origin_system_rate_w90',
    'origin_system_rate_w180',
    'origin_late_aircraft_rate_w90',
    'origin_late_aircraft_rate_w180',
    'origin_hour_dep_delay_mean_w30',
    'route_dep_delay_mean_w30',
    'route_delayed_rate_w30',
]

# Cat feature indices: [4,6,7,8,9,10,11,16,19,21] → names for readability
_CAT_FEATURES = [
    'AIRLINE',
    'TAIL_NUMBER',
    'ORIGIN_AIRPORT',
    'ORIGIN_STATE',
    'DESTINATION_AIRPORT',
    'DEST_STATE',
    'ROUTE',
    'sched_dep_period',
    'sched_arr_period',
    'distance_bucket',
]


def _compute_day_of_year(year: int, month: int, day: int) -> int:
    try:
        return (date(year, month, day) - date(year, 1, 1)).days + 1
    except ValueError:
        return 180


def _compute_distance_bucket(distance: float) -> str:
    if distance < 500:
        return 'short'
    if distance < 1500:
        return 'medium'
    return 'long'


def _estimate_arrival(dep: float, scheduled_time_min: float) -> float:
    dep_h, dep_m = int(dep / 100), int(dep % 100)
    total_min = dep_h * 60 + dep_m + scheduled_time_min
    arr_h, arr_m = int(total_min / 60) % 24, int(total_min % 60)
    return arr_h * 100 + arr_m


def build_feature_row(params: FlightParams) -> pd.DataFrame:
    route = _get_route_stats(params.origin, params.destination)

    distance = float(params.distance) if params.distance is not None else float(route.get('distance', 1000))
    sched_time = float(params.scheduled_time) if params.scheduled_time is not None else float(route.get('scheduled_time', 150))
    dep = float(params.scheduled_departure) if params.scheduled_departure is not None else 800.0
    arr = float(params.scheduled_arrival) if params.scheduled_arrival is not None else _estimate_arrival(dep, sched_time)

    dep_hour = int(dep / 100)
    arr_hour = int(arr / 100)

    doy = _compute_day_of_year(params.year, params.month, params.day)
    two_pi = 2.0 * math.pi

    row: dict = {
        'YEAR': params.year,
        'MONTH': params.month,
        'DAY': params.day,
        'DAY_OF_WEEK': params.day_of_week,
        'AIRLINE': params.airline.upper(),
        'FLIGHT_NUMBER': 999,
        'TAIL_NUMBER': 'N999AA',
        'ORIGIN_AIRPORT': params.origin.upper(),
        'ORIGIN_STATE': _get_airport_state(params.origin),
        'DESTINATION_AIRPORT': params.destination.upper(),
        'DEST_STATE': _get_airport_state(params.destination),
        'ROUTE': f'{params.origin.upper()}_{params.destination.upper()}',
        'DISTANCE': distance,
        'SCHEDULED_TIME': sched_time,
        'sched_dep_hour': dep_hour,
        'sched_dep_minute': int(dep % 100),
        'sched_dep_period': _period(dep_hour),
        'sched_arr_hour': arr_hour,
        'sched_arr_minute': int(arr % 100),
        'sched_arr_period': _period(arr_hour),
        'is_weekend': int(params.day_of_week in [6, 7]),
        'distance_bucket': _compute_distance_bucket(distance),
        'day_of_year': doy,
        'doy_sin': math.sin(two_pi * doy / 365.0),
        'doy_cos': math.cos(two_pi * doy / 365.0),
        'origin_day_hour_logvol': TRAIN_PRIOR,
        'dest_day_hour_logvol': TRAIN_PRIOR,
    }

    for col in [
        'te_airline_w7',
        'te_airline_w30',
        'te_origin_w7',
        'te_origin_w30',
        'te_dest_w7',
        'te_dest_w30',
        'te_route_w7',
        'te_route_w30',
        'te_origin_hour_w7',
        'te_origin_hour_w30',
        'te_origin_dow_w30',
        'tail_dep_delay_mean_w3',
        'tail_dep_delay_mean_w5',
        'tail_dep_delay_mean_w10',
        'tail_delayed_rate_w5',
        'tail_delayed_rate_w10',
        'origin_dep_delay_mean_w30',
        'origin_dep_delay_mean_w90',
        'origin_weather_rate_w90',
        'origin_weather_rate_w180',
        'origin_system_rate_w90',
        'origin_system_rate_w180',
        'origin_late_aircraft_rate_w90',
        'origin_late_aircraft_rate_w180',
        'origin_hour_dep_delay_mean_w30',
        'route_dep_delay_mean_w30',
        'route_delayed_rate_w30',
    ]:
        row[col] = TRAIN_PRIOR

    return pd.DataFrame([row])[_MODEL_FEATURES]


def run_prediction(feature_df: pd.DataFrame) -> tuple[float, float]:
    """Returns (probability, threshold). Requires catboost installed."""
    import joblib
    from catboost import CatBoostClassifier, Pool

    model_path = ARTIFACTS_DIR / 'catboost_model.cbm'
    calibrator_path = ARTIFACTS_DIR / 'platt_calibrator.joblib'
    threshold_path = ARTIFACTS_DIR / 'best_threshold.txt'

    if not model_path.exists():
        logger.warning('Model not found at %s, returning prior', model_path)
        return TRAIN_PRIOR, 0.5

    threshold = float(threshold_path.read_text().strip()) if threshold_path.exists() else 0.5

    cat_idx = [feature_df.columns.get_loc(c) for c in _CAT_FEATURES if c in feature_df.columns]

    model = CatBoostClassifier()
    model.load_model(str(model_path))
    pool = Pool(feature_df, cat_features=cat_idx)
    proba = float(model.predict_proba(pool)[:, 1][0])

    if calibrator_path.exists():
        calibrator = joblib.load(calibrator_path)
        proba = float(calibrator.predict_proba([[proba]])[:, 1][0])

    return proba, threshold


_feast_store: Any = None


def _get_feast_store() -> Any:
    global _feast_store
    if _feast_store is None:
        from feast import FeatureStore

        _feast_store = FeatureStore(repo_path='feature_store')
    return _feast_store


def predict_from_feature_store(flight_id: int) -> PredictionResult:
    """Predict using features loaded from the Feast online store."""
    fs = _get_feast_store()
    store_result = fs.get_online_features(
        features=_FEATURE_STORE_FEATURES,
        entity_rows=[{'flight_id': int(flight_id)}],
    ).to_dict()
    features = _flatten_feature_store_result(store_result)
    params = _params_from_feature_store(features)
    logger.info('Predict from feature store | flight_id=%s', flight_id)
    return predict(params)


def predict(params: FlightParams) -> PredictionResult:
    """Top-level entry point: build features → run model → return result."""
    feature_df = build_feature_row(params)
    proba, threshold = run_prediction(feature_df)
    logger.info(
        'Predict | %s→%s airline=%s proba=%.4f threshold=%.4f delayed=%s',
        params.origin,
        params.destination,
        params.airline,
        proba,
        threshold,
        proba >= threshold,
    )
    return PredictionResult(delayed_probability=proba, delayed=proba >= threshold, threshold=threshold)
