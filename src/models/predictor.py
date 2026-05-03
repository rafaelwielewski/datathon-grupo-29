from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path('data/processed/artifacts')
TRAIN_PRIOR = 0.18


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


def build_feature_row(params: FlightParams) -> pd.DataFrame:
    dep_hour = int(params.scheduled_departure / 100)
    arr_hour = int(params.scheduled_arrival / 100)

    row: dict = {
        'YEAR': params.year,
        'MONTH': params.month,
        'DAY': params.day,
        'DAY_OF_WEEK': params.day_of_week,
        'AIRLINE': params.airline.upper(),
        'FLIGHT_NUMBER': 999,
        'TAIL_NUMBER': 'N999AA',
        'ORIGIN_AIRPORT': params.origin.upper(),
        'ORIGIN_STATE': 'GA',
        'DESTINATION_AIRPORT': params.destination.upper(),
        'DEST_STATE': 'CA',
        'ROUTE': f'{params.origin.upper()}_{params.destination.upper()}',
        'DISTANCE': params.distance,
        'SCHEDULED_TIME': params.scheduled_time,
        'sched_dep_hour': dep_hour,
        'sched_dep_minute': int(params.scheduled_departure % 100),
        'sched_dep_period': _period(dep_hour),
        'sched_arr_hour': arr_hour,
        'sched_arr_minute': int(params.scheduled_arrival % 100),
        'sched_arr_period': _period(arr_hour),
        'is_weekend': int(params.day_of_week in [6, 7]),
        'distance_bucket': 'long',
        'day_of_year': 180,
        'doy_sin': 0.0,
        'doy_cos': -1.0,
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
