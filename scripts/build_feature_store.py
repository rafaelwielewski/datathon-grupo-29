from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from feast import FeatureStore

from src.features.feature_engineering import build_flight_features

logger = logging.getLogger(__name__)

DATA_DIR = Path('data')
RAW_DIR = DATA_DIR / 'raw'
OUT_DIR = DATA_DIR / 'processed' / 'feature_store'
OUT_PATH = OUT_DIR / 'flight_features.parquet'
REGISTRY_DIR = DATA_DIR / 'feature_store'


def _load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    flights = pd.read_csv(RAW_DIR / 'flights.csv', low_memory=False)
    airlines = pd.read_csv(RAW_DIR / 'airlines.csv')
    airports = pd.read_csv(RAW_DIR / 'airports.csv')
    return flights, airlines, airports


def _build_feature_rows(df: pd.DataFrame) -> pd.DataFrame:
    if 'flight_date' not in df.columns:
        raise ValueError('flight_date missing from engineered features')

    df = df.copy()
    df['event_timestamp'] = pd.to_datetime(df['flight_date'], errors='coerce')
    df = df[~df['event_timestamp'].isna()].copy()
    df = df.sort_values('event_timestamp').reset_index(drop=True)
    df['flight_id'] = df.index.astype('int64')

    keep_cols = [
        'flight_id',
        'event_timestamp',
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
        'is_weekend',
        'distance_bucket',
        'AIRLINE',
        'ORIGIN_AIRPORT',
        'DESTINATION_AIRPORT',
        'ROUTE',
    ]
    return df[keep_cols]


def build_feature_store() -> Path:
    flights, airlines, airports = _load_raw_data()
    logger.info('Building engineered features for feature store...')
    features = build_flight_features(flights, airlines, airports, use_ops=False)
    store_df = _build_feature_rows(features)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    store_df.to_parquet(OUT_PATH, index=False)
    logger.info('Feature store parquet written: %s', OUT_PATH)

    fs = FeatureStore(repo_path='feature_store')
    from feature_store.feature_store import flight_entity, flight_features_view

    fs.apply([flight_entity, flight_features_view])
    min_ts = store_df['event_timestamp'].min()
    max_ts = store_df['event_timestamp'].max()
    if min_ts is not None and max_ts is not None:
        fs.materialize(min_ts, max_ts)
        logger.info('Feature store materialized: %s -> %s', min_ts, max_ts)
    logger.info('Feature store registry updated')

    return OUT_PATH


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    build_feature_store()
