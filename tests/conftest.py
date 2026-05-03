from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def raw_flight_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 300
    months = np.repeat(np.arange(1, 13), 25)  # 25 flights per month
    days = rng.integers(1, 28, n)
    dow = rng.integers(1, 8, n)
    sched_dep = rng.choice([600, 800, 1000, 1200, 1400, 1600, 1800, 2000], n).astype(float)
    sched_arr = ((sched_dep + rng.integers(60, 300, n)) % 2400).astype(float)
    arrival_delay = rng.normal(5, 25, n).astype('float32')

    return pd.DataFrame(
        {
            'YEAR': pd.array(np.full(n, 2015, dtype=int), dtype='Int16'),
            'MONTH': pd.array(months, dtype='Int16'),
            'DAY': pd.array(days, dtype='Int16'),
            'DAY_OF_WEEK': pd.array(dow, dtype='Int16'),
            'AIRLINE': rng.choice(['AA', 'DL', 'UA', 'WN', 'AS'], n).tolist(),
            'FLIGHT_NUMBER': rng.integers(100, 9999, n).astype('float32'),
            'TAIL_NUMBER': rng.choice(['N123AA', 'N456DL', 'N789UA', 'N321WN', 'N654AS'], n).tolist(),
            'ORIGIN_AIRPORT': rng.choice(['ATL', 'LAX', 'ORD', 'DFW', 'JFK'], n).tolist(),
            'DESTINATION_AIRPORT': rng.choice(['ATL', 'LAX', 'ORD', 'DFW', 'JFK'], n).tolist(),
            'SCHEDULED_DEPARTURE': sched_dep,
            'SCHEDULED_ARRIVAL': sched_arr,
            'DEPARTURE_TIME': sched_dep,
            'DEPARTURE_DELAY': rng.normal(2, 15, n).astype('float32'),
            'ARRIVAL_DELAY': arrival_delay,
            'ARRIVAL_TIME': sched_arr,
            'SCHEDULED_TIME': rng.integers(60, 300, n).astype('float32'),
            'ELAPSED_TIME': rng.integers(60, 300, n).astype('float32'),
            'AIR_TIME': rng.integers(50, 280, n).astype('float32'),
            'DISTANCE': rng.integers(200, 2500, n).astype('float32'),
            'WHEELS_OFF': sched_dep,
            'WHEELS_ON': sched_arr,
            'TAXI_OUT': rng.normal(15, 5, n).astype('float32'),
            'TAXI_IN': rng.normal(8, 3, n).astype('float32'),
            'CANCELLED': pd.array(np.zeros(n, dtype=int), dtype='Int16'),
            'DIVERTED': pd.array(np.zeros(n, dtype=int), dtype='Int16'),
            'AIR_SYSTEM_DELAY': np.where(rng.random(n) < 0.15, rng.normal(10, 5, n), 0).astype('float32'),
            'SECURITY_DELAY': np.zeros(n, dtype='float32'),
            'AIRLINE_DELAY': np.where(rng.random(n) < 0.20, rng.normal(12, 8, n), 0).astype('float32'),
            'LATE_AIRCRAFT_DELAY': np.where(rng.random(n) < 0.18, rng.normal(15, 10, n), 0).astype('float32'),
            'WEATHER_DELAY': np.where(rng.random(n) < 0.10, rng.normal(8, 4, n), 0).astype('float32'),
        }
    )


@pytest.fixture
def airlines_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            'IATA_CODE': ['AA', 'DL', 'UA', 'WN', 'AS'],
            'AIRLINE': [
                'American Airlines',
                'Delta Air Lines',
                'United Airlines',
                'Southwest Airlines',
                'Alaska Airlines',
            ],
        }
    )


@pytest.fixture
def airports_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            'IATA_CODE': ['ATL', 'LAX', 'ORD', 'DFW', 'JFK'],
            'STATE': ['GA', 'CA', 'IL', 'TX', 'NY'],
            'CITY': ['Atlanta', 'Los Angeles', 'Chicago', 'Dallas', 'New York'],
            'COUNTRY': ['USA'] * 5,
        }
    )


@pytest.fixture
def flight_features_df(raw_flight_df, airlines_df, airports_df) -> pd.DataFrame:
    from src.features.feature_engineering import build_flight_features

    return build_flight_features(raw_flight_df, airlines_df, airports_df, use_ops=True)
