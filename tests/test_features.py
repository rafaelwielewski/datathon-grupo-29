from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.feature_engineering import (
    BASE_FEATURE_COLS,
    CAT_FEATURES,
    TRAIN_PRIOR,
    add_congestion_features,
    add_route,
    add_time_features,
    build_flight_features,
    make_date_cols,
)


def test_train_prior_is_float():
    assert isinstance(TRAIN_PRIOR, float)
    assert 0.0 < TRAIN_PRIOR < 1.0


def test_base_feature_cols_has_expected_length():
    assert len(BASE_FEATURE_COLS) >= 30


def test_cat_features_are_subset_of_base():
    for c in CAT_FEATURES:
        assert c in BASE_FEATURE_COLS, f"{c} not in BASE_FEATURE_COLS"


def test_add_time_features_extracts_hour_minute_period():
    df = pd.DataFrame({'SCHEDULED_DEPARTURE': [800.0, 1300.0, 2000.0, float('nan')]})
    result = add_time_features(df, 'SCHEDULED_DEPARTURE', 'sched_dep')
    assert result['sched_dep_hour'].iloc[0] == 8
    assert result['sched_dep_minute'].iloc[0] == 0
    assert result['sched_dep_period'].iloc[0] == 'morning'
    assert result['sched_dep_period'].iloc[1] == 'afternoon'
    assert result['sched_dep_period'].iloc[2] == 'evening'


def test_make_date_cols_creates_cyclical_encoding():
    df = pd.DataFrame({'YEAR': [2015], 'MONTH': [6], 'DAY': [15]})
    result = make_date_cols(df)
    assert 'doy_sin' in result.columns
    assert 'doy_cos' in result.columns
    assert 'day_of_year' in result.columns
    assert -1.0 <= float(result['doy_sin'].iloc[0]) <= 1.0


def test_add_route_concatenates_airports():
    df = pd.DataFrame({'ORIGIN_AIRPORT': ['ATL'], 'DESTINATION_AIRPORT': ['LAX']})
    result = add_route(df)
    assert result['ROUTE'].iloc[0] == 'ATL_LAX'


def test_add_congestion_requires_hour_cols():
    df = pd.DataFrame({'ORIGIN_AIRPORT': ['ATL'], 'DESTINATION_AIRPORT': ['LAX']})
    with pytest.raises(ValueError, match='Missing hour columns'):
        add_congestion_features(df)


def test_add_congestion_creates_logvol_cols(raw_flight_df):
    df = make_date_cols(raw_flight_df)
    df = add_time_features(df, 'SCHEDULED_DEPARTURE', 'sched_dep')
    df = add_time_features(df, 'SCHEDULED_ARRIVAL', 'sched_arr')
    result = add_congestion_features(df)
    assert 'origin_day_hour_logvol' in result.columns
    assert 'dest_day_hour_logvol' in result.columns
    assert (result['origin_day_hour_logvol'] >= 0).all()


def test_build_flight_features_has_all_base_cols(raw_flight_df, airlines_df, airports_df):
    result = build_flight_features(raw_flight_df, airlines_df, airports_df, use_ops=False)
    for col in BASE_FEATURE_COLS:
        assert col in result.columns, f"Missing: {col}"


def test_build_flight_features_with_ops_adds_ops_cols(raw_flight_df, airlines_df, airports_df):
    from src.features.feature_engineering import OPS_FEATURE_COLS
    result = build_flight_features(raw_flight_df, airlines_df, airports_df, use_ops=True)
    for col in OPS_FEATURE_COLS:
        assert col in result.columns, f"Missing ops col: {col}"


def test_build_flight_features_has_delayed_target(raw_flight_df, airlines_df, airports_df):
    result = build_flight_features(raw_flight_df, airlines_df, airports_df)
    assert 'delayed' in result.columns
    assert set(result['delayed'].unique()).issubset({0, 1})


def test_build_flight_features_filters_cancelled(raw_flight_df, airlines_df, airports_df):
    df = raw_flight_df.copy()
    df.loc[0, 'CANCELLED'] = 1
    result = build_flight_features(df, airlines_df, airports_df)
    assert len(result) < len(df)


def test_build_flight_features_no_nan_in_numeric(raw_flight_df, airlines_df, airports_df):
    from src.features.feature_engineering import OPS_FEATURE_COLS
    result = build_flight_features(raw_flight_df, airlines_df, airports_df, use_ops=True)
    num_cols = [
        c for c in (BASE_FEATURE_COLS + OPS_FEATURE_COLS)
        if c in result.columns and c not in CAT_FEATURES
    ]
    has_nan = result[num_cols].isnull().any()
    assert not has_nan.any(), f"NaN in: {has_nan[has_nan].index.tolist()}"
