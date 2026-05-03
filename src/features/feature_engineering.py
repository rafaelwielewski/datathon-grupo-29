from __future__ import annotations

import numpy as np
import pandas as pd

MISSING_TOKEN = 'MISSING'
TRAIN_PRIOR = 0.18

BASE_FEATURE_COLS = [
    'YEAR', 'MONTH', 'DAY', 'DAY_OF_WEEK',
    'AIRLINE', 'FLIGHT_NUMBER', 'TAIL_NUMBER',
    'ORIGIN_AIRPORT', 'ORIGIN_STATE',
    'DESTINATION_AIRPORT', 'DEST_STATE',
    'ROUTE',
    'DISTANCE', 'SCHEDULED_TIME',
    'sched_dep_hour', 'sched_dep_minute', 'sched_dep_period',
    'sched_arr_hour', 'sched_arr_minute', 'sched_arr_period',
    'is_weekend',
    'distance_bucket',
    'day_of_year', 'doy_sin', 'doy_cos',
    'origin_day_hour_logvol', 'dest_day_hour_logvol',
    'te_airline_w7', 'te_airline_w30',
    'te_origin_w7', 'te_origin_w30',
    'te_dest_w7', 'te_dest_w30',
    'te_route_w7', 'te_route_w30',
    'te_origin_hour_w7', 'te_origin_hour_w30',
    'te_origin_dow_w30',
]

OPS_FEATURE_COLS = [
    'tail_dep_delay_mean_w3', 'tail_dep_delay_mean_w5', 'tail_dep_delay_mean_w10',
    'tail_delayed_rate_w5', 'tail_delayed_rate_w10',
    'origin_dep_delay_mean_w30', 'origin_dep_delay_mean_w90',
    'origin_weather_rate_w90', 'origin_weather_rate_w180',
    'origin_system_rate_w90', 'origin_system_rate_w180',
    'origin_late_aircraft_rate_w90', 'origin_late_aircraft_rate_w180',
    'origin_hour_dep_delay_mean_w30',
    'route_dep_delay_mean_w30',
    'route_delayed_rate_w30',
]

CAT_FEATURES = [
    'AIRLINE', 'ORIGIN_AIRPORT', 'ORIGIN_STATE',
    'DESTINATION_AIRPORT', 'DEST_STATE',
    'sched_dep_period', 'sched_arr_period',
    'distance_bucket', 'ROUTE', 'TAIL_NUMBER',
]

FEATURE_COLS = BASE_FEATURE_COLS


def _to_hour_min(val):
    if pd.isna(val):
        return (np.nan, np.nan)
    try:
        s = str(int(val)).zfill(4)
        hh, mm = int(s[:2]), int(s[2:])
        if not (0 <= hh <= 23 and 0 <= mm <= 59):
            return (np.nan, np.nan)
        return (hh, mm)
    except Exception:
        return (np.nan, np.nan)


def add_time_features(df: pd.DataFrame, col: str, prefix: str) -> pd.DataFrame:
    df = df.copy()
    hm = df[col].apply(_to_hour_min)
    df[f'{prefix}_hour'] = hm.apply(lambda x: x[0])
    df[f'{prefix}_minute'] = hm.apply(lambda x: x[1])

    def _period(h):
        if pd.isna(h):
            return np.nan
        h = int(h)
        if 5 <= h <= 11:
            return 'morning'
        if 12 <= h <= 17:
            return 'afternoon'
        if 18 <= h <= 22:
            return 'evening'
        return 'night'

    df[f'{prefix}_period'] = df[f'{prefix}_hour'].apply(_period)
    return df


def distance_bucket(d) -> str | float:
    if pd.isna(d):
        return np.nan
    if d < 500:
        return 'short'
    if d < 1500:
        return 'medium'
    return 'long'


def sanitize_cat_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        df[c] = df[c].astype('object').where(df[c].notna(), MISSING_TOKEN).astype(str)
        df.loc[df[c].isin(['nan', 'NaN', '<NA>', 'None']), c] = MISSING_TOKEN
    return df


def make_date_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['flight_date'] = pd.to_datetime(
        dict(
            year=df['YEAR'].astype('int32'),
            month=df['MONTH'].astype('int32'),
            day=df['DAY'].astype('int32'),
        ),
        errors='coerce',
    )
    df = df[~df['flight_date'].isna()].copy()
    df['day_of_year'] = df['flight_date'].dt.dayofyear.astype('int16')
    two_pi = 2.0 * np.pi
    df['doy_sin'] = np.sin(two_pi * df['day_of_year'] / 365.0).astype('float32')
    df['doy_cos'] = np.cos(two_pi * df['day_of_year'] / 365.0).astype('float32')
    return df


def add_route(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['ROUTE'] = df['ORIGIN_AIRPORT'].astype(str) + '_' + df['DESTINATION_AIRPORT'].astype(str)
    return df


def add_congestion_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'sched_dep_hour' not in df.columns or 'sched_arr_hour' not in df.columns:
        raise ValueError('Missing hour columns; call add_time_features first')
    g1 = df.groupby(['flight_date', 'ORIGIN_AIRPORT', 'sched_dep_hour'], dropna=False).size()
    df['origin_day_hour_volume'] = (
        df.set_index(['flight_date', 'ORIGIN_AIRPORT', 'sched_dep_hour']).index.map(g1).astype('float32')
    )
    g2 = df.groupby(['flight_date', 'DESTINATION_AIRPORT', 'sched_arr_hour'], dropna=False).size()
    df['dest_day_hour_volume'] = (
        df.set_index(['flight_date', 'DESTINATION_AIRPORT', 'sched_arr_hour']).index.map(g2).astype('float32')
    )
    df['origin_day_hour_logvol'] = np.log1p(df['origin_day_hour_volume']).astype('float32')
    df['dest_day_hour_logvol'] = np.log1p(df['dest_day_hour_volume']).astype('float32')
    return df


def add_rolling_target_rate(
    df: pd.DataFrame, group_cols: list[str], out_col: str, windows: tuple = (7, 30)
) -> pd.DataFrame:
    df = df.copy().sort_values('flight_date')
    for w in windows:
        df[f'{out_col}_w{w}'] = (
            df.groupby(group_cols, dropna=False)['delayed']
            .transform(lambda s: s.shift(1).rolling(w, min_periods=1).mean())
            .astype('float32')
        )
    return df


def add_rolling_mean_feature(
    df: pd.DataFrame, group_cols: list[str], value_col: str, out_col: str, windows: tuple = (30,)
) -> pd.DataFrame:
    df = df.copy().sort_values('flight_date')
    for w in windows:
        df[f'{out_col}_w{w}'] = (
            df.groupby(group_cols, dropna=False)[value_col]
            .transform(lambda s: s.shift(1).rolling(w, min_periods=1).mean())
            .astype('float32')
        )
    return df


def add_rolling_rate_feature(
    df: pd.DataFrame, group_cols: list[str], flag_col: str, out_col: str, windows: tuple = (90,)
) -> pd.DataFrame:
    return add_rolling_mean_feature(df, group_cols, flag_col, out_col, windows)


def build_flight_features(
    flights_df: pd.DataFrame,
    airlines_df: pd.DataFrame,
    airports_df: pd.DataFrame,
    use_ops: bool = True,
) -> pd.DataFrame:
    df = flights_df.copy()

    airlines_ref = airlines_df.rename(columns={'IATA_CODE': 'AIRLINE', 'AIRLINE': 'AIRLINE_NAME'})[['AIRLINE', 'AIRLINE_NAME']]
    df = df.merge(airlines_ref, on='AIRLINE', how='left')

    origin_ref = airports_df.rename(columns={'IATA_CODE': 'ORIGIN_AIRPORT', 'STATE': 'ORIGIN_STATE'})[['ORIGIN_AIRPORT', 'ORIGIN_STATE']]
    df = df.merge(origin_ref, on='ORIGIN_AIRPORT', how='left')

    dest_ref = airports_df.rename(columns={'IATA_CODE': 'DESTINATION_AIRPORT', 'STATE': 'DEST_STATE'})[['DESTINATION_AIRPORT', 'DEST_STATE']]
    df = df.merge(dest_ref, on='DESTINATION_AIRPORT', how='left')

    df = df[(df['CANCELLED'] == 0) & (df['DIVERTED'] == 0)].copy()
    df = df[~pd.isna(df['ARRIVAL_DELAY'])].copy()
    df['delayed'] = (df['ARRIVAL_DELAY'] >= 15).astype(int)

    df = make_date_cols(df)
    df = add_time_features(df, 'SCHEDULED_DEPARTURE', 'sched_dep')
    df = add_time_features(df, 'SCHEDULED_ARRIVAL', 'sched_arr')
    df['is_weekend'] = df['DAY_OF_WEEK'].isin([6, 7]).astype('int8')
    df['distance_bucket'] = df['DISTANCE'].apply(distance_bucket)
    df = add_route(df)
    df = add_congestion_features(df)

    df = add_rolling_target_rate(df, ['AIRLINE'], 'te_airline', windows=(7, 30))
    df = add_rolling_target_rate(df, ['ORIGIN_AIRPORT'], 'te_origin', windows=(7, 30))
    df = add_rolling_target_rate(df, ['DESTINATION_AIRPORT'], 'te_dest', windows=(7, 30))
    df = add_rolling_target_rate(df, ['ORIGIN_AIRPORT', 'DESTINATION_AIRPORT'], 'te_route', windows=(7, 30))
    df = add_rolling_target_rate(df, ['ORIGIN_AIRPORT', 'sched_dep_hour'], 'te_origin_hour', windows=(7, 30))
    df = add_rolling_target_rate(df, ['ORIGIN_AIRPORT', 'DAY_OF_WEEK'], 'te_origin_dow', windows=(30,))

    if use_ops and 'WEATHER_DELAY' in df.columns:
        df['flag_weather_delay'] = (df['WEATHER_DELAY'].fillna(0) > 0).astype('int8')
        df['flag_system_delay'] = (df['AIR_SYSTEM_DELAY'].fillna(0) > 0).astype('int8')
        df['flag_late_aircraft'] = (df['LATE_AIRCRAFT_DELAY'].fillna(0) > 0).astype('int8')

        df = add_rolling_mean_feature(df, ['TAIL_NUMBER'], 'DEPARTURE_DELAY', 'tail_dep_delay_mean', windows=(3, 5, 10))
        df = add_rolling_target_rate(df, ['TAIL_NUMBER'], 'tail_delayed_rate', windows=(5, 10))
        df = add_rolling_mean_feature(df, ['ORIGIN_AIRPORT'], 'DEPARTURE_DELAY', 'origin_dep_delay_mean', windows=(30, 90))
        df = add_rolling_rate_feature(df, ['ORIGIN_AIRPORT'], 'flag_weather_delay', 'origin_weather_rate', windows=(90, 180))
        df = add_rolling_rate_feature(df, ['ORIGIN_AIRPORT'], 'flag_system_delay', 'origin_system_rate', windows=(90, 180))
        df = add_rolling_rate_feature(df, ['ORIGIN_AIRPORT'], 'flag_late_aircraft', 'origin_late_aircraft_rate', windows=(90, 180))
        df = add_rolling_mean_feature(df, ['ORIGIN_AIRPORT', 'sched_dep_hour'], 'DEPARTURE_DELAY', 'origin_hour_dep_delay_mean', windows=(30,))
        df = add_rolling_mean_feature(df, ['ROUTE'], 'DEPARTURE_DELAY', 'route_dep_delay_mean', windows=(30,))
        df = add_rolling_target_rate(df, ['ROUTE'], 'route_delayed_rate', windows=(30,))

    all_feat_cols = BASE_FEATURE_COLS + (OPS_FEATURE_COLS if use_ops else [])
    present_feat_cols = [c for c in all_feat_cols if c in df.columns]
    num_feat_cols = [c for c in present_feat_cols if c not in CAT_FEATURES]
    for c in num_feat_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(TRAIN_PRIOR).astype('float32')

    cat_cols_present = [c for c in CAT_FEATURES if c in df.columns]
    df = sanitize_cat_cols(df, cat_cols_present)
    return df
