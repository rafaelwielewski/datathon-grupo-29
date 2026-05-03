from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from langchain.tools import tool

ARTIFACTS_DIR = Path('data/processed/artifacts')
MISSING_TOKEN = 'UNKNOWN'
TRAIN_PRIOR = 0.18


def _build_inference_row(params: dict) -> pd.DataFrame:
    """Build inference row with ALL 54 features in correct order."""
    row = {
        'YEAR': int(params.get('year', 2015)),
        'MONTH': int(params.get('month', 6)),
        'DAY': int(params.get('day', 15)),
        'DAY_OF_WEEK': int(params.get('day_of_week', 3)),
        'AIRLINE': str(params.get('airline', MISSING_TOKEN)).upper(),
        'FLIGHT_NUMBER': 999,  # Placeholder
        'ORIGIN_AIRPORT': str(params.get('origin', MISSING_TOKEN)).upper(),
        'ORIGIN_STATE': 'GA',  # Placeholder mais comum
        'DESTINATION_AIRPORT': str(params.get('destination', MISSING_TOKEN)).upper(),
        'DEST_STATE': 'CA',  # Placeholder mais comum
        'ROUTE': str(params.get('origin', MISSING_TOKEN)).upper() + '_' + str(params.get('destination', MISSING_TOKEN)).upper(),
        'DISTANCE': float(params.get('distance', 1000)),
        'SCHEDULED_TIME': float(params.get('scheduled_time', 150)),
        'SCHEDULED_DEPARTURE': float(params.get('scheduled_departure', 800)),
        'SCHEDULED_ARRIVAL': float(params.get('scheduled_arrival', 1100)),
    }

    df = pd.DataFrame([row])

    # Time features
    df['IS_WEEKEND'] = int(df.loc[0, 'DAY_OF_WEEK'] in [6, 7])
    df['sched_dep_hour'] = int(df.loc[0, 'SCHEDULED_DEPARTURE'] / 100)
    df['sched_dep_minute'] = int(df.loc[0, 'SCHEDULED_DEPARTURE'] % 100)

    # Period features
    dep_hour = df.loc[0, 'sched_dep_hour']
    df['sched_dep_period'] = 'morning' if 5 <= dep_hour <= 11 else 'afternoon' if 12 <= dep_hour <= 17 else 'evening' if 18 <= dep_hour <= 22 else 'night'

    arr_hour = int(df.loc[0, 'SCHEDULED_ARRIVAL'] // 100)
    arr_minute = int(df.loc[0, 'SCHEDULED_ARRIVAL'] % 100)
    df['sched_arr_hour'] = arr_hour
    df['sched_arr_minute'] = arr_minute
    df['sched_arr_period'] = 'morning' if 5 <= arr_hour <= 11 else 'afternoon' if 12 <= arr_hour <= 17 else 'evening' if 18 <= arr_hour <= 22 else 'night'

    # Distance and date features (values placeholders)
    df['distance_bucket'] = 'long'  # Placeholder mais comum
    df['day_of_year'] = 180  # Placeholder
    df['doy_sin'] = 0.0  # Placeholder
    df['doy_cos'] = -1.0  # Placeholder

    # Congestion features (TRAIN_PRIOR placeholders)
    df['origin_day_hour_logvol'] = TRAIN_PRIOR
    df['dest_day_hour_logvol'] = TRAIN_PRIOR

    # Rolling target encodings (TRAIN_PRIOR)
    for col in ['te_airline_w7', 'te_airline_w30', 'te_origin_w7', 'te_origin_w30', 'te_dest_w7', 'te_dest_w30', 'te_route_w7', 'te_route_w30', 'te_origin_hour_w7', 'te_origin_hour_w30', 'te_origin_dow_w30']:
        df[col] = TRAIN_PRIOR

    # Operational features (TRAIN_PRIOR)
    for col in ['tail_dep_delay_mean_w3', 'tail_dep_delay_mean_w5', 'tail_dep_delay_mean_w10', 'tail_delayed_rate_w5', 'tail_delayed_rate_w10', 'origin_dep_delay_mean_w30', 'origin_dep_delay_mean_w90', 'origin_weather_rate_w90', 'origin_weather_rate_w180', 'origin_system_rate_w90', 'origin_system_rate_w180', 'origin_late_aircraft_rate_w90', 'origin_late_aircraft_rate_w180', 'origin_hour_dep_delay_mean_w30', 'route_dep_delay_mean_w30', 'route_delayed_rate_w30']:
        df[col] = TRAIN_PRIOR

    # TAIL_NUMBER placeholder
    df['TAIL_NUMBER'] = 'N999AA'  # Placeholder mais comum

    return df


def _run_model_pipeline(feature_df: pd.DataFrame) -> tuple[float, float]:
    import joblib
    from catboost import CatBoostClassifier, Pool

    model_path = ARTIFACTS_DIR / 'catboost_model.cbm'
    calibrator_path = ARTIFACTS_DIR / 'platt_calibrator.joblib'
    threshold_path = ARTIFACTS_DIR / 'best_threshold.txt'

    if not model_path.exists():
        return 0.5, 0.5

    threshold = 0.5
    if threshold_path.exists():
        threshold = float(threshold_path.read_text().strip())

    cat_cols = [
        c
        for c in ['AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT', 'ROUTE']
        if c in feature_df.columns
    ]
    cat_idx = [feature_df.columns.get_loc(c) for c in cat_cols]

    model = CatBoostClassifier()
    model.load_model(str(model_path))
    pool = Pool(feature_df, cat_features=cat_idx)
    proba = float(model.predict_proba(pool)[:, 1][0])

    if calibrator_path.exists():
        calibrator = joblib.load(calibrator_path)
        proba = float(calibrator.predict_proba([[proba]])[:, 1][0])

    return proba, threshold


@tool
def predict_flight_delay_llm(params: dict) -> str:
    """Chama endpoint /predict da API HTTP (usando CatBoost diretamente)."""
    import requests

    predict_endpoint = "http://localhost:8000/predict"

    try:
        response = requests.post(predict_endpoint, json=params, timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        return json.dumps({'error': f'Prediction API error: {exc}'})
    except Exception as exc:
        return json.dumps({'error': f'Internal error: {exc}'})


@tool
def predict_flight_delay_llm(params: str) -> str:
    """Chama endpoint /predict da API HTTP (usando CatBoost diretamente)."""
    import requests

    predict_endpoint = "http://localhost:8000/predict"

    try:
        response = requests.post(predict_endpoint, json=params, timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        return json.dumps({'error': f'Prediction API error: {exc}'})
    except Exception as exc:
        return json.dumps({'error': f'Internal error: {exc}'})


@tool
def predict_flight_delay(
    flight_info: str = '{"airline":"AA","origin":"ATL","destination":"LAX","month":6,"day":15,"day_of_week":2,"scheduled_departure":800,"scheduled_arrival":1100,"distance":1950,"scheduled_time":340}',
) -> str:
    """Predicts whether a flight will arrive 15+ minutes late.

    PRIMEIRO: Tenta usar predição direta via predict_flight_delay_llm (mais rápido e confiável).
    SE FALHAR: Retorna mensagem pedindo mais informações específicas.

    Fallback: A ferramenta não está implementada, retorna erro.
    """
    try:
        params = json.loads(flight_info)
    except (json.JSONDecodeError, ValueError) as exc:
        return json.dumps({'error': f'Invalid JSON: {exc}'})

    try:
        # Try new LLM-based prediction tool first
        result = predict_flight_delay_llm(flight_info)
        return result
    except Exception as exc:
        return json.dumps({'error': f'Internal error: {exc}'})


@tool
def get_airport_delay_stats(airport_code: str = 'ATL') -> str:
    """Returns historical delay statistics for a given airport."""
    try:
        stats = json.loads((ARTIFACTS_DIR / 'airport_stats.json').read_text())
        code = airport_code.strip().upper()
        if code not in stats:
            return json.dumps({'error': f"No data for airport '{code}'. Known: {list(stats.keys())[:10]}"})
        return json.dumps({'airport': code, **stats[code]})
    except Exception as exc:
        return json.dumps({'error': str(exc)})


@tool
def get_airline_delay_stats(airline_code: str = 'AA') -> str:
    """Returns historical delay statistics for a given airline."""
    try:
        stats = json.loads((ARTIFACTS_DIR / 'airline_stats.json').read_text())
        code = airline_code.strip().upper()
        if code not in stats:
            return json.dumps({'error': f"No data for airline '{code}'. Known: {list(stats.keys())}"})
        return json.dumps({'airline': code, **stats[code]})
    except Exception as exc:
        return json.dumps({'error': str(exc)})


def get_all_tools() -> list:
    return [predict_flight_delay, get_airport_delay_stats, get_airline_delay_stats]
