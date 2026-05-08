from __future__ import annotations

import json
from pathlib import Path

import yaml
from langchain.tools import tool

ARTIFACTS_DIR = Path('data/processed/artifacts')
CONFIG_PATH = Path('configs/agent_config.yaml')


def _load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, encoding='utf-8') as f:
        return yaml.safe_load(f)


@tool
def predict_flight_delay(
    flight_info: str = '{"airline":"AA","origin":"ATL","destination":"LAX","month":6,"day":15,"day_of_week":2,"scheduled_departure":800,"scheduled_arrival":1100,"distance":1950,"scheduled_time":340}',
) -> str:
    """Predicts whether a US domestic flight will arrive 15+ minutes late.

    flight_info must be a JSON string with keys:
      airline, origin, destination, month, day, day_of_week,
      scheduled_departure (HHMM), scheduled_arrival (HHMM),
      distance (miles), scheduled_time (minutes).
    Returns delayed_probability, delayed (bool), and threshold.
    """
    from src.models.predictor import FlightParams, predict

    try:
        params = json.loads(flight_info)
    except (json.JSONDecodeError, ValueError) as exc:
        return json.dumps({'error': f'Invalid JSON: {exc}'})

    try:
        result = predict(FlightParams(**params))
        return json.dumps(
            {
                'delayed_probability': result.delayed_probability,
                'delayed': result.delayed,
                'threshold': result.threshold,
            }
        )
    except Exception as exc:
        return json.dumps({'error': f'Prediction error: {exc}'})


@tool
def get_airport_delay_stats(airport_code: str = 'ATL') -> str:
    """Returns historical delay statistics for a given US airport (IATA code)."""
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
    """Returns historical delay statistics for a given US airline (IATA code)."""
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
