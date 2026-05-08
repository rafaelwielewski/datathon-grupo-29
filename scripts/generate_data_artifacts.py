"""
Generates lightweight serving artifacts from raw CSVs without running full training.
Run via: make data
Produces:
  data/processed/artifacts/airport_state_map.json
  data/processed/artifacts/route_stats.json
  data/processed/sample/flights_sample.csv  (10k rows sampled from real data)
  data/processed/sample/flights_synthetic.csv  (2k rows synthetic, no PII, safe for dev)
"""

from __future__ import annotations

import csv
import json
import logging
import pathlib
import random

logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger(__name__)

ROOT = pathlib.Path(__file__).parent.parent
RAW = ROOT / 'data' / 'raw'
ARTIFACTS = ROOT / 'data' / 'processed' / 'artifacts'
SAMPLE_DIR = ROOT / 'data' / 'processed' / 'sample'

ARTIFACTS.mkdir(parents=True, exist_ok=True)
SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_ROWS = 10_000
SYNTHETIC_ROWS = 2_000
RANDOM_SEED = 42

_AIRLINES = ['AA', 'WN', 'DL', 'UA', 'OO', 'EV', 'B6', 'AS', 'MQ', 'NK']
_AIRPORTS = [
    'ATL',
    'LAX',
    'ORD',
    'DFW',
    'DEN',
    'JFK',
    'SFO',
    'SEA',
    'LAS',
    'MCO',
    'PHX',
    'IAH',
    'MIA',
    'BOS',
    'MSP',
    'DTW',
    'FLL',
    'LGA',
    'BWI',
    'SLC',
]
_STATES = {
    'ATL': 'GA',
    'LAX': 'CA',
    'ORD': 'IL',
    'DFW': 'TX',
    'DEN': 'CO',
    'JFK': 'NY',
    'SFO': 'CA',
    'SEA': 'WA',
    'LAS': 'NV',
    'MCO': 'FL',
    'PHX': 'AZ',
    'IAH': 'TX',
    'MIA': 'FL',
    'BOS': 'MA',
    'MSP': 'MN',
    'DTW': 'MI',
    'FLL': 'FL',
    'LGA': 'NY',
    'BWI': 'MD',
    'SLC': 'UT',
}


def generate_airport_state_map() -> None:
    airports_csv = RAW / 'airports.csv'
    if not airports_csv.exists():
        log.warning('airports.csv not found — skipping airport_state_map')
        return
    mapping: dict[str, str] = {}
    with open(airports_csv) as f:
        for row in csv.DictReader(f):
            mapping[row['IATA_CODE']] = row['STATE']
    out = ARTIFACTS / 'airport_state_map.json'
    out.write_text(json.dumps(mapping))
    log.info('airport_state_map.json — %d airports', len(mapping))


def generate_route_stats_and_sample() -> None:
    flights_csv = RAW / 'flights.csv'
    if not flights_csv.exists():
        log.warning('flights.csv not found — skipping route_stats and sample')
        return

    from collections import defaultdict

    distances: dict[str, list[float]] = defaultdict(list)
    times: dict[str, list[float]] = defaultdict(list)

    random.seed(RANDOM_SEED)
    header: list[str] = []

    log.info('Reading flights.csv ...')
    with open(flights_csv) as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        all_rows: list[dict] = []
        for row in reader:
            if row.get('CANCELLED') == '1':
                continue
            orig = row.get('ORIGIN_AIRPORT', '').strip()
            dest = row.get('DESTINATION_AIRPORT', '').strip()
            try:
                distances[f'{orig}_{dest}'].append(float(row['DISTANCE']))
                times[f'{orig}_{dest}'].append(float(row['SCHEDULED_TIME']))
            except (ValueError, KeyError):
                pass
            all_rows.append(row)

    # Route stats
    route_stats = {
        key: {
            'distance': round(sum(v) / len(v)),
            'scheduled_time': round(sum(times[key]) / len(times[key])),
            'n_flights': len(v),
        }
        for key, v in distances.items()
    }
    out = ARTIFACTS / 'route_stats.json'
    out.write_text(json.dumps(route_stats))
    log.info('route_stats.json — %d routes', len(route_stats))

    # Sample
    sample = random.sample(all_rows, min(SAMPLE_ROWS, len(all_rows)))
    sample_path = SAMPLE_DIR / 'flights_sample.csv'
    with open(sample_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(sample)
    log.info('flights_sample.csv — %d rows → %s', len(sample), sample_path)


def generate_synthetic_flights() -> None:
    rng = random.Random(RANDOM_SEED)
    fieldnames = [
        'YEAR',
        'MONTH',
        'DAY',
        'DAY_OF_WEEK',
        'AIRLINE',
        'ORIGIN_AIRPORT',
        'DESTINATION_AIRPORT',
        'SCHEDULED_DEPARTURE',
        'DEPARTURE_DELAY',
        'SCHEDULED_TIME',
        'DISTANCE',
        'SCHEDULED_ARRIVAL',
        'ARRIVAL_DELAY',
        'DIVERTED',
        'CANCELLED',
    ]
    rows: list[dict] = []
    for _ in range(SYNTHETIC_ROWS):
        orig, dest = rng.sample(_AIRPORTS, 2)
        month = rng.randint(1, 12)
        day = rng.randint(1, 28)
        # day_of_week: 1=Mon … 7=Sun (matches dataset convention)
        dow = rng.randint(1, 7)
        sched_dep = rng.randint(0, 2359)
        distance = round(rng.uniform(150, 2800))
        sched_time = round(distance / 7.5 + rng.uniform(-15, 15))
        sched_arr = (sched_dep + sched_time) % 2400
        dep_delay = round(rng.gauss(5, 25))
        arr_delay = round(dep_delay + rng.gauss(0, 10))
        rows.append(
            {
                'YEAR': 2015,
                'MONTH': month,
                'DAY': day,
                'DAY_OF_WEEK': dow,
                'AIRLINE': rng.choice(_AIRLINES),
                'ORIGIN_AIRPORT': orig,
                'DESTINATION_AIRPORT': dest,
                'SCHEDULED_DEPARTURE': sched_dep,
                'DEPARTURE_DELAY': dep_delay,
                'SCHEDULED_TIME': sched_time,
                'DISTANCE': distance,
                'SCHEDULED_ARRIVAL': sched_arr,
                'ARRIVAL_DELAY': arr_delay,
                'DIVERTED': 0,
                'CANCELLED': 0,
            }
        )
    out = SAMPLE_DIR / 'flights_synthetic.csv'
    with open(out, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    log.info('flights_synthetic.csv — %d rows → %s', len(rows), out)


if __name__ == '__main__':
    generate_airport_state_map()
    generate_route_stats_and_sample()
    generate_synthetic_flights()
    log.info('Done. Artifacts ready in %s', ARTIFACTS)
