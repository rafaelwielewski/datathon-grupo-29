from __future__ import annotations

import json
import logging

from feast import FeatureStore

logger = logging.getLogger(__name__)


def demo_read_online(flight_id: int = 0) -> dict:
    fs = FeatureStore(repo_path='feature_store')
    features = [
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
    result = fs.get_online_features(
        features=features,
        entity_rows=[{'flight_id': int(flight_id)}],
    ).to_dict()
    logger.info('Online features for flight_id=%s loaded', flight_id)
    return result


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    data = demo_read_online(0)
    print(json.dumps(data, indent=2))
