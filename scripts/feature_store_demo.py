from __future__ import annotations

import json
import logging

from src.models.predictor import _FEATURE_STORE_FEATURES, _get_feast_store

logger = logging.getLogger(__name__)


def demo_read_online(flight_id: int = 0) -> dict:
    fs = _get_feast_store()
    result = fs.get_online_features(
        features=_FEATURE_STORE_FEATURES,
        entity_rows=[{'flight_id': int(flight_id)}],
    ).to_dict()
    logger.info('Online features for flight_id=%s loaded', flight_id)
    return result


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    data = demo_read_online(0)
    print(json.dumps(data, indent=2))
