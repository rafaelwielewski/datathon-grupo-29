from __future__ import annotations

from pathlib import Path

from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int64, String

_PROJECT_ROOT = Path(__file__).parent.parent

flight_feature_source = FileSource(
    path=str(_PROJECT_ROOT / 'data/processed/feature_store/flight_features.parquet'),
    event_timestamp_column='event_timestamp',
)

flight_entity = Entity(
    name='flight',
    join_keys=['flight_id'],
)

flight_features_view = FeatureView(
    name='flight_features',
    entities=[flight_entity],
    ttl=None,
    schema=[
        Field(name='YEAR', dtype=Int64),
        Field(name='MONTH', dtype=Int64),
        Field(name='DAY', dtype=Int64),
        Field(name='DAY_OF_WEEK', dtype=Int64),
        Field(name='sched_dep_hour', dtype=Int64),
        Field(name='sched_dep_minute', dtype=Int64),
        Field(name='sched_arr_hour', dtype=Int64),
        Field(name='sched_arr_minute', dtype=Int64),
        Field(name='DISTANCE', dtype=Float32),
        Field(name='SCHEDULED_TIME', dtype=Float32),
        Field(name='is_weekend', dtype=Int64),
        Field(name='distance_bucket', dtype=String),
        Field(name='AIRLINE', dtype=String),
        Field(name='ORIGIN_AIRPORT', dtype=String),
        Field(name='DESTINATION_AIRPORT', dtype=String),
        Field(name='ROUTE', dtype=String),
    ],
    source=flight_feature_source,
)
