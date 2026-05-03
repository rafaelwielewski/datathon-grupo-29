from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
import yaml
from catboost import CatBoostClassifier, Pool
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from src.features.feature_engineering import (
    BASE_FEATURE_COLS,
    CAT_FEATURES,
    OPS_FEATURE_COLS,
    build_flight_features,
)

logger = logging.getLogger(__name__)

CONFIG_PATH = Path('configs/model_config.yaml')


def _load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, encoding='utf-8') as f:
        return yaml.safe_load(f)


def _feature_columns(df: pd.DataFrame) -> list[str]:
    blocked = {
        'ARRIVAL_DELAY',
        'DEPARTURE_DELAY',
        'DELAYED',
        'delayed',
        'DELAY_SEVERITY',
        'CANCELLATION_REASON',
    }
    return [c for c in df.columns if c not in blocked]


def _categorical_columns(df: pd.DataFrame, feature_cols: list[str]) -> list[str]:
    cat_cols: list[str] = []
    for c in feature_cols:
        if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_categorical_dtype(df[c]):
            cat_cols.append(c)
    return cat_cols


def _sample_train_if_needed(train_df: pd.DataFrame, cfg: dict, random_state: int) -> pd.DataFrame:
    sampling_cfg = cfg.get('sampling', {})
    if not sampling_cfg.get('use_train_sampling', False):
        return train_df

    max_samples_per_month = int(sampling_cfg.get('max_samples_per_month', 200000))
    if 'MONTH' not in train_df.columns:
        return train_df.sample(min(len(train_df), max_samples_per_month), random_state=random_state)

    sampled_chunks = []
    for _, group in train_df.groupby('MONTH'):
        if len(group) > max_samples_per_month:
            sampled_chunks.append(group.sample(max_samples_per_month, random_state=random_state))
        else:
            sampled_chunks.append(group)
    return pd.concat(sampled_chunks, axis=0).sort_index()


def _optimize_threshold(y_true: np.ndarray, proba: np.ndarray, min_precision: float = 0.66) -> float:
    thresholds = np.arange(0.05, 0.96, 0.001)
    best_thr, best_recall = 0.5, -1.0

    for threshold in thresholds:
        pred = (proba >= threshold).astype(int)
        precision = precision_score(y_true, pred, zero_division=0)
        recall = recall_score(y_true, pred, zero_division=0)

        if precision >= min_precision and recall > best_recall:
            best_recall = recall
            best_thr = float(threshold)

    return best_thr


def _compute_airport_stats(df: pd.DataFrame) -> dict:
    stats: dict[str, dict] = {}
    grouped = df.groupby('ORIGIN_AIRPORT', dropna=False)

    for airport, group in grouped:
        if pd.isna(airport):
            continue

        top_delay_hours = {}
        if 'sched_dep_hour' in group.columns:
            delayed_rows = group[group['DELAYED'] == 1]
            top = delayed_rows['sched_dep_hour'].dropna().astype(int).value_counts().head(3)
            top_delay_hours = {str(k): int(v) for k, v in top.items()}

        stats[str(airport)] = {
            'delay_rate': round(float(group['DELAYED'].mean()), 4),
            'avg_delay_minutes': round(float(group['ARRIVAL_DELAY'].mean()), 2),
            'total_flights': int(len(group)),
            'top_delay_hours': top_delay_hours,
        }

    return stats


def _compute_airline_stats(df: pd.DataFrame, airlines_df: pd.DataFrame) -> dict:
    name_map = {}
    if {'IATA_CODE', 'AIRLINE'} <= set(airlines_df.columns):
        name_map = dict(zip(airlines_df['IATA_CODE'], airlines_df['AIRLINE']))

    stats: dict[str, dict] = {}
    grouped = df.groupby('AIRLINE', dropna=False)

    for airline, group in grouped:
        if pd.isna(airline):
            continue

        code = str(airline)
        stats[code] = {
            'delay_rate': round(float(group['DELAYED'].mean()), 4),
            'avg_delay_minutes': round(float(group['ARRIVAL_DELAY'].mean()), 2),
            'total_flights': int(len(group)),
            'name': name_map.get(code, code),
        }

    return stats


def train(config_path: Path = CONFIG_PATH) -> dict:
    cfg = _load_config(config_path)

    data_cfg = cfg['data']
    split_cfg = cfg['splits']
    model_cfg = cfg['model']
    paths_cfg = cfg['paths']
    cb_cfg = dict(cfg.get('catboost_params', {}))

    random_state = 42
    np.random.seed(random_state)

    artifacts_dir = Path(paths_cfg['artifacts_dir'])
    data_dir = Path(paths_cfg['data_dir'])
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    logger.info('Loading CSVs...')
    flights_df = pd.read_csv(data_cfg['flights_csv'], low_memory=False)
    airlines_df = pd.read_csv(data_cfg['airlines_csv'])
    airports_df = pd.read_csv(data_cfg['airports_csv'])
    logger.info('flights shape: %s', flights_df.shape)

    use_ops = bool(model_cfg.get('use_operational_delay_cols', True))
    logger.info('Building features (use_ops=%s)...', use_ops)
    engineered = build_flight_features(
        flights_df,
        airlines_df=airlines_df,
        airports_df=airports_df,
        use_ops=use_ops,
    )
    logger.info(
        'Engineered shape: %s | delayed rate: %.4f',
        engineered.shape,
        engineered.get('delayed', engineered.get('DELAYED', pd.Series([0]))).mean(),
    )

    threshold_minutes = int(cfg.get('target', {}).get('threshold_minutes', 15))
    engineered['DELAYED'] = (engineered['ARRIVAL_DELAY'] >= threshold_minutes).astype(int)

    train_months = set(split_cfg['train_months'])
    val_months = set(split_cfg['val_months'])
    test_months = set(split_cfg['test_months'])

    train_df = engineered[engineered['MONTH'].isin(train_months)].copy()
    val_df = engineered[engineered['MONTH'].isin(val_months)].copy()
    test_df = engineered[engineered['MONTH'].isin(test_months)].copy()

    train_df = _sample_train_if_needed(train_df, cfg, random_state)

    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError('Train/val/test split resulted in empty dataset. Check month config and source data.')

    all_feat = BASE_FEATURE_COLS + (OPS_FEATURE_COLS if use_ops else [])
    feature_cols = [c for c in all_feat if c in engineered.columns]
    cat_cols = [c for c in CAT_FEATURES if c in feature_cols]
    logger.info('Feature cols: %d (%d categorical)', len(feature_cols), len(cat_cols))

    for frame in (train_df, val_df, test_df):
        for c in cat_cols:
            frame[c] = frame[c].astype(str).fillna('MISSING')
        for c in feature_cols:
            if c not in cat_cols:
                frame[c] = pd.to_numeric(frame[c], errors='coerce').fillna(0.18)

    X_train = train_df[feature_cols]
    y_train = train_df['DELAYED'].astype(int).to_numpy()
    X_val = val_df[feature_cols]
    y_val = val_df['DELAYED'].astype(int).to_numpy()
    X_test = test_df[feature_cols]
    y_test = test_df['DELAYED'].astype(int).to_numpy()

    train_prior = float(y_train.mean())
    val_prior = float(y_val.mean())
    test_prior = float(y_test.mean())

    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    ratio = (neg / max(pos, 1)) if pos > 0 else 1.0
    cb_cfg['scale_pos_weight'] = float(ratio * float(model_cfg.get('scale_pos_weight_multiplier', 0.85)))

    if bool(cb_cfg.pop('use_gpu', False)):
        cb_cfg['task_type'] = 'GPU'
    else:
        cb_cfg['task_type'] = 'CPU'

    cat_idx = [feature_cols.index(c) for c in cat_cols]
    train_pool = Pool(X_train, y_train, cat_features=cat_idx)
    val_pool = Pool(X_val, y_val, cat_features=cat_idx)
    test_pool = Pool(X_test, y_test, cat_features=cat_idx)

    model = CatBoostClassifier(**cb_cfg)

    mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'sqlite:///mlflow.db'))
    mlflow.set_experiment('datathon-grupo-29')

    with mlflow.start_run(run_name='catboost-flight-delay-training') as run:
        mlflow.set_tag('model_type', 'classification')
        mlflow.set_tag('framework', 'catboost')
        mlflow.set_tag('owner', 'grupo-29')
        mlflow.set_tag('phase', 'datathon-fase05')

        mlflow.log_params(
            {
                'threshold_minutes': threshold_minutes,
                'n_features': len(feature_cols),
                'n_cat_features': len(cat_cols),
                'n_train': int(len(X_train)),
                'n_val': int(len(X_val)),
                'n_test': int(len(X_test)),
                'train_prior': train_prior,
                'val_prior': val_prior,
                'test_prior': test_prior,
                **{f'catboost_{k}': v for k, v in cb_cfg.items()},
            }
        )

        model.fit(train_pool, eval_set=val_pool, use_best_model=True)

        val_proba_raw = model.predict_proba(val_pool)[:, 1]
        calibrator = LogisticRegression(solver='lbfgs', max_iter=1000, random_state=random_state)
        calibrator.fit(val_proba_raw.reshape(-1, 1), y_val)

        test_proba_raw = model.predict_proba(test_pool)[:, 1]
        test_proba_cal = calibrator.predict_proba(test_proba_raw.reshape(-1, 1))[:, 1]

        best_threshold = _optimize_threshold(y_val, calibrator.predict_proba(val_proba_raw.reshape(-1, 1))[:, 1])
        test_pred = (test_proba_cal >= best_threshold).astype(int)

        auc = float(roc_auc_score(y_test, test_proba_cal))
        precision = float(precision_score(y_test, test_pred, zero_division=0))
        recall = float(recall_score(y_test, test_pred, zero_division=0))
        f1 = float(f1_score(y_test, test_pred, zero_division=0))

        mlflow.log_metrics(
            {
                'test_auc': auc,
                'test_precision': precision,
                'test_recall': recall,
                'test_f1': f1,
                'best_threshold': float(best_threshold),
            }
        )

        model.save_model(str(artifacts_dir / 'catboost_model.cbm'))
        joblib.dump(calibrator, artifacts_dir / 'platt_calibrator.joblib')
        (artifacts_dir / 'best_threshold.txt').write_text(f'{best_threshold:.6f}', encoding='utf-8')

        airport_stats = _compute_airport_stats(train_df)
        airline_stats = _compute_airline_stats(train_df, airlines_df)

        (artifacts_dir / 'airport_stats.json').write_text(json.dumps(airport_stats, indent=2, ensure_ascii=False), encoding='utf-8')
        (artifacts_dir / 'airline_stats.json').write_text(json.dumps(airline_stats, indent=2, ensure_ascii=False), encoding='utf-8')

        airport_state_map = airports_df.set_index('IATA_CODE')['STATE'].to_dict()
        (artifacts_dir / 'airport_state_map.json').write_text(json.dumps(airport_state_map, ensure_ascii=False), encoding='utf-8')

        route_stats = (
            engineered.groupby('ROUTE', dropna=False)
            .agg(distance=('DISTANCE', 'median'), scheduled_time=('SCHEDULED_TIME', 'median'), n_flights=('DISTANCE', 'count'))
            .round(0)
            .astype({'distance': int, 'scheduled_time': int, 'n_flights': int})
            .to_dict(orient='index')
        )
        (artifacts_dir / 'route_stats.json').write_text(json.dumps(route_stats, ensure_ascii=False), encoding='utf-8')

        drift_export = pd.concat(
            [
                train_df.assign(split='train')[feature_cols + ['split']],
                test_df.assign(split='test')[feature_cols + ['split']],
            ],
            axis=0,
        )
        drift_path = data_dir / 'test_features_ref.parquet'
        drift_export.to_parquet(drift_path, index=False)

        metadata = {
            'model': 'CatBoostClassifier',
            'target': 'DELAYED',
            'threshold_minutes': threshold_minutes,
            'feature_columns': feature_cols,
            'categorical_columns': cat_cols,
            'splits': {
                'train_months': sorted(train_months),
                'val_months': sorted(val_months),
                'test_months': sorted(test_months),
                'train_rows': int(len(train_df)),
                'val_rows': int(len(val_df)),
                'test_rows': int(len(test_df)),
            },
            'priors': {
                'train': train_prior,
                'val': val_prior,
                'test': test_prior,
            },
            'mlflow_run_id': run.info.run_id,
            'trained_at': datetime.now().isoformat(),
        }
        metrics = {
            'test_auc': auc,
            'test_precision': precision,
            'test_recall': recall,
            'test_f1': f1,
            'best_threshold': float(best_threshold),
        }

        (artifacts_dir / 'metadata.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding='utf-8')
        (artifacts_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding='utf-8')

        mlflow.log_artifacts(str(artifacts_dir))
        mlflow.log_artifact(str(drift_path))

        model_name = os.getenv('MLFLOW_MODEL_NAME', 'flight-delay-catboost')
        approve_model = os.getenv('MLFLOW_APPROVE', 'false').lower() == 'true'

        try:
            from mlflow import pyfunc
            from mlflow.tracking import MlflowClient

            class _FlightDelayPyfunc(pyfunc.PythonModel):
                def load_context(self, context):
                    from catboost import CatBoostClassifier

                    self.model = CatBoostClassifier()
                    self.model.load_model(context.artifacts['catboost_model'])
                    self.calibrator = joblib.load(context.artifacts['calibrator'])

                def predict(self, context, model_input):
                    proba_raw = self.model.predict_proba(model_input)[:, 1]
                    proba_cal = self.calibrator.predict_proba(proba_raw.reshape(-1, 1))[:, 1]
                    return proba_cal

            model_info = pyfunc.log_model(
                artifact_path='model_package',
                python_model=_FlightDelayPyfunc(),
                artifacts={
                    'catboost_model': str(artifacts_dir / 'catboost_model.cbm'),
                    'calibrator': str(artifacts_dir / 'platt_calibrator.joblib'),
                },
                registered_model_name=model_name,
            )

            client = MlflowClient()
            version = getattr(model_info, 'registered_model_version', None)
            if version is None:
                latest = client.get_latest_versions(model_name, stages=['None'])
                if latest:
                    version = latest[-1].version

            if version is not None:
                client.set_model_version_tag(
                    model_name,
                    version,
                    'approval_status',
                    'approved' if approve_model else 'pending',
                )
                if approve_model:
                    client.transition_model_version_stage(
                        name=model_name,
                        version=version,
                        stage='Staging',
                        archive_existing_versions=True,
                    )
        except Exception as exc:
            logger.warning('Model registry step failed: %s', exc)

    logger.info('Training complete. Artifacts saved to %s', artifacts_dir.resolve())
    return metrics


def main() -> None:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    train(CONFIG_PATH)


if __name__ == '__main__':
    main()
