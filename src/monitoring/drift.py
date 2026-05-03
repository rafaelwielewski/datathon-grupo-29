from __future__ import annotations

import logging
from pathlib import Path

import mlflow
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

CONFIG_PATH = Path('configs/monitoring_config.yaml')


def _load_config(path: Path | None = None) -> dict:
    if path is None:
        path = CONFIG_PATH
    if not path.exists():
        return {
            'drift': {
                'warning_threshold': 0.1,
                'retrain_threshold': 0.2,
                'reference_parquet': 'data/processed/test_features_ref.parquet',
                'reference_split': 'train',
                'current_split': 'test',
            }
        }
    with open(path, encoding='utf-8') as f:
        return yaml.safe_load(f)


def run_drift_report(reference_df: pd.DataFrame, current_df: pd.DataFrame, warning_threshold: float = 0.1) -> dict:
    """Detecta drift de features usando Evidently DataDriftPreset.

    Args:
        reference_df: DataFrame de referência (dados de treino).
        current_df: DataFrame atual (dados de produção recentes).
        warning_threshold: Threshold para detecção de drift.

    Returns:
        Dict com drift_detected, drift_share, per_feature e thresholds.
    """
    from evidently import Report
    from evidently.presets import DataDriftPreset

    report = Report(metrics=[DataDriftPreset()])
    run_result = report.run(reference_data=reference_df, current_data=current_df)
    result_dict = run_result.dict()

    # Na v0.7.21, o preset gera várias métricas.
    # A primeira métrica (índice 0) costuma ser o DriftedColumnsCount que tem o share global.
    drift_share = 0.0
    per_feature = {}

    for metric in result_dict.get('metrics', []):
        metric_name = metric.get('metric_name', '')
        # Verifica se é a métrica de contagem global
        if 'DriftedColumnsCount' in metric_name:
            drift_share = float(metric.get('value', {}).get('share', 0.0))
        # Verifica se é a métrica de drift de uma coluna específica
        elif 'ValueDrift' in metric_name:
            column_name = metric.get('config', {}).get('column')
            if column_name:
                # Na v0.7.21, ValueDrift retorna o p-value.
                # Drift é detectado se p-value < threshold (default 0.05)
                p_value = metric.get('value', 1.0)
                threshold = metric.get('config', {}).get('threshold', 0.05)
                per_feature[column_name] = bool(p_value < threshold)

    drift_detected = drift_share > warning_threshold

    logger.info('Drift report: share=%.2f detected=%s', drift_share, drift_detected)

    return {
        'drift_detected': drift_detected,
        'drift_share': drift_share,
        'per_feature': per_feature,
    }


def detect_and_log_drift() -> dict:
    """Carrega parquet de features, detecta drift treino vs teste e loga no MLflow."""

    cfg = _load_config()
    drift_cfg = cfg['drift']
    ref_parquet = Path(drift_cfg.get('reference_parquet', 'data/processed/test_features_ref.parquet'))
    ref_split = drift_cfg.get('reference_split', 'train')
    cur_split = drift_cfg.get('current_split', 'test')
    warning_threshold = drift_cfg['warning_threshold']
    retrain_threshold = drift_cfg.get('retrain_threshold', 0.2)

    if not ref_parquet.exists():
        return {'error': f'Reference parquet not found: {ref_parquet}. Run make train first.'}

    try:
        df = pd.read_parquet(ref_parquet)
    except Exception as exc:
        return {'error': f'Failed to read parquet: {exc}'}

    if 'split' not in df.columns:
        return {'error': "Parquet missing 'split' column. Re-run make train."}

    excluded_cols = {'split', 'ARRIVAL_DELAY', 'DELAYED'}
    feature_cols = [c for c in df.columns if c not in excluded_cols]
    feats_ref = df[df['split'] == ref_split][feature_cols].dropna()
    feats_cur = df[df['split'] == cur_split][feature_cols].dropna()

    if feats_ref.empty or feats_cur.empty:
        return {'error': f'Empty split: ref={len(feats_ref)} rows, cur={len(feats_cur)} rows.'}

    result = run_drift_report(feats_ref, feats_cur, warning_threshold)

    from src.monitoring.metrics import DRIFT_SHARE

    DRIFT_SHARE.set(result['drift_share'])

    mlflow.set_tracking_uri('sqlite:///mlflow.db')
    mlflow.set_experiment('datathon-grupo-29-monitoring')
    with mlflow.start_run(run_name='drift_detection'):
        mlflow.set_tag('model_type', 'drift_monitor')
        mlflow.set_tag('framework', 'evidently')
        mlflow.set_tag('owner', 'grupo-29')
        mlflow.set_tag('phase', 'datathon-fase05')
        mlflow.log_metric('drift_share', result['drift_share'])
        mlflow.log_metric('drift_detected', int(result['drift_detected']))
        mlflow.log_param('ref_split', ref_split)
        mlflow.log_param('cur_split', cur_split)
        mlflow.log_param('warning_threshold', warning_threshold)

    return {
        **result,
        'warning_threshold': warning_threshold,
        'retrain_threshold': retrain_threshold,
    }


if __name__ == '__main__':
    import json

    logging.basicConfig(level=logging.INFO)
    print(json.dumps(detect_and_log_drift(), indent=2))
