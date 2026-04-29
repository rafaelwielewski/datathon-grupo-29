from __future__ import annotations

import logging
from pathlib import Path

import mlflow
import pandas as pd
import yaml
import yfinance as yf

logger = logging.getLogger(__name__)

CONFIG_PATH = Path('configs/monitoring_config.yaml')


def _load_config(path: Path = CONFIG_PATH) -> dict:
    if not path.exists():
        return {
            'drift': {
                'warning_threshold': 0.1,
                'retrain_threshold': 0.2,
                'lookback_days': 90,
                'reference_window': {'start': '2024-01-01', 'end': '2026-01-01'},
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
    """Baixa dados atuais do AAPL, detecta drift vs janela de referência e loga no MLflow.

    Returns:
        Dict com resultado do drift ou {'error': ...} em caso de falha.
    """
    from src.features.feature_engineering import FEATURE_COLS, build_features

    cfg = _load_config()
    drift_cfg = cfg['drift']
    ref_cfg = drift_cfg['reference_window']

    lookback_days = drift_cfg['lookback_days']
    reference_start = ref_cfg['start']
    reference_end = ref_cfg['end']
    warning_threshold = drift_cfg['warning_threshold']

    df_current = yf.download('AAPL', period=f'{lookback_days + 30}d', progress=False)
    if df_current is None or df_current.empty:
        return {'error': 'Failed to download current market data.'}

    df_ref = yf.download('AAPL', start=reference_start, end=reference_end, progress=False)
    if df_ref is None or df_ref.empty:
        return {'error': 'Failed to download reference market data.'}

    feats_current = build_features(df_current)[FEATURE_COLS].dropna()
    feats_ref = build_features(df_ref)[FEATURE_COLS].dropna()

    result = run_drift_report(feats_ref, feats_current, warning_threshold)

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
        mlflow.log_param('lookback_days', lookback_days)
        mlflow.log_param('reference_start', reference_start)
        mlflow.log_param('reference_end', reference_end)
        mlflow.log_param('warning_threshold', warning_threshold)

    return result


if __name__ == '__main__':
    import json

    logging.basicConfig(level=logging.INFO)
    print(json.dumps(detect_and_log_drift(), indent=2))
