from __future__ import annotations

import logging
import os
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


def run_drift_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    warning_threshold: float = 0.35,
    top_n: int = 10,
) -> dict:
    """Detecta drift de features usando Evidently DataDriftPreset.

    Returns:
        Dict com drift_detected, drift_share, per_feature (p-values),
        top_drifted_features e thresholds.
    """
    from evidently import Report
    from evidently.presets import DataDriftPreset

    report = Report(metrics=[DataDriftPreset()])
    run_result = report.run(reference_data=reference_df, current_data=current_df)
    result_dict = run_result.dict()

    drift_share = 0.0
    per_feature: dict[str, float] = {}

    for metric in result_dict.get('metrics', []):
        metric_name = metric.get('metric_name', '')
        if 'ValueDrift' in metric_name:
            column_name = metric.get('config', {}).get('column')
            if column_name:
                p_value = float(metric.get('value', 1.0))
                per_feature[column_name] = p_value

    # Bonferroni-corrected threshold: α/n_features reduces false positives
    # from simultaneous testing (with 11 features at α=0.05, expected ~0.55
    # false positives but variance allows 2-3 by chance).
    n = len(per_feature) or 1
    bonferroni_alpha = 0.05 / n
    n_drifted = sum(1 for p in per_feature.values() if p < bonferroni_alpha)
    drift_share = n_drifted / n

    drift_detected = drift_share > warning_threshold

    top_drifted = sorted(per_feature.items(), key=lambda x: x[1])[:top_n]

    logger.info('Drift report: share=%.2f detected=%s (bonferroni_alpha=%.4f)', drift_share, drift_detected, bonferroni_alpha)

    return {
        'drift_detected': drift_detected,
        'drift_share': drift_share,
        'per_feature': per_feature,
        'top_drifted_features': [{'feature': f, 'p_value': round(p, 4)} for f, p in top_drifted],
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

    cfg_excluded: list[str] = drift_cfg.get('excluded_columns', [])
    max_sample: int = drift_cfg.get('max_sample_size', 5000)

    # Rolling-window features always diverge between temporal splits — not production signals.
    _ROLLING_PREFIXES = (
        'te_',
        'tail_',
        'origin_dep_',
        'origin_weather_',
        'origin_system_',
        'origin_late_',
        'origin_hour_',
        'dest_day_',
        'origin_day_',
        'route_dep_',
        'route_delayed_',
    )

    def _is_rolling(col: str) -> bool:
        return col.endswith(('_w3', '_w5', '_w7', '_w10', '_w14', '_w30', '_w60', '_w90', '_w180')) or any(
            col.startswith(p) for p in _ROLLING_PREFIXES
        )

    excluded_cols = {'split', 'ARRIVAL_DELAY', 'DELAYED'} | set(cfg_excluded)
    feature_cols = [c for c in df.columns if c not in excluded_cols and not _is_rolling(c)]
    feats_ref = df[df['split'] == ref_split][feature_cols].dropna()
    feats_cur = df[df['split'] == cur_split][feature_cols].dropna()

    strategy = f'{ref_split}_vs_{cur_split}'
    if feats_ref.empty:
        logger.warning(
            'Split "%s" not found in parquet; falling back to random half-split of "%s"',
            ref_split,
            cur_split,
        )
        shuffled = feats_cur.sample(frac=1, random_state=42)
        mid = len(shuffled) // 2
        feats_ref = shuffled.iloc[:mid]
        feats_cur = shuffled.iloc[mid:]
        strategy = f'{cur_split}_random_half_split'

    if feats_ref.empty or feats_cur.empty:
        return {'error': f'Empty split: ref={len(feats_ref)} rows, cur={len(feats_cur)} rows.'}

    # Subsample to avoid KS-test rejecting H0 on trivially small differences at large n.
    if len(feats_ref) > max_sample:
        feats_ref = feats_ref.sample(max_sample, random_state=42)
    if len(feats_cur) > max_sample:
        feats_cur = feats_cur.sample(max_sample, random_state=42)

    result = run_drift_report(feats_ref, feats_cur, warning_threshold)

    from src.monitoring.metrics import DRIFT_SHARE

    DRIFT_SHARE.set(result['drift_share'])

    mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'sqlite:///mlflow.db'))
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
        mlflow.log_param('strategy', strategy)
        mlflow.log_param('warning_threshold', warning_threshold)
        for entry in result.get('top_drifted_features', []):
            mlflow.log_metric(f'drift_pval_{entry["feature"]}', entry['p_value'])

    return {
        **result,
        'strategy': strategy,
        'warning_threshold': warning_threshold,
        'retrain_threshold': retrain_threshold,
    }


if __name__ == '__main__':
    import json

    logging.basicConfig(level=logging.INFO)
    print(json.dumps(detect_and_log_drift(), indent=2))
