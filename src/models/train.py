from __future__ import annotations

import json
import os
import shutil
from datetime import datetime
from pathlib import Path

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('TF_XLA_FLAGS', '--tf_xla_auto_jit=0')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')

import joblib
import mlflow
import mlflow.tensorflow
import numpy as np
import pandas as pd
import yaml
import yfinance as yf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import RobustScaler
from tensorflow import keras

from src.features.feature_engineering import (
    FEATURE_COLS,
    build_features,
    create_sequences,
    temporal_split,
)
from src.models.baseline import NaiveBaseline, SMABaseline, build_lstm_model

CONFIG_PATH = Path('configs/model_config.yaml')


def _load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, encoding='utf-8') as f:
        return yaml.safe_load(f)


def _download_data(symbol: str, start: str, end: str):
    df = yf.download(symbol, start=start, end=end)
    assert df is not None, f'yfinance returned None for {symbol}'
    df = df.dropna()
    mask = (df['Close'] > 0) & (df['Open'] > 0) & (df['High'] > 0) & (df['Low'] > 0)
    if isinstance(mask, pd.DataFrame):
        mask = mask.iloc[:, 0]
    return df[mask]


def _metrics_price(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape = float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-9))) * 100)
    return mae, rmse, mape


def _dir_accuracy(true_price: np.ndarray, pred_price: np.ndarray, close_t: np.ndarray) -> float:
    return float((np.sign(true_price - close_t) == np.sign(pred_price - close_t)).mean() * 100)


def train(config_path: Path = CONFIG_PATH) -> dict:
    cfg = _load_config(config_path)

    data_cfg = cfg['data']
    split_cfg = cfg['splits']
    model_cfg = cfg['model']
    train_cfg = cfg['training']
    cb_cfg = cfg['callbacks']
    paths_cfg = cfg['paths']

    artifacts_dir = Path(paths_cfg['artifacts_dir'])
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    Path(paths_cfg['data_dir']).mkdir(parents=True, exist_ok=True)

    import tensorflow as tf
    tf.random.set_seed(train_cfg['seed'])
    np.random.seed(train_cfg['seed'])

    # 1. Download
    symbol = data_cfg['symbol']
    print(f'Downloading {symbol}...')
    df = _download_data(symbol, data_cfg['start_date'], data_cfg['end_date'])

    # 2. Features
    feats = build_features(df, horizon=model_cfg['horizon'])
    X_raw = feats[FEATURE_COLS].values
    y_raw = feats[['y_delta_h']].values
    close_t_raw = feats[['close_t']].values
    close_th_raw = feats[['close_t_h']].values
    dates = feats.index

    # 3. Temporal split
    train_end, val_end = temporal_split(
        pd.DatetimeIndex(dates), split_cfg['val_ratio'], split_cfg['test_ratio']
    )
    n_train = (dates <= train_end).sum()

    # 4. Scale (fit only on train)
    scaler_X = RobustScaler()
    scaler_y = RobustScaler()
    X_scaled = np.vstack([
        scaler_X.fit_transform(X_raw[:n_train]),
        scaler_X.transform(X_raw[n_train:]),
    ])
    y_scaled = np.vstack([
        scaler_y.fit_transform(y_raw[:n_train]),
        scaler_y.transform(y_raw[n_train:]),
    ])

    # 5. Sequences
    lookback = model_cfg['lookback']
    Xw, yw, ct_w, cth_w, dt_w = create_sequences(
        X_scaled, y_scaled, close_t_raw, close_th_raw, dates.values, lookback
    )
    train_m = dt_w <= train_end
    val_m = (dt_w > train_end) & (dt_w <= val_end)
    test_m = dt_w > val_end

    X_train, y_train = Xw[train_m], yw[train_m]
    X_val, y_val = Xw[val_m], yw[val_m]
    X_test = Xw[test_m]
    ct_test = ct_w[test_m]
    cth_test = cth_w[test_m]

    print(f'Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}')

    # 6. MLflow run
    mlflow.set_tracking_uri('sqlite:///mlflow.db')
    mlflow.set_experiment('datathon-grupo-29')
    with mlflow.start_run():
        mlflow.log_params({
            'symbol': symbol,
            'lookback': lookback,
            'horizon': model_cfg['horizon'],
            'epochs': train_cfg['epochs'],
            'batch_size': train_cfg['batch_size'],
            'learning_rate': train_cfg['learning_rate'],
        })

        model = build_lstm_model(
            lookback=lookback,
            n_features=len(FEATURE_COLS),
            lstm_units=model_cfg['lstm_units'],
            dense_units=model_cfg['dense_units'],
            dropout=model_cfg['dropout'],
            recurrent_dropout=model_cfg['recurrent_dropout'],
            learning_rate=train_cfg['learning_rate'],
            clipnorm=train_cfg['clipnorm'],
            huber_delta=train_cfg['huber_delta'],
        )
        model.summary()

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=cb_cfg['early_stopping_patience'], restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=cb_cfg['reduce_lr_factor'],
                patience=cb_cfg['reduce_lr_patience'],
                min_lr=cb_cfg['reduce_lr_min'],
            ),
            keras.callbacks.ModelCheckpoint(
                str(artifacts_dir / 'best_model.keras'), monitor='val_loss', save_best_only=True
            ),
        ]

        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=train_cfg['epochs'],
            batch_size=train_cfg['batch_size'],
            callbacks=callbacks,
            verbose=1,
        )

        for epoch, (tl, vl) in enumerate(zip(history.history['loss'], history.history['val_loss'])):
            mlflow.log_metrics({'train_loss': tl, 'val_loss': vl}, step=epoch)

        # 7. Evaluate
        pred_delta = scaler_y.inverse_transform(model.predict(X_test, verbose=0))
        pred_price = ct_test + pred_delta
        true_price = cth_test

        mae, rmse, mape = _metrics_price(true_price, pred_price)
        dir_acc = _dir_accuracy(true_price, pred_price, ct_test)
        horizon = model_cfg['horizon']

        print(f'\n=== LSTM D+{horizon} ===')
        print(f'MAE: {mae:.4f} | RMSE: {rmse:.4f} | MAPE: {mape:.2f}% | DirAcc: {dir_acc:.2f}%')

        mlflow.log_metrics({
            'test_mae': mae, 'test_rmse': rmse,
            'test_mape': mape, 'test_dir_accuracy': dir_acc,
        })

        # 8. Baselines
        naive = NaiveBaseline().fit(X_train, y_train)
        naive_m = naive.evaluate(true_price, ct_test)

        sma = SMABaseline(window=lookback).fit(X_train, y_train)
        sma_pred_all = sma.predict(feats['close'].values)
        sma_pred_test = sma_pred_all[feats.index.get_indexer(dt_w[test_m])]
        sma_pred_test = np.where(np.isnan(sma_pred_test), ct_test, sma_pred_test)
        sma_mae, sma_rmse, sma_mape = _metrics_price(true_price, sma_pred_test)

        print(f'Naive  -> MAE: {naive_m["mae"]:.4f} | MAPE: {naive_m["mape"]:.2f}%')
        print(f'SMA{lookback}  -> MAE: {sma_mae:.4f} | MAPE: {sma_mape:.2f}%')

        # 9. Save artifacts
        model.save(str(artifacts_dir / 'final_model.keras'))
        joblib.dump(scaler_X, artifacts_dir / 'scaler_X.joblib')
        joblib.dump(scaler_y, artifacts_dir / 'scaler_y.joblib')

        metadata = {
            'symbol': symbol,
            'start_date': data_cfg['start_date'],
            'end_date': data_cfg['end_date'],
            'lookback': lookback,
            'horizon_days': horizon,
            'features': FEATURE_COLS,
            'target': f'delta_close = close_(t+{horizon}) - close_t',
            'trained_at': datetime.now().isoformat(),
            'splits': {
                'train_rows': int(X_train.shape[0]),
                'val_rows': int(X_val.shape[0]),
                'test_rows': int(X_test.shape[0]),
            },
        }
        metrics_out = {
            'model': {
                'mae_price': mae, 'rmse_price': rmse,
                'mape_price_pct': mape, 'directional_accuracy_pct': dir_acc,
            },
            'baselines': {
                'naive': naive_m,
                f'sma_{lookback}': {'mae': sma_mae, 'rmse': sma_rmse, 'mape': sma_mape},
            },
        }
        (artifacts_dir / 'metadata.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
        (artifacts_dir / 'metrics.json').write_text(json.dumps(metrics_out, indent=2, ensure_ascii=False))

        mlflow.log_artifacts(str(artifacts_dir))

        # 10. Export ONNX
        _export_onnx(model, artifacts_dir)

        print(f'\nArtifacts saved in: {artifacts_dir.resolve()}')

    return metrics_out


def _export_onnx(model, artifacts_dir: Path) -> None:
    saved_model_dir = str(artifacts_dir / 'temp_saved_model')
    onnx_path = str(artifacts_dir / 'final_model.onnx')
    try:
        model.export(saved_model_dir)
        cmd = f'python -m tf2onnx.convert --saved-model {saved_model_dir} --output {onnx_path} --opset 13'
        if os.system(cmd) == 0:
            print(f'ONNX model saved: {onnx_path}')
        else:
            print('ONNX conversion failed — Keras model still available.')
    finally:
        if os.path.exists(saved_model_dir):
            shutil.rmtree(saved_model_dir)


if __name__ == '__main__':
    train()
