"""
XGBoost Baseline (v3.5 — 잔차 학습 + 13피처)
─────────────────────────────────────────────────────────────────────
LSTM/Transformer와 공정 비교: 동일 윈도우(30) → 통계 피처(45) + 현재 soh_norm(1)
타겟: ΔSOH (정규화 공간) — 절대 SOH 대신 잔차 학습
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from .soh_lstm import (
    WINDOW_SIZE,
    FEATURES as SEQ_FEATURES,
    _normalize_soh,
    _denormalize_soh,
    SOH_MIN,
    SOH_MAX,
)


def _window_to_features(window: np.ndarray) -> np.ndarray:
    """(WINDOW, F) → (F*5,) 통계: mean/std/min/max/slope"""
    n_steps, n_feats = window.shape
    out = np.empty(n_feats * 5, dtype=np.float32)
    x_idx = np.arange(n_steps, dtype=np.float32)
    x_mean = x_idx.mean()
    x_var = ((x_idx - x_mean) ** 2).sum()
    for j in range(n_feats):
        col = window[:, j]
        out[j * 5 + 0] = col.mean()
        out[j * 5 + 1] = col.std()
        out[j * 5 + 2] = col.min()
        out[j * 5 + 3] = col.max()
        cov = ((x_idx - x_mean) * (col - col.mean())).sum()
        out[j * 5 + 4] = cov / x_var if x_var > 0 else 0.0
    return out


def make_xgb_features(df: pd.DataFrame, feat_scaler=None):
    """
    Returns:
      X: (N, F*5 + 1) — 통계 + 직전 soh_norm
      y_delta: (N,) — ΔSOH 잔차 (정규화 공간)
      y_abs_norm: (N,) — 절대 soh_norm (재구성 용)
      prev_soh_n: (N,) — 윈도우 직전 시점의 soh_norm (재구성 anchor)
    """
    X_list, yd_list, ya_list, prev_list = [], [], [], []
    target_col = "soh_norm" if "soh_norm" in df.columns else "soh"

    for bat_id, group in df.groupby("battery_id"):
        group = group.sort_values("cycle").reset_index(drop=True)
        features = group[SEQ_FEATURES].values
        soh = group[target_col].values
        soh_n = _normalize_soh(soh)

        if feat_scaler is not None:
            features = feat_scaler.transform(features)

        delta = np.diff(soh_n, prepend=soh_n[0])

        for i in range(WINDOW_SIZE, len(group)):
            window = features[i - WINDOW_SIZE : i]
            stats = _window_to_features(window)
            prev_n = soh_n[i - 1]
            X_list.append(np.concatenate([stats, [prev_n]]))
            yd_list.append(delta[i])
            ya_list.append(soh_n[i])
            prev_list.append(prev_n)

    return (
        np.array(X_list, dtype=np.float32),
        np.array(yd_list, dtype=np.float32),
        np.array(ya_list, dtype=np.float32),
        np.array(prev_list, dtype=np.float32),
    )


def train_xgboost(
    train_df: pd.DataFrame,
    save_dir: str = "models/saved",
    n_estimators: int = 800,
    max_depth: int = 6,
    learning_rate: float = 0.03,
    early_stopping_rounds: int = 40,
    verbose: bool = True,
):
    try:
        from xgboost import XGBRegressor
    except ImportError as e:
        raise ImportError("xgboost 패키지 필요: pip install xgboost") from e

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    scaler = joblib.load(f"{save_dir}/feature_scaler_lstm.pkl")

    X, yd, _, _ = make_xgb_features(train_df, feat_scaler=scaler)

    n_val = max(int(len(X) * 0.15), 1)
    X_tr, yd_tr = X[:-n_val], yd[:-n_val]
    X_v, yd_v = X[-n_val:], yd[-n_val:]

    model = XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.01,
        reg_lambda=0.1,
        objective="reg:squarederror",
        random_state=42,
        early_stopping_rounds=early_stopping_rounds,
        verbosity=1 if verbose else 0,
        n_jobs=-1,
        tree_method="hist",
    )
    model.fit(X_tr, yd_tr, eval_set=[(X_v, yd_v)], verbose=False)
    joblib.dump(model, f"{save_dir}/xgboost_soh.pkl")
    return model


def evaluate_xgboost(test_df: pd.DataFrame, save_dir: str = "models/saved"):
    scaler = joblib.load(f"{save_dir}/feature_scaler_lstm.pkl")
    model = joblib.load(f"{save_dir}/xgboost_soh.pkl")
    X, _, y_abs_n, prev_n = make_xgb_features(test_df, feat_scaler=scaler)
    pred_delta = model.predict(X)
    pred_soh_n = np.clip(prev_n + pred_delta, 0.0, 1.0)

    y_true = _denormalize_soh(y_abs_n)
    y_pred = _denormalize_soh(pred_soh_n)

    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    mae = float(np.mean(np.abs(y_pred - y_true)))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    return {"rmse": rmse, "mae": mae, "r2": r2, "y_true": y_true, "y_pred": y_pred}
