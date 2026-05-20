"""
EVBattery NCM 실차 데이터 로더 (v3.5 — 병렬 로딩 + 캐시)
─────────────────────────────────────────────────────────────────────
출처: Figshare EVBattery dataset

성능:
  · ThreadPoolExecutor (16 workers) 병렬 torch.load
  · 첫 로딩 후 raw_sessions.parquet 캐시 → 다음 실행 즉시
  · cache 무효화: --force-reload 또는 파일 삭제
"""

import os
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

_BASE = Path(__file__).parent.parent / "data" / "evbattery"
_DATASETS = ["battery_dataset1", "battery_dataset2", "battery_dataset3"]
_CACHE = Path(__file__).parent.parent / "models" / "saved" / "raw_sessions.parquet"

# 물리 상수
PEUKERT_N = 0.621
_ARRHENIUS_EA = 31_000.0
_R_GAS = 8.314
_T_REF_K = 298.15

N_WORKERS = 16


def _load_one(args):
    """단일 파일 로드 (병렬 worker에서 호출)"""
    full_path = args
    try:
        d = torch.load(full_path, map_location="cpu", weights_only=False)
        arr, meta = d
        cap = float(meta["capacity"])
        if cap <= 0:
            return None
        return {
            "car":          int(meta["car"]),
            "mileage":      float(meta["mileage"]),
            "capacity":     cap,
            "voltage_mean": float(arr[:, 0].mean()),
            "voltage_std":  float(arr[:, 0].std()),
            "current_mean": float(np.abs(arr[:, 1]).mean()),
            "temp_mean":    float(arr[:, 2].mean()),
        }
    except Exception:
        return None


def _load_all_sessions_raw(force_reload: bool = False) -> pd.DataFrame:
    """
    모든 pkl을 병렬 로딩 → flat DataFrame.
    parquet 캐시 사용. 차량별 그룹화는 후처리 단계.
    """
    if not force_reload and _CACHE.exists():
        print(f"  캐시 사용: {_CACHE}", flush=True)
        return pd.read_parquet(_CACHE)

    all_files = []
    for ds in _DATASETS:
        pkl_dir = _BASE / ds / "data"
        if not pkl_dir.exists():
            continue
        files = os.listdir(pkl_dir)
        all_files.extend([str(pkl_dir / f) for f in files])
        print(f"  [{ds}] {len(files):,} files", flush=True)

    print(f"  총 {len(all_files):,} files, {N_WORKERS} workers 병렬 로딩 시작", flush=True)

    records = []
    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        for rec in tqdm(ex.map(_load_one, all_files, chunksize=64),
                        total=len(all_files), unit="file", mininterval=3.0):
            if rec is not None:
                records.append(rec)

    print(f"  → {len(records):,} valid sessions", flush=True)
    df = pd.DataFrame.from_records(records)

    _CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(_CACHE)
    print(f"  캐시 저장: {_CACHE}", flush=True)
    return df


def build_ev_dataframe(min_sessions: int = 30, force_reload: bool = False) -> pd.DataFrame:
    raw = _load_all_sessions_raw(force_reload=force_reload)

    records = []
    for car_id, group in raw.groupby("car"):
        if len(group) < min_sessions:
            continue

        df = group.sort_values("mileage").reset_index(drop=True)

        initial_capacity = float(df["capacity"].quantile(0.95))
        if initial_capacity <= 0:
            continue

        df["battery_id"] = f"EV_{car_id}"
        df["cycle"] = np.arange(1, len(df) + 1)
        df["initial_capacity"] = initial_capacity
        df["soh"] = (df["capacity"] / initial_capacity * 100.0).clip(0, 100)

        first_soh = float(df["soh"].iloc[0])
        if first_soh <= 0:
            continue
        df["soh_norm"] = (df["soh"] / first_soh * 100.0).clip(0, 100)

        df["capacity_fade_rate"] = df["capacity"].diff(5).fillna(0).abs() / 5

        n = len(df)
        df["cycle_norm"] = (df["cycle"] - 1) / (n - 1 + 1e-8)

        # 물리 피처
        df["c_rate"] = df["current_mean"] / (initial_capacity + 1e-8)
        df["thermal_stress"] = (df["temp_mean"] - 25.0).clip(lower=0.0)
        peukert_term = np.power(np.clip(df["c_rate"].values, 1e-4, None), PEUKERT_N)
        T_k = df["temp_mean"].values + 273.15
        arrhenius_term = np.exp(_ARRHENIUS_EA / _R_GAS * (1.0 / _T_REF_K - 1.0 / T_k))
        df["degradation_index"] = peukert_term * arrhenius_term

        # v3.5: 누적 + rolling
        df["cum_degradation"] = df["degradation_index"].cumsum()
        cd_max = df["cum_degradation"].iloc[-1]
        if cd_max > 0:
            df["cum_degradation"] = df["cum_degradation"] / cd_max

        df["temp_roll_5"] = df["temp_mean"].rolling(5, min_periods=1).mean()
        df["c_rate_roll_5"] = df["c_rate"].rolling(5, min_periods=1).mean()
        df["deg_roll_5"] = df["degradation_index"].rolling(5, min_periods=1).mean()

        records.append(df[[
            "battery_id", "cycle", "capacity", "soh", "soh_norm", "initial_capacity",
            "voltage_mean", "voltage_std", "current_mean", "temp_mean",
            "capacity_fade_rate", "cycle_norm",
            "c_rate", "thermal_stress", "degradation_index",
            "cum_degradation",
            "temp_roll_5", "c_rate_roll_5", "deg_roll_5",
        ]])

    if not records:
        raise RuntimeError("EVBattery 데이터 로딩 실패")
    out = pd.concat(records, ignore_index=True)
    print(f"  → {out['battery_id'].nunique()} cars, {len(out):,} sessions (after filter)", flush=True)
    return out


def load_ev_train_test(test_ratio: float = 0.2, min_sessions: int = 30, seed: int = 42,
                       force_reload: bool = False):
    df = build_ev_dataframe(min_sessions=min_sessions, force_reload=force_reload)

    cars = np.array(df["battery_id"].unique(), dtype=str)
    rng = np.random.default_rng(seed)
    rng.shuffle(cars)

    n_test = max(1, int(len(cars) * test_ratio))
    test_cars = set(cars[:n_test])
    train_cars = set(cars[n_test:])

    train_df = df[df["battery_id"].isin(train_cars)].reset_index(drop=True)
    test_df = df[df["battery_id"].isin(test_cars)].reset_index(drop=True)
    return train_df, test_df


def kfold_car_splits(df: pd.DataFrame, n_splits: int = 5, seed: int = 42):
    cars = df["battery_id"].unique().copy()
    rng = np.random.default_rng(seed)
    rng.shuffle(cars)
    folds = np.array_split(cars, n_splits)
    for k in range(n_splits):
        test_cars = set(folds[k])
        train_cars = set(np.concatenate([folds[i] for i in range(n_splits) if i != k]))
        train_df = df[df["battery_id"].isin(train_cars)].reset_index(drop=True)
        test_df = df[df["battery_id"].isin(test_cars)].reset_index(drop=True)
        yield k, train_df, test_df
