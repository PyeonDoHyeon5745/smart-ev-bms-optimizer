"""
운전자 유형 3-클래스 분류기 (절약형 / 평균형 / 공격형)
─────────────────────────────────────────────────────────────────────
입력 5피처 (app.py 슬라이더와 일치):
  · mean_speed     : 평균 주행 속도 (km/h)
  · accel_events   : 시간당 급가속 횟수
  · brake_events   : 시간당 급제동 횟수
  · avg_soc_range  : 평균 충전 SOC 폭 (%) — 클수록 deep discharge
  · charge_freq    : 주당 충전 횟수

알고리즘: Random Forest (300 trees)
학습 데이터: 합성 (가우시안 혼합, 유형당 N=300)
strategy.py 의 C_RATE_MAP (절약형/평균형/공격형) 과 정합.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# ── 피처 정의 (app.py 입력 키와 동일) ──────────────────────────────────
FEATURES = [
    "mean_speed",
    "accel_events",
    "brake_events",
    "avg_soc_range",
    "charge_freq",
]

FEATURE_NAMES_KR = {
    "mean_speed":    "평균 주행 속도",
    "accel_events":  "시간당 급가속",
    "brake_events":  "시간당 급제동",
    "avg_soc_range": "충전 SOC 폭",
    "charge_freq":   "주당 충전 횟수",
}

# ── 유형 정의 (strategy.py 와 동일 키) ─────────────────────────────────
DRIVER_TYPES = {
    0: "절약형",
    1: "평균형",
    2: "공격형",
}

DRIVER_TYPE_INFO = {
    "절약형": {
        "emoji": "🟢",
        "color": "green",
        "description": "배터리 친화적 습관으로 수명을 최대화하고 있습니다",
        "key_habits": ["완속 충전 위주", "30~80% SOC 유지", "부드러운 가감속"],
        "degradation_rate": 0.6,
    },
    "평균형": {
        "emoji": "🟡",
        "color": "yellow",
        "description": "일반적인 사용 패턴으로 평균적인 배터리 수명이 예상됩니다",
        "key_habits": ["혼합 충전 패턴", "20~80% SOC", "일상 주행"],
        "degradation_rate": 1.0,
    },
    "공격형": {
        "emoji": "🔴",
        "color": "red",
        "description": "급가속·급제동·깊은 방전이 많아 배터리 열화가 빠릅니다",
        "key_habits": ["급가속/급제동 빈번", "10~100% 깊은 방전", "급속충전 다용"],
        "degradation_rate": 1.8,
    },
}


# ─────────────────────────────────────────────────────────────────────
# 합성 운전자 데이터
# ─────────────────────────────────────────────────────────────────────
def generate_synthetic_data(n_per_type: int = 300, seed: int = 42) -> pd.DataFrame:
    """
    각 유형별 가우시안 분포에서 N명 샘플링.
    분포 가중치는 BMS 도메인 직관 + strategy.py degradation_rate 정합.
    """
    rng = np.random.default_rng(seed)
    records = []

    # (label, name, ranges) — ranges = (mean_speed, accel, brake, soc_range, charge_freq)
    # 각 항목은 (mean, std)
    configs = [
        (0, "절약형", {
            "mean_speed":    (55, 8),
            "accel_events":  (3, 2),
            "brake_events":  (2, 1.5),
            "avg_soc_range": (55, 10),
            "charge_freq":   (2, 1),
        }),
        (1, "평균형", {
            "mean_speed":    (75, 10),
            "accel_events":  (10, 3),
            "brake_events":  (8, 3),
            "avg_soc_range": (65, 10),
            "charge_freq":   (4, 1),
        }),
        (2, "공격형", {
            "mean_speed":    (110, 12),
            "accel_events":  (20, 4),
            "brake_events":  (16, 4),
            "avg_soc_range": (85, 8),
            "charge_freq":   (6, 1),
        }),
    ]

    for label, name, dists in configs:
        n = n_per_type
        d = {
            "mean_speed":    rng.normal(*dists["mean_speed"], n),
            "accel_events":  rng.normal(*dists["accel_events"], n),
            "brake_events":  rng.normal(*dists["brake_events"], n),
            "avg_soc_range": rng.normal(*dists["avg_soc_range"], n),
            "charge_freq":   rng.normal(*dists["charge_freq"], n),
            "driver_type":   [name] * n,
            "true_label":    [label] * n,
        }
        records.append(pd.DataFrame(d))

    df = pd.concat(records, ignore_index=True)

    # 슬라이더 범위로 클리핑 (app.py UI 범위와 일치)
    df["mean_speed"]    = df["mean_speed"].clip(40, 140)
    df["accel_events"]  = df["accel_events"].clip(0, 30)
    df["brake_events"]  = df["brake_events"].clip(0, 25)
    df["avg_soc_range"] = df["avg_soc_range"].clip(40, 100)
    df["charge_freq"]   = df["charge_freq"].clip(1, 7).round().astype(int)

    return df


# ─────────────────────────────────────────────────────────────────────
# RF 학습/예측
# ─────────────────────────────────────────────────────────────────────
def train_rf_classifier(df: pd.DataFrame, save_dir: str = "models/saved"):
    X = df[FEATURES].values
    y = df["true_label"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    joblib.dump(rf, f"{save_dir}/rf_model.pkl")
    joblib.dump(scaler, f"{save_dir}/scaler.pkl")

    importances = dict(zip(FEATURES, rf.feature_importances_.tolist()))
    return rf, scaler, acc, cm, importances


def predict_driver_type(features: dict, save_dir: str = "models/saved") -> str:
    rf = joblib.load(f"{save_dir}/rf_model.pkl")
    scaler = joblib.load(f"{save_dir}/scaler.pkl")

    X = np.array([[float(features[f]) for f in FEATURES]])
    X_scaled = scaler.transform(X)
    label = int(rf.predict(X_scaled)[0])
    return DRIVER_TYPES[label]


def predict_driver_type_proba(features: dict, save_dir: str = "models/saved") -> dict:
    rf = joblib.load(f"{save_dir}/rf_model.pkl")
    scaler = joblib.load(f"{save_dir}/scaler.pkl")

    X = np.array([[float(features[f]) for f in FEATURES]])
    X_scaled = scaler.transform(X)
    proba = rf.predict_proba(X_scaled)[0]
    classes = rf.classes_
    return {DRIVER_TYPES[int(c)]: float(p) for c, p in zip(classes, proba)}
