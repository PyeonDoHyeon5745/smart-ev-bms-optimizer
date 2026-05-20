"""
전체 학습 파이프라인
  1. RF 운전자 분류기 학습 (합성 운전자 900명, 3유형)
  2. EVBattery NCM 데이터 로딩 (153대 학습 / 38대 테스트)
  3. LSTM 학습 (멀티태스크 + PINN + MC Dropout 지원)
  4. Transformer 학습 (Pre-LN + Learnable PE)
  5. XGBoost baseline 학습 (윈도우 통계 피처)
  6. Neural CDE 학습 (연속시간 경로 적분)
  7. 4-way 비교: R² / RMSE / MAE  →  models/saved/comparison.json
  8. parquet 캐시 저장 (app.py 빠른 로딩용)
"""
import json
import shutil
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from src.ev_data_loader import load_ev_train_test
from src.clustering import generate_synthetic_data, train_rf_classifier
from src.soh_lstm import train_model, evaluate_model, PEUKERT_N
from src.baselines import train_xgboost, evaluate_xgboost
from src.soh_cde import train_cde, evaluate_cde

SAVE_DIR = "models/saved"


def subsample(df, max_per_car: int = 500):
    """메모리 절약: 차량당 최대 max_per_car 세션"""
    parts = []
    for _, g in df.groupby("battery_id", sort=False):
        step = max(1, len(g) // max_per_car)
        parts.append(g.iloc[::step])
    return pd.concat(parts, ignore_index=True)


def main():
    Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)

    # ──────────────────────────────────────────────────────
    print("=" * 60)
    print("  Step 1: Random Forest 운전자 유형 분류기 (3유형)")
    print("=" * 60)
    df_synthetic = generate_synthetic_data(n_per_type=300)
    print(f"  합성 데이터: {len(df_synthetic)}명 (유형당 300명 × 3유형)")
    rf, scaler, acc, cm, importances = train_rf_classifier(df_synthetic, save_dir=SAVE_DIR)
    print(f"  Random Forest 정확도: {acc*100:.1f}%")
    print(f"  저장: {SAVE_DIR}/rf_model.pkl\n")

    # ──────────────────────────────────────────────────────
    print("=" * 60)
    print("  Step 2: EVBattery NCM 데이터 로딩")
    print(f"  Peukert n = {PEUKERT_N:.4f}")
    print("=" * 60)
    print("  EVBattery 로딩 중 (최초 실행 시 수 분 소요)...")
    t0 = time.time()
    train_df, test_df = load_ev_train_test(test_ratio=0.2, min_sessions=30, seed=42)
    print(f"  로딩 완료: {time.time()-t0:.1f}초")

    train_df = subsample(train_df, max_per_car=500)
    test_df = subsample(test_df, max_per_car=500)
    train_cars = train_df["battery_id"].nunique()
    test_cars = test_df["battery_id"].nunique()
    print(f"  학습: {train_cars}대 ({len(train_df):,}세션)")
    print(f"  테스트: {test_cars}대 ({len(test_df):,}세션)\n")

    # parquet 캐시 저장 (app.py 빠른 로딩)
    train_df.to_parquet(f"{SAVE_DIR}/train_df.parquet")
    test_df.to_parquet(f"{SAVE_DIR}/test_df.parquet")
    print(f"  → parquet 캐시 저장: train_df / test_df\n")

    # ──────────────────────────────────────────────────────
    print("=" * 60)
    print("  Step 3: LSTM 학습 (MultiTask + PINN + MC Dropout)")
    print("=" * 60)
    train_model(train_df, model_type="lstm", save_dir=SAVE_DIR,
                epochs=100, batch_size=64, lr=5e-4, patience=20,
                lambda_pinn=0.05, lambda_aux=0.1)
    lstm_m = evaluate_model(test_df, model_type="lstm", save_dir=SAVE_DIR)
    print(f"  [LSTM]        RMSE={lstm_m['rmse']:.3f}%  MAE={lstm_m['mae']:.3f}%  R²={lstm_m['r2']:.4f}\n")

    # ──────────────────────────────────────────────────────
    print("=" * 60)
    print("  Step 4: Transformer 학습 (Pre-LN + Learnable PE)")
    print("=" * 60)
    train_model(train_df, model_type="transformer", save_dir=SAVE_DIR,
                epochs=100, batch_size=64, lr=5e-4, patience=20,
                lambda_pinn=0.05, lambda_aux=0.1)
    trans_m = evaluate_model(test_df, model_type="transformer", save_dir=SAVE_DIR)
    print(f"  [Transformer] RMSE={trans_m['rmse']:.3f}%  MAE={trans_m['mae']:.3f}%  R²={trans_m['r2']:.4f}\n")

    # ──────────────────────────────────────────────────────
    print("=" * 60)
    print("  Step 5: XGBoost Baseline (윈도우 통계 피처)")
    print("=" * 60)
    train_xgboost(train_df, save_dir=SAVE_DIR, n_estimators=500,
                  max_depth=6, learning_rate=0.05)
    xgb_m = evaluate_xgboost(test_df, save_dir=SAVE_DIR)
    print(f"  [XGBoost]     RMSE={xgb_m['rmse']:.3f}%  MAE={xgb_m['mae']:.3f}%  R²={xgb_m['r2']:.4f}\n")

    # ──────────────────────────────────────────────────────
    print("=" * 60)
    print("  Step 6: Neural CDE 학습 (연속시간 경로 적분)")
    print("=" * 60)
    train_cde(train_df, save_dir=SAVE_DIR,
              epochs=100, batch_size=64, lr=5e-4, patience=20,
              lambda_pinn=0.05, lambda_aux=0.1)
    cde_m = evaluate_cde(test_df, save_dir=SAVE_DIR)
    print(f"  [Neural CDE]  RMSE={cde_m['rmse']:.3f}%  MAE={cde_m['mae']:.3f}%  R²={cde_m['r2']:.4f}\n")

    # ──────────────────────────────────────────────────────
    print("=" * 60)
    print("  최종 4-way 비교")
    print("=" * 60)
    rows = [
        {"Model": "LSTM",        "R²": lstm_m["r2"],  "RMSE(%)": lstm_m["rmse"],  "MAE(%)": lstm_m["mae"]},
        {"Model": "Transformer", "R²": trans_m["r2"], "RMSE(%)": trans_m["rmse"], "MAE(%)": trans_m["mae"]},
        {"Model": "XGBoost",     "R²": xgb_m["r2"],   "RMSE(%)": xgb_m["rmse"],   "MAE(%)": xgb_m["mae"]},
        {"Model": "Neural CDE",  "R²": cde_m["r2"],   "RMSE(%)": cde_m["rmse"],   "MAE(%)": cde_m["mae"]},
    ]
    winner = max(rows, key=lambda r: r["R²"])["Model"]

    print(f"  {'Model':<15} {'R²':>8}  {'RMSE(%)':>10}  {'MAE(%)':>10}")
    print(f"  {'-'*50}")
    for r in rows:
        marker = " ★" if r["Model"] == winner else ""
        print(f"  {r['Model']:<15} {r['R²']:>8.4f}  {r['RMSE(%)']:>10.3f}  {r['MAE(%)']:>10.3f}{marker}")
    print(f"\n  → 승자: {winner}")

    # comparison.json
    with open(f"{SAVE_DIR}/comparison.json", "w", encoding="utf-8") as f:
        json.dump({"models": rows, "winner": winner}, f, ensure_ascii=False, indent=2)
    print(f"  비교 결과 저장: {SAVE_DIR}/comparison.json")

    # 기본 모델 별칭 (app.py에서 lstm_soh.pt 로드)
    # XGBoost는 신경망이 아니므로 제외, Neural CDE 지원
    best = winner.lower().replace(" ", "_")   # "neural cde" → "neural_cde"
    model_key = {
        "lstm": "lstm",
        "transformer": "transformer",
        "neural_cde": "cde",
        "xgboost": "lstm",    # XGBoost 승리 시 LSTM 폴백
    }.get(best, "lstm")
    src = Path(SAVE_DIR) / f"{model_key}_soh.pt"
    dst = Path(SAVE_DIR) / "lstm_soh.pt"
    if src.exists() and src.resolve() != dst.resolve():
        shutil.copy2(src, dst)
    sc_src = Path(SAVE_DIR) / f"feature_scaler_{model_key}.pkl"
    sc_dst = Path(SAVE_DIR) / "feature_scaler_lstm.pkl"
    if sc_src.exists() and sc_src.resolve() != sc_dst.resolve():
        shutil.copy2(sc_src, sc_dst)
    print(f"  기본 추론 모델 = {winner} (별칭 {dst.name})")

    print("\n  학습 완료. 'streamlit run app.py' 로 앱을 실행하세요.")


if __name__ == "__main__":
    main()
