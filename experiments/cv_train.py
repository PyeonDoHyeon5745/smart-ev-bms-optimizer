"""
5-Fold Cross Validation (일반화 검증)
─────────────────────────────────────────────────────────────────────
차량 단위 5-Fold:
  · 153대를 5등분 → 각 fold마다 학습/평가
  · R² mean ± std 발표 → "한 번 잘 나온 게 아님" 증명

결과:
  · models/saved/cv_results.json
  · 콘솔에 fold별 + 평균 성능

전제: train.py 한 번 실행해서 train_df.parquet 존재 (전체 학습 차량 데이터)
사용:
  python cv_train.py            # LSTM 5-fold (기본)
  python cv_train.py transformer
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from preprocessing.ev_data_loader import kfold_car_splits
from modeling.soh_lstm import train_model, evaluate_model

SAVE_DIR = "models/saved"


def main():
    model_type = sys.argv[1] if len(sys.argv) > 1 else "lstm"
    n_splits = 5

    train_path = Path(SAVE_DIR) / "train_df.parquet"
    test_path = Path(SAVE_DIR) / "test_df.parquet"
    if not train_path.exists() or not test_path.exists():
        print("ERROR: train_df.parquet / test_df.parquet 없음. `python train.py` 먼저 실행")
        return

    # 전체 데이터 = train+test (CV는 전체에서 다시 split)
    df = pd.concat([pd.read_parquet(train_path), pd.read_parquet(test_path)], ignore_index=True)
    n_cars = df["battery_id"].nunique()
    print(f"  CV 대상: 전체 {n_cars}대, {n_splits}-fold")

    fold_results = []
    for fold_idx, train_df, test_df in kfold_car_splits(df, n_splits=n_splits, seed=42):
        print("=" * 60)
        print(f"  Fold {fold_idx + 1} / {n_splits}  "
              f"(train {train_df['battery_id'].nunique()}대 / test {test_df['battery_id'].nunique()}대)")
        print("=" * 60)
        fold_dir = f"{SAVE_DIR}/cv_fold_{fold_idx}"
        Path(fold_dir).mkdir(parents=True, exist_ok=True)

        t0 = time.time()
        train_model(
            train_df, model_type=model_type, save_dir=fold_dir,
            epochs=80, batch_size=64, lr=5e-4, patience=15,
            lambda_pinn=0.05, lambda_aux=0.1, verbose=False,
        )
        m = evaluate_model(test_df, model_type=model_type, save_dir=fold_dir)
        dt = time.time() - t0
        print(f"  R²={m['r2']:.4f}  RMSE={m['rmse']:.3f}%  MAE={m['mae']:.3f}%  ({dt:.0f}s)\n")
        fold_results.append({
            "fold": fold_idx,
            "R²": m["r2"],
            "RMSE(%)": m["rmse"],
            "MAE(%)": m["mae"],
            "train_cars": int(train_df["battery_id"].nunique()),
            "test_cars": int(test_df["battery_id"].nunique()),
        })

    # 요약
    import statistics
    r2_vals = [r["R²"] for r in fold_results]
    rmse_vals = [r["RMSE(%)"] for r in fold_results]
    mae_vals = [r["MAE(%)"] for r in fold_results]

    print("=" * 60)
    print(f"  {n_splits}-Fold CV Summary ({model_type})")
    print("=" * 60)
    print(f"  {'Fold':<6} {'R²':>8} {'RMSE(%)':>10} {'MAE(%)':>10}")
    for r in fold_results:
        print(f"  {r['fold']:<6} {r['R²']:>8.4f} {r['RMSE(%)']:>10.3f} {r['MAE(%)']:>10.3f}")
    print(f"  {'-'*40}")
    print(f"  mean     {statistics.mean(r2_vals):>8.4f} {statistics.mean(rmse_vals):>10.3f} {statistics.mean(mae_vals):>10.3f}")
    print(f"  std      {statistics.stdev(r2_vals):>8.4f} {statistics.stdev(rmse_vals):>10.3f} {statistics.stdev(mae_vals):>10.3f}")

    summary = {
        "model_type": model_type,
        "n_splits": n_splits,
        "folds": fold_results,
        "mean_R²": statistics.mean(r2_vals),
        "std_R²": statistics.stdev(r2_vals),
        "mean_RMSE": statistics.mean(rmse_vals),
        "std_RMSE": statistics.stdev(rmse_vals),
        "mean_MAE": statistics.mean(mae_vals),
        "std_MAE": statistics.stdev(mae_vals),
    }
    with open(f"{SAVE_DIR}/cv_results_{model_type}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n  저장: {SAVE_DIR}/cv_results_{model_type}.json")


if __name__ == "__main__":
    main()
