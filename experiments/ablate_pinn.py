"""
PINN loss 가중치 ablation
─────────────────────────────────────────────────────────────────────
lambda_pinn ∈ {0, 0.01, 0.05, 0.1, 0.5} 비교
각 값에 대해 LSTM 학습 → 테스트셋 R² / RMSE 측정
결과: models/saved/pinn_ablation.json

전제: train.py를 한 번 실행하여 train_df.parquet / test_df.parquet 이 존재해야 함.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from modeling.soh_lstm import train_model, evaluate_model

SAVE_DIR = "models/saved"
LAMBDAS = [0.0, 0.01, 0.05, 0.1, 0.5]


def main():
    train_path = Path(SAVE_DIR) / "train_df.parquet"
    test_path = Path(SAVE_DIR) / "test_df.parquet"
    if not train_path.exists() or not test_path.exists():
        print("ERROR: train_df.parquet / test_df.parquet 없음. 먼저 `python train.py` 실행 필요")
        return

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    print(f"  학습 {train_df['battery_id'].nunique()}대 / 테스트 {test_df['battery_id'].nunique()}대\n")

    results = []
    for lam in LAMBDAS:
        print("=" * 60)
        print(f"  λ_PINN = {lam}")
        print("=" * 60)
        # 별도 save_dir로 격리
        sub_dir = f"{SAVE_DIR}/pinn_lam_{lam}"
        Path(sub_dir).mkdir(parents=True, exist_ok=True)

        train_model(
            train_df, model_type="lstm", save_dir=sub_dir,
            epochs=80, batch_size=64, lr=5e-4, patience=15,
            lambda_pinn=lam, lambda_aux=0.1, verbose=False,
        )
        m = evaluate_model(test_df, model_type="lstm", save_dir=sub_dir)
        print(f"  R²={m['r2']:.4f}  RMSE={m['rmse']:.3f}%  MAE={m['mae']:.3f}%\n")

        results.append({
            "lambda_pinn": lam,
            "R²": m["r2"],
            "RMSE(%)": m["rmse"],
            "MAE(%)": m["mae"],
        })

    best = max(results, key=lambda r: r["R²"])
    print("=" * 60)
    print("  PINN λ Ablation Summary")
    print("=" * 60)
    print(f"  {'λ':<8} {'R²':>8} {'RMSE(%)':>10} {'MAE(%)':>10}")
    for r in results:
        marker = " ★" if r["lambda_pinn"] == best["lambda_pinn"] else ""
        print(f"  {r['lambda_pinn']:<8} {r['R²']:>8.4f} {r['RMSE(%)']:>10.3f} {r['MAE(%)']:>10.3f}{marker}")
    print(f"\n  최적 λ = {best['lambda_pinn']} (R² = {best['R²']:.4f})")

    with open(f"{SAVE_DIR}/pinn_ablation.json", "w", encoding="utf-8") as f:
        json.dump({"results": results, "best_lambda": best["lambda_pinn"]}, f,
                  ensure_ascii=False, indent=2)
    print(f"  저장: {SAVE_DIR}/pinn_ablation.json")


if __name__ == "__main__":
    main()
