"""CDE만 단독 학습 + 기존 결과와 비교"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from src.soh_cde import train_cde, evaluate_cde

SAVE_DIR = "models/saved"

train_df = pd.read_parquet(f"{SAVE_DIR}/train_df.parquet")
test_df  = pd.read_parquet(f"{SAVE_DIR}/test_df.parquet")
print(f"학습: {train_df['battery_id'].nunique()}대 ({len(train_df):,}세션)")
print(f"테스트: {test_df['battery_id'].nunique()}대 ({len(test_df):,}세션)\n")

print("=" * 50)
print("  Neural CDE 학습")
print("=" * 50)
train_cde(train_df, save_dir=SAVE_DIR,
          epochs=100, batch_size=64, lr=1e-4, patience=20,
          lambda_pinn=0.05, lambda_aux=0.1)

cde_m = evaluate_cde(test_df, save_dir=SAVE_DIR)
print(f"\n  [Neural CDE] RMSE={cde_m['rmse']:.3f}%  MAE={cde_m['mae']:.3f}%  R²={cde_m['r2']:.4f}\n")

# 기존 결과 불러와서 4-way 비교
with open(f"{SAVE_DIR}/comparison.json", encoding="utf-8") as f:
    prev = json.load(f)

rows = prev["models"] + [{"Model": "Neural CDE", "R²": cde_m["r2"],
                           "RMSE(%)": cde_m["rmse"], "MAE(%)": cde_m["mae"]}]
winner = max(rows, key=lambda r: r["R²"])["Model"]

print("=" * 50)
print("  4-way 최종 비교")
print("=" * 50)
print(f"  {'Model':<15} {'R²':>8}  {'RMSE(%)':>10}  {'MAE(%)':>10}")
print(f"  {'-'*48}")
for r in rows:
    mark = " ★" if r["Model"] == winner else ""
    print(f"  {r['Model']:<15} {r['R²']:>8.4f}  {r['RMSE(%)']:>10.3f}  {r['MAE(%)']:>10.3f}{mark}")
print(f"\n  승자: {winner}")

with open(f"{SAVE_DIR}/comparison.json", "w", encoding="utf-8") as f:
    json.dump({"models": rows, "winner": winner}, f, ensure_ascii=False, indent=2)
print(f"  comparison.json 업데이트 완료")
