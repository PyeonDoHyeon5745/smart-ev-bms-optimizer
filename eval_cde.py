import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import pandas as pd
from src.soh_cde import evaluate_cde

SAVE_DIR = "models/saved"
test_df = pd.read_parquet(f"{SAVE_DIR}/test_df.parquet")
print("CDE 평가 중...")
m = evaluate_cde(test_df, save_dir=SAVE_DIR)
print(f"Neural CDE  R2={m['r2']:.4f}  RMSE={m['rmse']:.3f}%  MAE={m['mae']:.3f}%")

with open(f"{SAVE_DIR}/comparison.json", encoding="utf-8") as f:
    prev = json.load(f)

rows = prev["models"] + [{"Model": "Neural CDE", "R²": m["r2"], "RMSE(%)": m["rmse"], "MAE(%)": m["mae"]}]
winner = max(rows, key=lambda r: r["R²"])["Model"]

with open(f"{SAVE_DIR}/comparison.json", "w", encoding="utf-8") as f:
    json.dump({"models": rows, "winner": winner}, f, ensure_ascii=False, indent=2)

print()
print(f"  {'Model':<15} {'R2':>8}  {'RMSE':>8}  {'MAE':>8}")
print(f"  {'-'*45}")
for r in rows:
    mark = " ★" if r["Model"] == winner else ""
    print(f"  {r['Model']:<15} {r['R²']:>8.4f}  {r['RMSE(%)']:>8.3f}  {r['MAE(%)']:>8.3f}{mark}")
print(f"\n승자: {winner}")
