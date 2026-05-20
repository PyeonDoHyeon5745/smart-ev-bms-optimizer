"""
발표용 시각화 + 일반화 검증 (v3.5)
─────────────────────────────────────────────────────────────────────
출력 (figures/):
  · scatter_lstm.png/html         예측 vs 실측 산점도
  · residual_hist.png/html        잔차 히스토그램 (편향 검증)
  · soh_band_r2.png/html          SOH 구간별 R² (95~100/90~95/80~90/<80)
  · car_curves_lstm.html          대표 차량 5대 시간순 예측
  · model_comparison.png/html     3-way Bar chart
  · cv_results.png/html           5-Fold CV R² mean±std
  · physics_calibration.html      Peukert/Arrhenius 일관성 검증
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.soh_lstm import (
    evaluate_model, FEATURES, WINDOW_SIZE,
    _denormalize_soh, _normalize_soh, _build_model, _reconstruct_soh,
    PEUKERT_N, arrhenius_multiplier, BASE_C_RATE,
)
import joblib
import torch

SAVE_DIR = "models/saved"
FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)


def fig_scatter(m):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=m["y_true"], y=m["y_pred"], mode="markers",
        marker=dict(size=4, color="#60A5FA", opacity=0.4),
    ))
    lo = float(min(m["y_true"].min(), m["y_pred"].min()))
    fig.add_trace(go.Scatter(x=[lo, 100], y=[lo, 100], mode="lines",
                             line=dict(color="#EF4444", dash="dash"), name="y=x"))
    fig.update_layout(
        title=f"LSTM 예측 vs 실측  |  R²={m['r2']:.4f}  RMSE={m['rmse']:.2f}%",
        xaxis_title="실측 SOH (%)", yaxis_title="예측 SOH (%)",
        width=700, height=600, showlegend=False,
    )
    return fig


def fig_residual(m):
    res = m["y_pred"] - m["y_true"]
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=res, nbinsx=60, marker_color="#60A5FA"))
    fig.add_vline(x=0, line_dash="dash", line_color="red")
    fig.update_layout(
        title=f"잔차 분포  |  μ={res.mean():.3f}%  σ={res.std():.3f}%  (편향 검증)",
        xaxis_title="잔차 (예측 - 실측, %)", yaxis_title="빈도",
        width=700, height=400,
    )
    return fig


def fig_band_r2(m):
    """SOH 구간별 R² — 일반화 검증의 핵심"""
    bands = [
        ("95~100%", (95, 100)),
        ("90~95%",  (90, 95)),
        ("80~90%",  (80, 90)),
        ("<80%",    (0,  80)),
    ]
    labels, r2s, counts = [], [], []
    for name, (lo, hi) in bands:
        mask = (m["y_true"] >= lo) & (m["y_true"] < hi)
        if mask.sum() < 5:
            continue
        yt, yp = m["y_true"][mask], m["y_pred"][mask]
        ss_res = np.sum((yt - yp) ** 2)
        ss_tot = np.sum((yt - yt.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        labels.append(name)
        r2s.append(r2)
        counts.append(int(mask.sum()))

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels, y=r2s,
        text=[f"R²={r:.3f}<br>n={c}" for r, c in zip(r2s, counts)],
        textposition="outside",
        marker_color=["#22C55E", "#FBBF24", "#F97316", "#EF4444"][:len(labels)],
    ))
    fig.update_layout(
        title="SOH 구간별 R² — 열화 구간 (<90%) 성능이 진짜 중요",
        xaxis_title="실측 SOH 구간", yaxis_title="R²",
        yaxis=dict(range=[0, 1.05]),
        width=700, height=450,
    )
    return fig


def fig_car_curves(test_df: pd.DataFrame, model_type: str = "lstm"):
    drift = test_df.groupby("battery_id")["soh_norm"].agg(lambda s: s.max() - s.min())
    top5 = drift.sort_values(ascending=False).head(5).index.tolist()

    scaler = joblib.load(f"{SAVE_DIR}/feature_scaler_{model_type}.pkl")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_model(model_type).to(device)
    model.load_state_dict(torch.load(f"{SAVE_DIR}/{model_type}_soh.pt", map_location=device))
    model.eval()

    fig = make_subplots(rows=len(top5), cols=1,
                        subplot_titles=[f"{cid}" for cid in top5],
                        vertical_spacing=0.05)
    for i, cid in enumerate(top5, start=1):
        g = test_df[test_df["battery_id"] == cid].sort_values("cycle").reset_index(drop=True)
        if len(g) <= WINDOW_SIZE:
            continue
        feat = scaler.transform(g[FEATURES].values)
        soh_n = _normalize_soh(g["soh_norm"].values)
        x = np.concatenate([feat, soh_n.reshape(-1, 1)], axis=1)
        seqs = np.stack([x[t - WINDOW_SIZE : t] for t in range(WINDOW_SIZE, len(g))])
        seqs_t = torch.tensor(seqs, dtype=torch.float32).to(device)
        with torch.no_grad():
            d, _ = model(seqs_t)
            soh_n_prev = seqs_t[:, -1, -1]
            pred_soh_n = _reconstruct_soh(soh_n_prev, d).cpu().numpy()
        y_pred = _denormalize_soh(pred_soh_n)
        cycles = g["cycle"].values[WINDOW_SIZE:]
        y_true = g["soh_norm"].values[WINDOW_SIZE:]

        fig.add_trace(go.Scatter(x=cycles, y=y_true, name="실측",
                                 line=dict(color="#60A5FA"), showlegend=(i == 1)), row=i, col=1)
        fig.add_trace(go.Scatter(x=cycles, y=y_pred, name="예측",
                                 line=dict(color="#F97316", dash="dash"),
                                 showlegend=(i == 1)), row=i, col=1)
    fig.update_layout(title=f"테스트 차량 5대 시간순 SOH 예측 ({model_type.upper()})",
                      height=200 * len(top5), width=900)
    return fig


def fig_three_way():
    path = Path(SAVE_DIR) / "comparison.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    models = [r["Model"] for r in data["models"]]
    r2 = [r["R²"] for r in data["models"]]
    rmse = [r["RMSE(%)"] for r in data["models"]]
    fig = make_subplots(rows=1, cols=2, subplot_titles=("R² (높을수록 좋음)", "RMSE % (낮을수록 좋음)"))
    fig.add_trace(go.Bar(x=models, y=r2, marker_color="#22C55E",
                         text=[f"{v:.4f}" for v in r2], textposition="outside"),
                  row=1, col=1)
    fig.add_trace(go.Bar(x=models, y=rmse, marker_color="#F97316",
                         text=[f"{v:.2f}" for v in rmse], textposition="outside"),
                  row=1, col=2)
    fig.update_layout(title=f"3-way 모델 비교 — 승자: {data['winner']}",
                      height=420, width=900, showlegend=False)
    return fig


def fig_cv():
    """5-Fold CV 결과 시각화"""
    path = Path(SAVE_DIR) / "cv_results_lstm.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    folds = [f"Fold {r['fold']+1}" for r in data["folds"]]
    r2 = [r["R²"] for r in data["folds"]]
    mean_r2 = data["mean_R²"]
    std_r2 = data["std_R²"]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=folds, y=r2, marker_color="#60A5FA",
                         text=[f"{v:.4f}" for v in r2], textposition="outside",
                         name="Fold R²"))
    fig.add_hline(y=mean_r2, line_dash="dash", line_color="#22C55E",
                  annotation_text=f"평균 {mean_r2:.4f} ± {std_r2:.4f}",
                  annotation_position="right")
    fig.update_layout(
        title=f"5-Fold CV — 일반화 검증 (평균 R² {mean_r2:.4f} ± {std_r2:.4f})",
        xaxis_title="Fold", yaxis_title="R²",
        yaxis=dict(range=[max(0, min(r2) - 0.1), 1.0]),
        width=700, height=450, showlegend=False,
    )
    return fig


def fig_physics_calibration(test_df: pd.DataFrame, model_type: str = "lstm"):
    """
    물리 일관성 검증:
      · 같은 윈도우에 가상 c_rate / temp 변화를 주입 → 모델 예측이 Peukert/Arrhenius 따르는지
    """
    scaler = joblib.load(f"{SAVE_DIR}/feature_scaler_{model_type}.pkl")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_model(model_type).to(device)
    model.load_state_dict(torch.load(f"{SAVE_DIR}/{model_type}_soh.pt", map_location=device))
    model.eval()

    # 대표 차량 한 대 골라서 baseline 윈도우 추출
    bat_ids = test_df["battery_id"].unique()
    g = test_df[test_df["battery_id"] == bat_ids[0]].sort_values("cycle").reset_index(drop=True)
    if len(g) < WINDOW_SIZE + 1:
        return None

    feat_raw = g[FEATURES].values[-WINDOW_SIZE - 1 : -1]
    soh_n = _normalize_soh(g["soh_norm"].values[-WINDOW_SIZE - 1 : -1])

    # Peukert sweep: c_rate 0.3~2.5 변화 → 마지막 timestep의 c_rate, degradation_index 조정
    c_idx = FEATURES.index("c_rate")
    deg_idx = FEATURES.index("degradation_index")
    t_idx = FEATURES.index("temp_mean")

    c_grid = np.linspace(0.3, 2.5, 12)
    peukert_preds = []
    for c in c_grid:
        f = feat_raw.copy()
        f[-1, c_idx] = c  # 마지막 시점만 변화
        T_k = f[-1, t_idx] + 273.15
        f[-1, deg_idx] = (max(c, 1e-4) ** PEUKERT_N) * np.exp(
            31000.0 / 8.314 * (1 / 298.15 - 1 / T_k)
        )
        f_sc = scaler.transform(f)
        x = np.concatenate([f_sc, soh_n.reshape(-1, 1)], axis=1)
        x_t = torch.tensor(x[np.newaxis, :, :], dtype=torch.float32).to(device)
        with torch.no_grad():
            d, _ = model(x_t)
            soh_n_prev = x_t[:, -1, -1]
            pred = _reconstruct_soh(soh_n_prev, d).cpu().numpy()[0]
        peukert_preds.append(_denormalize_soh(pred))

    # Arrhenius sweep: temp 0~50°C
    t_grid = np.linspace(0, 50, 12)
    arrhenius_preds = []
    for tmp in t_grid:
        f = feat_raw.copy()
        f[-1, t_idx] = tmp
        T_k = tmp + 273.15
        c = f[-1, c_idx]
        f[-1, deg_idx] = (max(c, 1e-4) ** PEUKERT_N) * np.exp(
            31000.0 / 8.314 * (1 / 298.15 - 1 / T_k)
        )
        f_sc = scaler.transform(f)
        x = np.concatenate([f_sc, soh_n.reshape(-1, 1)], axis=1)
        x_t = torch.tensor(x[np.newaxis, :, :], dtype=torch.float32).to(device)
        with torch.no_grad():
            d, _ = model(x_t)
            soh_n_prev = x_t[:, -1, -1]
            pred = _reconstruct_soh(soh_n_prev, d).cpu().numpy()[0]
        arrhenius_preds.append(_denormalize_soh(pred))

    # Peukert 이론값: SOH 변화 ∝ -c_rate^n (개념 비교 — 절대값 아닌 추세)
    peukert_theory = -np.power(c_grid, PEUKERT_N)
    arrhenius_theory = np.array([-arrhenius_multiplier(t) for t in t_grid])

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            f"Peukert 일관성 (n={PEUKERT_N:.3f})",
            "Arrhenius 일관성 (Ea=31 kJ/mol)",
        ),
    )
    # Peukert
    fig.add_trace(go.Scatter(x=c_grid, y=peukert_preds, mode="lines+markers",
                             name="모델 예측 SOH", line=dict(color="#F97316")),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=c_grid, y=(peukert_theory - peukert_theory[0]) * 5 + peukert_preds[0],
                             mode="lines", name="Peukert 추세 (정성)",
                             line=dict(color="#22C55E", dash="dash")),
                  row=1, col=1)
    # Arrhenius
    fig.add_trace(go.Scatter(x=t_grid, y=arrhenius_preds, mode="lines+markers",
                             name="모델 예측 SOH", line=dict(color="#F97316"),
                             showlegend=False),
                  row=1, col=2)
    fig.add_trace(go.Scatter(x=t_grid, y=(arrhenius_theory - arrhenius_theory[0]) * 5 + arrhenius_preds[0],
                             mode="lines", name="Arrhenius 추세 (정성)",
                             line=dict(color="#22C55E", dash="dash"),
                             showlegend=False),
                  row=1, col=2)

    fig.update_xaxes(title_text="C-rate", row=1, col=1)
    fig.update_yaxes(title_text="예측 SOH (%)", row=1, col=1)
    fig.update_xaxes(title_text="온도 (°C)", row=1, col=2)
    fig.update_yaxes(title_text="예측 SOH (%)", row=1, col=2)
    fig.update_layout(
        title="모델의 물리 일관성 검증 (학습 데이터 외 가상 입력)",
        width=1000, height=450,
    )
    return fig


def main():
    test_path = Path(SAVE_DIR) / "test_df.parquet"
    if not test_path.exists():
        print("ERROR: test_df.parquet 없음. `python train.py` 먼저 실행")
        return
    test_df = pd.read_parquet(test_path)

    print("LSTM 평가 중...")
    m = evaluate_model(test_df, model_type="lstm", save_dir=SAVE_DIR)
    print(f"  R²={m['r2']:.4f}  RMSE={m['rmse']:.2f}%\n")

    figs = [
        ("scatter_lstm",        fig_scatter(m)),
        ("residual_hist",       fig_residual(m)),
        ("soh_band_r2",         fig_band_r2(m)),
        ("car_curves_lstm",     fig_car_curves(test_df, "lstm")),
        ("physics_calibration", fig_physics_calibration(test_df, "lstm")),
    ]
    cmp_fig = fig_three_way()
    if cmp_fig is not None:
        figs.append(("model_comparison", cmp_fig))
    cv_fig = fig_cv()
    if cv_fig is not None:
        figs.append(("cv_results", cv_fig))

    for name, fig in figs:
        if fig is None:
            continue
        png_path = FIG_DIR / f"{name}.png"
        html_path = FIG_DIR / f"{name}.html"
        try:
            fig.write_image(str(png_path))
            print(f"  saved {png_path}")
        except Exception as e:
            print(f"  PNG 실패 ({name}): {type(e).__name__}  → HTML만")
        fig.write_html(str(html_path))
        print(f"  saved {html_path}")

    print(f"\n  완료: figures/")


if __name__ == "__main__":
    main()
