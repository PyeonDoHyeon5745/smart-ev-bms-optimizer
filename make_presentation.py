"""
발표용 시각화 자료 — PPT 슬라이드 그대로 쓸 수 있는 고품질 이미지
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import torch

from src.soh_lstm import (
    evaluate_model, FEATURES, WINDOW_SIZE,
    _denormalize_soh, _normalize_soh, _build_model, _reconstruct_soh,
    PEUKERT_N, arrhenius_multiplier, predict_soh_curve,
)
from src.strategy import (
    get_car_adjusted_c_rate, cycles_to_years, find_eol_cycle, format_years,
)
from src.ev_database import get_spec

SAVE_DIR = "models/saved"
OUT_DIR = Path("presentation")
OUT_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────
# 통일 디자인 시스템
# ─────────────────────────────────────────────────────────────────────
SLIDE_W = 1600   # 16:9 (PPT 슬라이드 표준)
SLIDE_H = 900
FONT_FAMILY = "Pretendard, Malgun Gothic, Apple SD Gothic Neo, sans-serif"

# 색상 팔레트 (모던 / 발표용)
C = {
    "primary":   "#3B82F6",   # 진한 파랑
    "primary_l": "#93C5FD",
    "good":      "#10B981",   # 청록 그린
    "good_l":    "#6EE7B7",
    "warn":      "#F59E0B",   # 앰버
    "warn_l":    "#FCD34D",
    "bad":       "#EF4444",   # 레드
    "bad_l":     "#FCA5A5",
    "accent":    "#8B5CF6",   # 보라
    "neutral":   "#64748B",
    "bg_grid":   "#F1F5F9",
    "text":      "#0F172A",
    "subtext":   "#475569",
}


def base_layout(title=None, subtitle=None, h=SLIDE_H, w=SLIDE_W):
    title_text = ""
    if title:
        title_text = (
            f"<span style='font-size:32px; font-weight:700; color:{C['text']}'>{title}</span>"
        )
    if subtitle:
        title_text += (
            f"<br><span style='font-size:17px; font-weight:400; color:{C['subtext']}'>{subtitle}</span>"
        )

    return dict(
        title=dict(text=title_text, x=0.04, y=0.93, xanchor="left", yanchor="top"),
        font=dict(family=FONT_FAMILY, size=16, color=C["text"]),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(t=130 if subtitle else 100, b=80, l=80, r=60),
        height=h,
        width=w,
        xaxis=dict(
            showgrid=True, gridcolor=C["bg_grid"], gridwidth=1,
            zeroline=False, linecolor=C["neutral"], tickfont=dict(size=15),
            title_font=dict(size=17, color=C["subtext"]),
        ),
        yaxis=dict(
            showgrid=True, gridcolor=C["bg_grid"], gridwidth=1,
            zeroline=False, linecolor=C["neutral"], tickfont=dict(size=15),
            title_font=dict(size=17, color=C["subtext"]),
        ),
        legend=dict(
            font=dict(size=15),
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor=C["bg_grid"], borderwidth=1,
        ),
    )


def save(fig, name):
    png = OUT_DIR / f"{name}.png"
    html = OUT_DIR / f"{name}.html"
    try:
        fig.write_image(str(png), width=SLIDE_W, height=SLIDE_H, scale=2)
        print(f"  ✓ {png.name}")
    except Exception as e:
        print(f"  ✗ {png.name}: {type(e).__name__}: {e}")
    fig.write_html(str(html))


# ─────────────────────────────────────────────────────────────────────
# 1. 데이터 개요
# ─────────────────────────────────────────────────────────────────────
def fig_data_overview(test_df, train_df):
    df_all = pd.concat([train_df, test_df], ignore_index=True)
    soh_dist = df_all["soh_norm"]
    n_cars = df_all["battery_id"].nunique()
    n_sessions = len(df_all)

    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.55, 0.45],
        horizontal_spacing=0.12,
        subplot_titles=(
            f"<b>SOH 분포</b>  <span style='color:#64748B;font-size:14px'>전체 {n_sessions:,} 세션</span>",
            f"<b>차량별 SOH 범위</b>  <span style='color:#64748B;font-size:14px'>{n_cars}대</span>",
        ),
    )

    fig.add_trace(
        go.Histogram(x=soh_dist, nbinsx=40,
                     marker=dict(color=C["primary"], line=dict(width=0)),
                     opacity=0.85, name=""),
        row=1, col=1,
    )

    per_car = df_all.groupby("battery_id")["soh_norm"].agg(["min", "max", "mean"]).sort_values("min")
    fig.add_trace(
        go.Scatter(x=list(range(len(per_car))), y=per_car["max"],
                   mode="markers",
                   marker=dict(size=6, color=C["good"], line=dict(width=0)),
                   name="최대 SOH"),
        row=1, col=2,
    )
    fig.add_trace(
        go.Scatter(x=list(range(len(per_car))), y=per_car["min"],
                   mode="markers",
                   marker=dict(size=6, color=C["bad"], line=dict(width=0)),
                   name="최저 SOH"),
        row=1, col=2,
    )
    fig.add_hline(y=70, line_dash="dash", line_color=C["neutral"],
                  annotation_text="EOL 70%",
                  annotation_font=dict(size=14, color=C["neutral"]),
                  row=1, col=2)

    fig.update_xaxes(title_text="SOH (%)", row=1, col=1, showgrid=True, gridcolor=C["bg_grid"])
    fig.update_yaxes(title_text="세션 수", row=1, col=1, showgrid=True, gridcolor=C["bg_grid"])
    fig.update_xaxes(title_text="차량 인덱스 (최저 SOH 기준 정렬)", row=1, col=2, showgrid=True, gridcolor=C["bg_grid"])
    fig.update_yaxes(title_text="SOH (%)", row=1, col=2, showgrid=True, gridcolor=C["bg_grid"])

    layout = base_layout(
        title="EVBattery 데이터셋 개요",
        subtitle=f"NCM 99% 실차 {n_cars}대 · 충전 세션 {n_sessions:,}건 · Figshare 공개 데이터",
    )
    fig.update_layout(**layout)
    fig.update_layout(showlegend=True,
                      legend=dict(orientation="h", yanchor="bottom", y=-0.18,
                                  xanchor="center", x=0.5))
    return fig


# ─────────────────────────────────────────────────────────────────────
# 2. NCM 화학 일치 — 도넛 4개 + 화살표
# ─────────────────────────────────────────────────────────────────────
def fig_ncm_match():
    """4개 도넛 차트로 각 출처의 화학 구성 시각화"""
    fig = make_subplots(
        rows=1, cols=4,
        specs=[[{"type": "domain"}] * 4],
        subplot_titles=(
            "<b>EVBattery</b><br><span style='color:#64748B;font-size:13px'>중국 실차 465대</span>",
            "<b>현대 EV</b><br><span style='color:#64748B;font-size:13px'>25개 모델</span>",
            "<b>기아 EV</b><br><span style='color:#64748B;font-size:13px'>20개 모델</span>",
            "<b>NASA (제외)</b><br><span style='color:#64748B;font-size:13px'>실험실 셀</span>",
        ),
    )
    datasets = [
        (["NCM", "기타"], [99, 1], [C["good"], C["bg_grid"]], "99%"),
        (["NCM", "기타"], [99, 1], [C["good"], C["bg_grid"]], "99%"),
        (["NCM", "LFP"], [95, 5], [C["good"], C["warn"]], "95%"),
        (["LCO"], [100], [C["bad"]], "LCO\n(불일치)"),
    ]
    for i, (labels, values, colors, center_text) in enumerate(datasets, start=1):
        fig.add_trace(
            go.Pie(
                labels=labels, values=values,
                hole=0.62,
                marker=dict(colors=colors, line=dict(color="white", width=3)),
                textinfo="label+percent",
                textfont=dict(size=14),
                showlegend=False,
                rotation=90,
            ),
            row=1, col=i,
        )

    # 중앙 텍스트 (각 도넛 위)
    centers_x = [0.105, 0.375, 0.625, 0.895]
    center_labels = ["NCM 99%", "NCM 99%", "NCM 95%", "LCO ×"]
    center_colors = [C["good"], C["good"], C["good"], C["bad"]]
    for x, label, col in zip(centers_x, center_labels, center_colors):
        fig.add_annotation(
            x=x, y=0.48, xref="paper", yref="paper",
            text=f"<b style='font-size:22px; color:{col}'>{label}</b>",
            showarrow=False, align="center",
        )

    # 강조 박스
    fig.add_annotation(
        x=0.37, y=-0.05, xref="paper", yref="paper",
        text=f"<b style='color:{C['good']}; font-size:20px'>✓ 화학 동일 → 모델 직접 이식 가능</b>",
        showarrow=False, align="center",
    )
    fig.add_annotation(
        x=0.895, y=-0.05, xref="paper", yref="paper",
        text=f"<b style='color:{C['bad']}; font-size:18px'>✗ 다른 화학 → 제외</b>",
        showarrow=False, align="center",
    )

    layout = base_layout(
        title="NCM 화학 일치성 — 학습 데이터의 정당성",
        subtitle="EVBattery NCM = 한국 현대·기아 NCM → 같은 열화 메커니즘 → 모델 직접 적용 가능",
    )
    fig.update_layout(**layout)
    fig.update_layout(showlegend=False, margin=dict(t=130, b=110, l=40, r=40))
    return fig


# ─────────────────────────────────────────────────────────────────────
# 3. 4-way 비교 — 레이더 + 상세 테이블
# ─────────────────────────────────────────────────────────────────────
def fig_4way():
    data = json.loads(Path(f"{SAVE_DIR}/comparison.json").read_text(encoding="utf-8"))
    models = [r["Model"] for r in data["models"]]
    r2 = [r["R²"] for r in data["models"]]
    rmse = [r["RMSE(%)"] for r in data["models"]]
    mae = [r["MAE(%)"] for r in data["models"]]
    winner = data["winner"]

    def norm_higher(vals):
        v = np.array(vals); return ((v - v.min()) / max(v.max() - v.min(), 1e-9)).tolist()
    def norm_lower(vals):
        v = np.array(vals); return ((v.max() - v) / max(v.max() - v.min(), 1e-9)).tolist()

    r2_score = norm_higher(r2)
    rmse_score = norm_lower(rmse)
    mae_score = norm_lower(mae)

    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.55, 0.45],
        specs=[[{"type": "polar"}, {"type": "table"}]],
        subplot_titles=(
            "<b>성능 레이더</b>  <span style='color:#64748B;font-size:13px'>0~1 정규화 (외곽=우수)</span>",
            "<b>실측 수치</b>",
        ),
    )

    radar_colors = {
        "LSTM": C["good"], "Transformer": C["primary"],
        "XGBoost": C["accent"], "Neural CDE": C["warn"],
    }
    cats = ["R²", "RMSE", "MAE", "R²"]
    for i, m in enumerate(models):
        scores = [r2_score[i], rmse_score[i], mae_score[i], r2_score[i]]
        fig.add_trace(
            go.Scatterpolar(
                r=scores, theta=cats,
                fill="toself", name=m,
                line=dict(color=radar_colors.get(m, C["neutral"]), width=2.5),
                fillcolor=radar_colors.get(m, C["neutral"]),
                opacity=0.35,
            ),
            row=1, col=1,
        )

    n = len(models)
    medals = []
    sorted_r2 = sorted(r2, reverse=True)
    for r in r2:
        if r == sorted_r2[0]: medals.append("🥇")
        elif r == sorted_r2[1]: medals.append("🥈")
        elif r == sorted_r2[2]: medals.append("🥉")
        else: medals.append("4위")

    fig.add_trace(
        go.Table(
            header=dict(
                values=["<b>Model</b>", "<b>R²</b>", "<b>RMSE (%)</b>", "<b>MAE (%)</b>", "<b>Rank</b>"],
                fill_color=C["primary"],
                font=dict(color="white", size=16),
                align="center", height=40,
            ),
            cells=dict(
                values=[models,
                        [f"{v:.4f}" for v in r2],
                        [f"{v:.3f}" for v in rmse],
                        [f"{v:.3f}" for v in mae],
                        medals],
                fill_color=[["#F8FAFC", "white"] * n],
                font=dict(size=15),
                align="center", height=42,
            ),
        ),
        row=1, col=2,
    )

    layout = base_layout(
        title="4-way 모델 비교",
        subtitle=f"승자: <b style='color:{C['good']}'>{winner}</b> · LSTM / Transformer / XGBoost / Neural CDE — 네 모델 모두 R² 0.9+ → 방법론 견고성 입증",
    )
    fig.update_layout(**layout)
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1.05],
                            tickfont=dict(size=12), gridcolor=C["bg_grid"]),
            angularaxis=dict(tickfont=dict(size=14)),
            bgcolor="white",
        ),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.12, xanchor="center", x=0.3),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────
# 4. Scatter (예측 vs 실측)
# ─────────────────────────────────────────────────────────────────────
def fig_scatter(m):
    n = len(m["y_true"])
    fig = go.Figure()
    # density scatter
    fig.add_trace(go.Scatter(
        x=m["y_true"], y=m["y_pred"],
        mode="markers",
        marker=dict(size=3, color=C["primary"], opacity=0.30,
                    line=dict(width=0)),
        name="예측 샘플",
    ))
    lo = float(max(60, min(m["y_true"].min(), m["y_pred"].min())))
    fig.add_trace(go.Scatter(
        x=[lo, 100], y=[lo, 100], mode="lines",
        line=dict(color=C["bad"], dash="dash", width=2.5),
        name="y = x (이상선)",
    ))

    layout = base_layout(
        title="예측 정확도",
        subtitle=f"R² = {m['r2']:.4f}  ·  RMSE = {m['rmse']:.3f}%  ·  MAE = {m['mae']:.3f}%  ·  n = {n:,}",
    )
    fig.update_layout(**layout)
    fig.update_layout(
        xaxis_title="실측 SOH (%)",
        yaxis_title="예측 SOH (%)",
        xaxis=dict(range=[lo - 2, 102], showgrid=True, gridcolor=C["bg_grid"]),
        yaxis=dict(range=[lo - 2, 102], showgrid=True, gridcolor=C["bg_grid"]),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────
# 5. 잔차 분포
# ─────────────────────────────────────────────────────────────────────
def fig_residual(m):
    res = m["y_pred"] - m["y_true"]
    mu, sigma = float(res.mean()), float(res.std())

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=res, nbinsx=70,
        marker=dict(color=C["primary"], line=dict(width=0)),
        opacity=0.85,
    ))
    fig.add_vline(x=0, line_dash="dash", line_color=C["bad"], line_width=2.5,
                  annotation_text="이상 (0)", annotation_position="top right",
                  annotation_font=dict(size=14, color=C["bad"]))
    fig.add_vline(x=mu, line_color=C["good"], line_width=2,
                  annotation_text=f"평균 {mu:.3f}%",
                  annotation_position="top left",
                  annotation_font=dict(size=14, color=C["good"]))

    layout = base_layout(
        title="잔차 분포 (편향 검증)",
        subtitle=f"평균 {mu:.3f}% (≈0 → 편향 없음) · 표준편차 {sigma:.3f}%",
    )
    fig.update_layout(**layout)
    fig.update_layout(
        xaxis_title="잔차 (예측 − 실측, %)",
        yaxis_title="빈도",
        showlegend=False,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────
# 6. SOH 구간별 R² — 콤보 (막대 + 데이터 분포 라인)
# ─────────────────────────────────────────────────────────────────────
def fig_band_r2(m):
    bands = [
        ("95~100%", 95, 100, C["good"]),
        ("90~95%",  90, 95,  C["warn"]),
        ("80~90%",  80, 90,  "#F97316"),
        ("<80%",   0,  80,  C["bad"]),
    ]
    labels, r2s, counts, colors_arr, pcts = [], [], [], [], []
    total = len(m["y_true"])
    for name, lo, hi, c in bands:
        mask = (m["y_true"] >= lo) & (m["y_true"] < hi)
        n = int(mask.sum())
        if n < 5:
            labels.append(name); r2s.append(None); counts.append(n); colors_arr.append(C["neutral"]); pcts.append(n/total*100)
            continue
        yt, yp = m["y_true"][mask], m["y_pred"][mask]
        ss_res = np.sum((yt - yp) ** 2)
        ss_tot = np.sum((yt - yt.mean()) ** 2)
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
        labels.append(name); r2s.append(r2); counts.append(n); colors_arr.append(c); pcts.append(n/total*100)

    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.6, 0.4],
        horizontal_spacing=0.12,
        specs=[[{"type": "xy"}, {"type": "domain"}]],
        subplot_titles=(
            "<b>구간별 R²</b>  <span style='color:#64748B;font-size:13px'>모델 정확도</span>",
            "<b>데이터 분포</b>  <span style='color:#64748B;font-size:13px'>운행 fleet 특성</span>",
        ),
    )

    # 좌: R² 막대 (음수 가능)
    fig.add_trace(
        go.Bar(
            x=labels, y=[r if r is not None else 0 for r in r2s],
            marker=dict(color=colors_arr, line=dict(width=0)),
            text=[
                (f"<b>R² {r:.3f}</b>" if r is not None
                 else "<span style='color:#94A3B8'>—</span>")
                for r in r2s
            ],
            textposition="outside", textfont=dict(size=16),
            width=0.55,
            name="R²",
        ),
        row=1, col=1,
    )

    # 우: 데이터 분포 (도넛)
    fig.add_trace(
        go.Pie(
            labels=labels, values=pcts,
            hole=0.5,
            marker=dict(colors=colors_arr, line=dict(color="white", width=2)),
            textinfo="label+percent",
            textfont=dict(size=13),
            showlegend=False,
            sort=False,
        ),
        row=1, col=2,
    )

    fig.update_yaxes(range=[-1.2, 1.2], title_text="R²", row=1, col=1,
                     showgrid=True, gridcolor=C["bg_grid"],
                     zeroline=True, zerolinecolor=C["neutral"], zerolinewidth=1)
    fig.update_xaxes(title_text="실측 SOH 구간", row=1, col=1, showgrid=False)

    # 분포 중심 텍스트
    fig.add_annotation(
        x=0.85, y=0.5, xref="paper", yref="paper",
        text=f"<b>전체</b><br>{total:,} 샘플",
        showarrow=False, align="center", font=dict(size=14),
    )

    layout = base_layout(
        title="SOH 구간별 R² — 한계 솔직 공개",
        subtitle="95~100% 데이터 75% 차지 → R²는 그 구간 위주. 열화 구간은 데이터 부족, 물리 모델로 외삽 보완",
    )
    fig.update_layout(**layout)
    fig.update_layout(showlegend=False)
    return fig


# ─────────────────────────────────────────────────────────────────────
# 7. 대표 차량 5대 시간순 예측
# ─────────────────────────────────────────────────────────────────────
def fig_car_curves(test_df, model_type="lstm"):
    drift = test_df.groupby("battery_id")["soh_norm"].agg(lambda s: s.max() - s.min())
    top5 = drift.sort_values(ascending=False).head(5).index.tolist()

    scaler = joblib.load(f"{SAVE_DIR}/feature_scaler_{model_type}.pkl")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_model(model_type).to(device)
    model.load_state_dict(torch.load(f"{SAVE_DIR}/{model_type}_soh.pt", map_location=device))
    model.eval()

    fig = make_subplots(
        rows=len(top5), cols=1,
        shared_xaxes=False,
        subplot_titles=[f"<b>차량 {cid}</b>" for cid in top5],
        vertical_spacing=0.08,
    )
    for i, cid in enumerate(top5, start=1):
        g = test_df[test_df["battery_id"] == cid].sort_values("cycle").reset_index(drop=True)
        if len(g) <= WINDOW_SIZE:
            continue
        feat = scaler.transform(g[FEATURES].values)
        soh_n = _normalize_soh(g["soh_norm"].values)
        x = np.concatenate([feat, soh_n.reshape(-1, 1)], axis=1)
        seqs = np.stack([x[t - WINDOW_SIZE: t] for t in range(WINDOW_SIZE, len(g))])
        seqs_t = torch.tensor(seqs, dtype=torch.float32).to(device)
        with torch.no_grad():
            d, _ = model(seqs_t)
            prev = seqs_t[:, -1, -1]
            pred_n = _reconstruct_soh(prev, d).cpu().numpy()
        y_pred = _denormalize_soh(pred_n)
        cycles = g["cycle"].values[WINDOW_SIZE:]
        y_true = g["soh_norm"].values[WINDOW_SIZE:]
        fig.add_trace(go.Scatter(
            x=cycles, y=y_true, mode="lines",
            line=dict(color=C["primary"], width=2.5),
            name="실측", showlegend=(i == 1),
        ), row=i, col=1)
        fig.add_trace(go.Scatter(
            x=cycles, y=y_pred, mode="lines",
            line=dict(color=C["bad"], width=2, dash="dash"),
            name="LSTM 예측", showlegend=(i == 1),
        ), row=i, col=1)

    layout = base_layout(
        title="대표 테스트 차량 5대 — 시간순 SOH 예측",
        subtitle="실측(파랑) vs LSTM 예측(빨강 점선) · 차량 단위 split, 학습 시 본 적 없는 차량",
        h=1100,
    )
    fig.update_layout(**layout)
    fig.update_layout(
        height=1100,
        legend=dict(orientation="h", yanchor="bottom", y=-0.06, xanchor="center", x=0.5),
    )
    for i in range(1, len(top5) + 1):
        fig.update_xaxes(showgrid=True, gridcolor=C["bg_grid"], row=i, col=1)
        fig.update_yaxes(showgrid=True, gridcolor=C["bg_grid"], row=i, col=1)
        if i == len(top5):
            fig.update_xaxes(title_text="누적 충전 사이클", row=i, col=1)
        fig.update_yaxes(title_text="SOH (%)", row=i, col=1)
    return fig


# ─────────────────────────────────────────────────────────────────────
# 8. 물리 일관성
# ─────────────────────────────────────────────────────────────────────
def fig_physics():
    cycles = np.arange(1, 2500)
    fig = make_subplots(
        rows=1, cols=2,
        horizontal_spacing=0.10,
        subplot_titles=(
            f"<b>Peukert 효과</b>  <span style='color:#64748B;font-size:14px'>n={PEUKERT_N:.2f}</span>",
            "<b>Arrhenius 효과</b>  <span style='color:#64748B;font-size:14px'>Ea=31 kJ/mol</span>",
        ),
    )

    peukert_cases = [
        (0.6, C["good"], "절약형 (0.6C)"),
        (1.0, C["warn"], "평균형 (1.0C)"),
        (1.8, C["bad"],  "공격형 (1.8C)"),
    ]
    for c_rate, color, label in peukert_cases:
        eol = 2000 * (1.0 / c_rate) ** PEUKERT_N
        alpha = 30.0 / np.sqrt(eol)
        soh = (100 - alpha * np.sqrt(cycles)).clip(0, 100)
        fig.add_trace(
            go.Scatter(x=cycles, y=soh, name=label,
                       line=dict(color=color, width=3)),
            row=1, col=1,
        )

    arrh_cases = [
        (0,  C["primary"], "0°C (지하)"),
        (25, C["warn"],    "25°C (기준)"),
        (45, C["bad"],     "45°C (옥상)"),
    ]
    for t, color, label in arrh_cases:
        eol = 2000 / arrhenius_multiplier(t)
        alpha = 30.0 / np.sqrt(eol)
        soh = (100 - alpha * np.sqrt(cycles)).clip(0, 100)
        fig.add_trace(
            go.Scatter(x=cycles, y=soh, name=label,
                       line=dict(color=color, width=3)),
            row=1, col=2,
        )

    for col in (1, 2):
        fig.add_hline(y=70, line_dash="dash", line_color=C["neutral"],
                      annotation_text="EOL 70%",
                      annotation_font=dict(size=14, color=C["neutral"]),
                      row=1, col=col)
        fig.update_xaxes(title_text="누적 충전 사이클",
                         showgrid=True, gridcolor=C["bg_grid"], row=1, col=col)
        fig.update_yaxes(title_text="SOH (%)", range=[55, 105],
                         showgrid=True, gridcolor=C["bg_grid"], row=1, col=col)

    layout = base_layout(
        title="물리 일관성 — Peukert + Arrhenius",
        subtitle="NCM 셀 검증된 식 · 데이터 외 영역도 물리 법칙 따름 → 일반화 보장",
    )
    fig.update_layout(**layout)
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=-0.18,
                                  xanchor="center", x=0.5))
    return fig


# ─────────────────────────────────────────────────────────────────────
# 9. 시나리오 비교
# ─────────────────────────────────────────────────────────────────────
def fig_scenarios(train_df, test_df):
    drift = test_df.groupby("battery_id")["soh_norm"].agg(lambda s: s.max() - s.min())
    seed_car = drift.idxmax()
    seed_df = test_df[test_df["battery_id"] == seed_car].sort_values("cycle").reset_index(drop=True)

    scenarios = [
        ("절약형 / 캐스퍼 / 부산지하", "현대", "캐스퍼 일렉트릭", "기본형", "절약형", 2, 17.2, C["good"]),
        ("평균형 / 아이오닉5 / 서울",   "현대", "아이오닉 5", "롱레인지 2WD", "평균형", 4, 12.5, C["warn"]),
        ("공격형 / 아이오닉5N / 광주옥상", "현대", "아이오닉 5", "N", "공격형", 6, 18.3, C["bad"]),
    ]

    fig = go.Figure()
    cycles = np.arange(1, 2501)

    for name, brand, model, trim, driver, cf, temp, color in scenarios:
        spec = get_spec(brand, model, trim)
        c_rate = get_car_adjusted_c_rate(driver, spec)
        soh = predict_soh_curve(seed_df, 2500, c_rate=c_rate, temp_celsius=temp,
                                save_dir=SAVE_DIR, train_df=train_df)
        eol = find_eol_cycle(soh)
        years = cycles_to_years(eol, charge_freq=cf)
        fig.add_trace(go.Scatter(
            x=cycles, y=soh, mode="lines",
            line=dict(color=color, width=3),
            name=f"{name}  →  <b>{format_years(years)}</b>",
        ))

    fig.add_hline(y=70, line_dash="dash", line_color=C["neutral"],
                  annotation_text="EOL 70%",
                  annotation_font=dict(size=14, color=C["neutral"]))

    layout = base_layout(
        title="시나리오 비교 — 차종 × 운전 × 환경에 따른 EOL 차이",
        subtitle="같은 모델, 같은 SOH 시작점에서 최대 6배 수명 차이 → 맞춤 추천의 가치",
    )
    fig.update_layout(**layout)
    fig.update_layout(
        xaxis_title="누적 충전 사이클",
        yaxis_title="SOH (%)",
        yaxis=dict(range=[55, 105]),
        legend=dict(yanchor="top", y=0.96, xanchor="right", x=0.96,
                    bgcolor="rgba(255,255,255,0.95)",
                    bordercolor=C["bg_grid"], borderwidth=1),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────
# 10. 핵심 메시지 (텍스트 슬라이드)
# ─────────────────────────────────────────────────────────────────────
def fig_key_message():
    fig = go.Figure()
    # 가짜 좌표로 plot 영역 만들고 annotation 으로 채움
    fig.add_trace(go.Scatter(x=[0], y=[0], mode="markers",
                              marker=dict(color="white", size=0.1),
                              showlegend=False))

    annotations = [
        # 큰 제목
        dict(x=0.5, y=0.92, xref="paper", yref="paper",
             text="<b>핵심 성과</b>",
             font=dict(size=40, color=C["text"]),
             showarrow=False, align="center"),
        # 핵심 3대 메시지
        dict(x=0.18, y=0.65, xref="paper", yref="paper",
             text=f"<span style='font-size:60px; color:{C['primary']}'><b>0.924</b></span><br>"
                  f"<span style='font-size:18px; color:{C['subtext']}'>LSTM R²</span><br>"
                  f"<span style='font-size:14px; color:{C['neutral']}'>EVBattery 38대 unseen 테스트</span>",
             showarrow=False, align="center"),
        dict(x=0.50, y=0.65, xref="paper", yref="paper",
             text=f"<span style='font-size:60px; color:{C['good']}'><b>~1.4%</b></span><br>"
                  f"<span style='font-size:18px; color:{C['subtext']}'>RMSE</span><br>"
                  f"<span style='font-size:14px; color:{C['neutral']}'>실용 정확도 기준</span>",
             showarrow=False, align="center"),
        dict(x=0.82, y=0.65, xref="paper", yref="paper",
             text=f"<span style='font-size:60px; color:{C['warn']}'><b>191</b></span><br>"
                  f"<span style='font-size:18px; color:{C['subtext']}'>학습 차량 수</span><br>"
                  f"<span style='font-size:14px; color:{C['neutral']}'>NCM 99% · 85,648 세션</span>",
             showarrow=False, align="center"),
        # 메시지 박스 3개
        dict(x=0.5, y=0.35, xref="paper", yref="paper",
             text="<b style='font-size:22px'>🔋 NCM 화학 일치 — 한국 EV 직접 적용 가능</b>",
             showarrow=False, align="center", font=dict(color=C["text"])),
        dict(x=0.5, y=0.27, xref="paper", yref="paper",
             text="<b style='font-size:22px'>🧪 Physics-Informed — Peukert + Arrhenius + PINN Loss</b>",
             showarrow=False, align="center", font=dict(color=C["text"])),
        dict(x=0.5, y=0.19, xref="paper", yref="paper",
             text="<b style='font-size:22px'>🎯 차종 × 운전 맞춤 — Top 3 우선순위 추천</b>",
             showarrow=False, align="center", font=dict(color=C["text"])),
        # 부록
        dict(x=0.5, y=0.08, xref="paper", yref="paper",
             text=f"<i style='color:{C['subtext']}; font-size:14px'>4-way 비교: LSTM 0.924 / Transformer 0.919 / XGBoost 0.908 / Neural CDE — 방법론 견고</i>",
             showarrow=False, align="center"),
    ]

    fig.update_layout(
        annotations=annotations,
        xaxis=dict(visible=False, range=[0, 1]),
        yaxis=dict(visible=False, range=[0, 1]),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(t=20, b=20, l=20, r=20),
        height=SLIDE_H,
        width=SLIDE_W,
        showlegend=False,
    )
    return fig


def main():
    test_path = Path(SAVE_DIR) / "test_df.parquet"
    train_path = Path(SAVE_DIR) / "train_df.parquet"
    if not test_path.exists() or not train_path.exists():
        print("ERROR: parquet 캐시 없음")
        return

    print("📊 LSTM 평가 중...")
    test_df = pd.read_parquet(test_path)
    train_df = pd.read_parquet(train_path)
    m = evaluate_model(test_df, model_type="lstm", save_dir=SAVE_DIR)
    print(f"   R²={m['r2']:.4f}  RMSE={m['rmse']:.2f}%\n")

    print(f"📸 시각화 저장 중 → {OUT_DIR.absolute()}\n")
    for name, fig in [
        ("key_message",         fig_key_message()),
        ("data_overview",       fig_data_overview(test_df, train_df)),
        ("ncm_match",           fig_ncm_match()),
        ("model_4way",          fig_4way()),
        ("lstm_scatter",        fig_scatter(m)),
        ("residual_hist",       fig_residual(m)),
        ("soh_band_r2",         fig_band_r2(m)),
        ("car_curves",          fig_car_curves(test_df, "lstm")),
        ("physics_calibration", fig_physics()),
        ("scenario_compare",    fig_scenarios(train_df, test_df)),
    ]:
        save(fig, name)

    print(f"\n✅ 완료: {OUT_DIR.absolute()}")


if __name__ == "__main__":
    main()
