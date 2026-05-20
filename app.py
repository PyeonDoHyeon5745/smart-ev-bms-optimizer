import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from src.clustering import (
    predict_driver_type,
    predict_driver_type_proba,
    FEATURES as CLUSTER_FEATURES,
)
from src.soh_lstm import (
    predict_soh_curve,
    arrhenius_multiplier,
    PEUKERT_N,
)
from src.strategy import (
    get_strategy,
    find_eol_cycle,
    cycles_to_years,
    format_years,
    estimate_cost_saving,
    estimate_current_cycle,
    DAYS_PER_CYCLE,
    get_car_adjusted_c_rate,
    get_car_replacement_cost,
    get_car_km_per_cycle,
    get_car_power_density,
    personalized_recommendations,
    total_recommendation_effect,
)
from src.ev_database import get_brands, get_models, get_trims, get_spec

_APP_DIR = Path(__file__).parent
MODELS_DIR = str(_APP_DIR / "models" / "saved")
N_FUTURE = 2500

TRAIN_CACHE = Path(MODELS_DIR) / "train_df.parquet"
TEST_CACHE = Path(MODELS_DIR) / "test_df.parquet"

# 한국 지역별 연평균 기온 (기상청 평년값 기준)
KOREA_REGION_TEMP = {
    "서울":         12.5,
    "부산":         14.7,
    "대구":         14.1,
    "인천":         12.1,
    "광주":         13.8,
    "대전":         13.0,
    "울산":         14.1,
    "제주":         15.6,
    "강원도":       11.0,
    "충청도":       12.5,
    "전라도":       13.5,
    "경상도":       13.5,
    "수도권 (경기)": 12.5,
}


# ─────────────────────────────────────────────────────────────────────
# Data cache
# ─────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def get_train_df() -> pd.DataFrame:
    if not TRAIN_CACHE.exists():
        st.error("train_df 캐시가 없습니다. `python train.py` 를 먼저 실행하세요.")
        st.stop()
    return pd.read_parquet(TRAIN_CACHE)


@st.cache_data(show_spinner=False)
def get_test_df() -> pd.DataFrame:
    if not TEST_CACHE.exists():
        st.error("test_df 캐시가 없습니다. `python train.py` 를 먼저 실행하세요.")
        st.stop()
    return pd.read_parquet(TEST_CACHE)


@st.cache_data(show_spinner=False)
def get_seed_df() -> pd.DataFrame:
    """대표 테스트 차량 한 대의 시계열"""
    test_df = get_test_df()
    soh_range = test_df.groupby("battery_id")["soh_norm"].agg(lambda s: s.max() - s.min())
    seed_car = soh_range.idxmax()
    return test_df[test_df["battery_id"] == seed_car].sort_values("cycle").reset_index(drop=True)


def check_models_exist() -> bool:
    required = ["lstm_soh.pt", "feature_scaler_lstm.pkl", "rf_model.pkl", "scaler.pkl",
                "train_df.parquet", "test_df.parquet"]
    return all((Path(MODELS_DIR) / f).exists() for f in required)


def soh_color(soh: float) -> str:
    if soh >= 90:
        return "#22C55E"
    elif soh >= 80:
        return "#FBBF24"
    elif soh >= 70:
        return "#F97316"
    return "#EF4444"


def soh_status(soh: float) -> str:
    if soh >= 90:
        return "양호"
    elif soh >= 80:
        return "주의"
    elif soh >= 70:
        return "관찰"
    return "교체 권고"


# ─────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────
def build_soh_graph(
    soh_current: np.ndarray,
    soh_optimized: np.ndarray,
    eol_current: int,
    eol_optimized: int,
    current_cycle: int,
    current_soh: float,
    current_std: float = None,
):
    fig = go.Figure()
    x = np.arange(1, N_FUTURE + 1)

    # 과거 (회색 점선)
    if current_cycle > 0:
        fig.add_trace(go.Scatter(
            x=x[:current_cycle], y=soh_current[:current_cycle],
            name="이미 지난 기간",
            line=dict(color="#6B7280", width=1.5, dash="dot"),
        ))

    fig.add_trace(go.Scatter(
        x=x[current_cycle:], y=soh_current[current_cycle:],
        name="현재 습관 유지",
        line=dict(color="#EF4444", width=2.8),
    ))
    fig.add_trace(go.Scatter(
        x=x[current_cycle:], y=soh_optimized[current_cycle:],
        name="맞춤 개선 적용",
        line=dict(color="#22C55E", width=2.8),
    ))

    # EOL 70%
    fig.add_hline(y=70, line_dash="dash", line_color="#94A3B8",
                  annotation_text="현대·기아 보증 한계 (70%)",
                  annotation_position="bottom right",
                  annotation=dict(font=dict(color="#94A3B8")))

    # 현재 위치 마커
    if current_cycle > 0:
        fig.add_vline(x=current_cycle, line_color="#FBBF24", line_width=2)
        unc = f" ± {current_std:.1f}%" if current_std is not None else ""
        fig.add_annotation(
            x=current_cycle, y=min(current_soh + 3, 102),
            text=f"📍 현재  {current_soh:.1f}%{unc}",
            showarrow=False,
            font=dict(color="#FBBF24", size=13),
            bgcolor="rgba(30,30,46,0.9)",
            bordercolor="#FBBF24",
            borderwidth=1, borderpad=4,
        )

    if eol_current < N_FUTURE:
        fig.add_vline(x=eol_current, line_dash="dot", line_color="#EF4444", opacity=0.45)
    if eol_optimized < N_FUTURE and eol_optimized != eol_current:
        fig.add_vline(x=eol_optimized, line_dash="dot", line_color="#22C55E", opacity=0.45)

    fig.update_layout(
        title=None,
        xaxis_title="누적 충전 사이클",
        yaxis_title="SOH (%)",
        yaxis=dict(range=[55, 105]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=420,
        margin=dict(t=30, b=40, l=60, r=30),
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="EV 배터리 수명 최적화",
        page_icon="🔋",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ── 사이드바 ───────────────────────────────────────────────
    with st.sidebar:
        st.title("🔋 BMS 튜닝 시스템")
        st.caption("운전 습관과 차량을 입력하면 개인 맞춤형 분석을 제공합니다")

        if not check_models_exist():
            st.error("모델 파일 누락. `python train.py` 먼저 실행")
            st.stop()

        st.divider()
        st.markdown("##### 🚙 차량 정보")
        brand = st.selectbox("브랜드", get_brands(), key="brand")
        model = st.selectbox("모델", get_models(brand), key="model")
        trim = st.selectbox("트림", get_trims(brand, model), key="trim")
        spec = get_spec(brand, model, trim)

        with st.container(border=True):
            c1, c2 = st.columns(2)
            c1.metric("배터리 용량", f"{spec.get('kwh','?')} kWh")
            c2.metric("정격 전압", f"{spec.get('voltage','?')} V")
            c3, c4 = st.columns(2)
            c3.metric("최대 출력", f"{spec.get('motor_kw','?')} kW")
            c4.metric("셀 형태", spec.get('cell', '?'))
            st.caption(f"제조사: {spec.get('maker','?')} · 화학: {spec.get('chemistry','?')}")

        st.divider()
        st.markdown("##### 📍 지역")
        region = st.selectbox("주거 지역", list(KOREA_REGION_TEMP.keys()), index=0)
        avg_temp_base = KOREA_REGION_TEMP[region]
        # 주차 환경 modifier
        parking_type = st.radio(
            "주차 환경",
            ["지하 (안정)", "지상 (옥외)", "옥상/직사광"],
            index=0,
            horizontal=False,
        )
        parking_mod = {"지하 (안정)": 2.5, "지상 (옥외)": 0.0, "옥상/직사광": 4.5}[parking_type]
        avg_temp = avg_temp_base + parking_mod
        st.caption(f"적용 온도: **{avg_temp:.1f}°C** ({region} 연평균 {avg_temp_base:.1f}°C + 주차 {parking_mod:+.1f}°C)")

        st.divider()
        st.markdown("##### 🚗 차량 현황")
        years_used = st.slider("사용 기간 (년)", 0, 20, 3)
        km_driven = st.slider("누적 주행 거리 (km)", 0, 300_000, 45_000,
                              step=5_000, format="%d km")

        st.divider()
        st.markdown("##### 🏎️ 운전 습관")
        mean_speed = st.slider("평균 주행 속도 (km/h)", 40, 140, 75)
        accel_events = st.slider("시간당 급가속 횟수", 0, 30, 10)
        brake_events = st.slider("시간당 급제동 횟수", 0, 25, 8)
        avg_soc_range = st.slider("평균 충전 SOC 폭 (%)", 40, 100, 60,
                                  help="자주 사용하는 SOC 범위의 폭. 클수록 깊은 방전")
        charge_freq = st.slider("주당 충전 횟수", 1, 7, 4)

        st.divider()
        analyze = st.button("🔍 분석하기", use_container_width=True, type="primary")

    # ── 메인 영역 ───────────────────────────────────────────────
    st.title("🔋 EV 배터리 자가진단 + 맞춤 BMS 튜닝")
    st.markdown(
        f"<div style='color:#94A3B8; margin-top:-10px;'>"
        f"NCM 실차 191대 학습 · LSTM R² 0.924 · Peukert n={PEUKERT_N:.2f} · Arrhenius Ea=31 kJ/mol"
        f"</div>",
        unsafe_allow_html=True,
    )
    st.write("")

    if not analyze:
        # 시작 안내
        with st.container(border=True):
            st.markdown("### 시작하기")
            st.markdown(
                "왼쪽 사이드바에서 다음 정보를 입력하고 **🔍 분석하기** 버튼을 누르세요.\n\n"
                "- **차량 정보** — 브랜드/모델/트림\n"
                "- **지역** — 한국 13개 지역 중 선택\n"
                "- **사용 기간 + 누적 주행거리**\n"
                "- **운전 습관 5가지** — 평균 속도, 급가속, 급제동, SOC 폭, 충전 빈도"
            )
            st.caption("분석 결과: 현재 SOH 추정, 미래 열화 곡선, 교체 시점, **맞춤 TOP 3 추천**")
        return

    # ── 분석 실행 ─────────────────────────────────────────────
    with st.spinner("분석 중..."):
        features = {
            "mean_speed":    mean_speed,
            "accel_events":  accel_events,
            "brake_events":  brake_events,
            "avg_soc_range": avg_soc_range,
            "charge_freq":   charge_freq,
        }
        driver_type = predict_driver_type(features, save_dir=MODELS_DIR)
        proba = predict_driver_type_proba(features, save_dir=MODELS_DIR)
        strategy = get_strategy(driver_type, spec)

        seed_df = get_seed_df()
        train_df = get_train_df()
        c_current = get_car_adjusted_c_rate(driver_type, spec, optimized=False)
        c_optimized = get_car_adjusted_c_rate(driver_type, spec, optimized=True)

        soh_current, anchor_std = predict_soh_curve(
            seed_df, N_FUTURE, c_rate=c_current,
            temp_celsius=avg_temp, save_dir=MODELS_DIR, train_df=train_df,
            return_uncertainty=True,
        )
        soh_optimized = predict_soh_curve(
            seed_df, N_FUTURE, c_rate=c_optimized,
            temp_celsius=avg_temp, save_dir=MODELS_DIR, train_df=train_df,
        )

        current_cycle = min(
            estimate_current_cycle(years_used, km_driven, spec, charge_freq=charge_freq),
            N_FUTURE - 1,
        )
        current_soh = float(soh_current[current_cycle])
        current_std = float(anchor_std[current_cycle]) if anchor_std is not None else None

        eol_total = find_eol_cycle(soh_current)
        eol_optimized_total = find_eol_cycle(soh_optimized)

        remaining_current = max(0, eol_total - current_cycle)
        years_total = cycles_to_years(eol_total, charge_freq=charge_freq)
        years_remaining = cycles_to_years(remaining_current, charge_freq=charge_freq)

        recs = personalized_recommendations(
            features, current_soh, spec, avg_temp, driver_type, top_k=3
        )
        total_gain = total_recommendation_effect(recs)
        years_remaining_opt = years_remaining + total_gain
        cost_saving = estimate_cost_saving(total_gain, spec)
        replace_cost = get_car_replacement_cost(spec)

    # ── 결과 표시 ─────────────────────────────────────────────
    emoji_map = {"공격형": "🔴", "평균형": "🟡", "절약형": "🟢"}
    already_eol = current_soh < 70.0
    color = soh_color(current_soh)
    status_text = soh_status(current_soh)

    # SECTION 1 — 운전자 유형 + 현재 SOH 한 줄에
    st.subheader("📊 진단 요약")
    s1, s2, s3, s4 = st.columns([1.2, 1, 1, 1])
    with s1:
        with st.container(border=True):
            st.caption("당신의 운전자 유형")
            unc = f" ± {current_std:.1f}%" if current_std is not None else ""
            st.markdown(
                f"<div style='font-size:38px; line-height:1.1;'>"
                f"{emoji_map[driver_type]} <b>{driver_type}</b>"
                f"</div>"
                f"<div style='color:#94A3B8; margin-top:4px;'>"
                f"실효 C-rate {c_current:.2f}C → 최적 {c_optimized:.2f}C "
                f"(차종 보정 {get_car_power_density(spec):.1f} kW/kWh)"
                f"</div>",
                unsafe_allow_html=True,
            )
    with s2:
        st.metric("현재 SOH (MC Dropout)",
                  f"{current_soh:.1f}%",
                  status_text,
                  delta_color="off")
    with s3:
        st.metric("사용 현황",
                  f"{years_used}년",
                  f"{km_driven:,} km · {current_cycle}cycle")
    with s4:
        st.metric("예상 총 수명",
                  format_years(years_total),
                  f"EOL {eol_total} cycle")

    # 분류 확률 (간단)
    p_cols = st.columns(3)
    for col, dt in zip(p_cols, ["절약형", "평균형", "공격형"]):
        p = proba.get(dt, 0.0)
        marker = " ←" if dt == driver_type else ""
        col.progress(p, text=f"{emoji_map[dt]} {dt}  {p*100:.0f}%{marker}")

    if already_eol:
        st.error("⚠️ 현재 SOH가 보증 한계(70%) 미만입니다. 배터리 교체를 권장합니다.")
    elif current_soh < 80:
        st.warning(f"배터리 SOH {current_soh:.1f}% — 적극적 관리가 필요한 구간입니다.")

    st.divider()

    # SECTION 2 — SOH 곡선
    st.subheader("📈 SOH 열화 예측 곡선")
    st.plotly_chart(
        build_soh_graph(soh_current, soh_optimized, eol_total, eol_optimized_total,
                        current_cycle, current_soh, current_std),
        use_container_width=True,
    )

    r1, r2, r3 = st.columns(3)
    r1.metric(
        "현재 습관 유지 시 잔여 수명",
        format_years(years_remaining) if not already_eol else "0개월",
        f"잔여 {remaining_current} cycle",
    )
    r2.metric(
        "맞춤 개선 적용 시 잔여 수명",
        format_years(years_remaining_opt) if not already_eol else "—",
        f"+{format_years(total_gain)} 연장",
        delta_color="normal" if total_gain > 0.01 else "off",
    )
    r3.metric(
        "예상 절감 비용",
        f"약 {cost_saving:,}원",
        f"교체비 {replace_cost//10000:,}만원 ({spec.get('kwh',0)}kWh)",
        delta_color="off",
    )

    st.divider()

    # SECTION 3 — 맞춤 추천 TOP 3
    st.subheader(f"🎯 당신만을 위한 BMS 튜닝 — {brand} {model} {trim}")
    st.caption(
        f"운전자 패턴 + 차종 spec ({spec.get('kwh','?')}kWh / {spec.get('motor_kw','?')}kW / {spec.get('cell','?')}) "
        f"+ 지역 환경 ({region} {avg_temp:.1f}°C) 종합 분석"
    )

    if not recs:
        st.success(
            "✅ 운전 · 충전 · 환경 패턴이 모두 양호합니다. "
            "큰 개선 여지 없음 — 현재 습관 유지를 권장합니다."
        )
    else:
        medals = ["🥇", "🥈", "🥉"]
        rec_cols = st.columns(len(recs))
        for i, (col, r) in enumerate(zip(rec_cols, recs)):
            with col:
                with st.container(border=True):
                    st.caption(f"{medals[i]} 우선순위 {i+1}")
                    st.markdown(
                        f"<div style='font-size:36px;'>{r['icon']}</div>"
                        f"<div style='font-size:17px; font-weight:600; margin-top:4px;'>{r['category']}</div>",
                        unsafe_allow_html=True,
                    )
                    st.caption(f"현재: **{r['current']}** → 목표: **{r['target']}**")
                    st.markdown(
                        f"<div style='font-size:28px; font-weight:700; color:#22C55E;'>"
                        f"+{format_years(r['effect_years'])}"
                        f"</div>"
                        f"<div style='color:#94A3B8; font-size:12px;'>예상 수명 연장</div>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(f"💡 {r['action']}")
                    st.caption(f"근거: {r['physics']} · 난이도: {r['difficulty']}")

        st.info(
            f"💡 위 {len(recs)}개 모두 적용 시 **+{format_years(total_gain)}** 수명 연장 가능. "
            f"운전 유형 일반 권장(완속 충전, 20-80% SOC 등)이 아니라, "
            f"**본인 패턴에서 효과 큰 행동부터 우선 표시**됩니다."
        )

    # 일반 가이드 (접기)
    with st.expander("📋 일반 충전 가이드 (모든 운전자 공통)"):
        gc1, gc2 = st.columns(2)
        gc1.markdown(f"**권장 SOC 구간**\n\n{strategy['soc_range']}")
        gc2.markdown(f"**급속충전 한도**\n\n{strategy['fast_charge_limit']}")

    st.divider()
    st.caption(
        "EVBattery NCM 465대 (Figshare) · Physics-Informed LSTM + MC Dropout · "
        f"Peukert n={PEUKERT_N:.2f} (Safari 2011, Pelletier 2017) · Arrhenius Ea=31 kJ/mol (Wang 2014, Waldmann 2014)"
    )


if __name__ == "__main__":
    main()
