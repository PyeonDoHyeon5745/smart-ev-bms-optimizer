"""
전략/추천 계산 (v3.5 — 차종 spec 활용 + 맞춤 추천 엔진)
─────────────────────────────────────────────────────────────────────
교체 기준: SOH 70% (현대·기아 공식 보증)
"""

import numpy as np
from src.soh_lstm import arrhenius_multiplier, PEUKERT_N

# ─────────────────────────────────────────────────────────────────────
# 운전자 유형별 기본 C-rate
# ─────────────────────────────────────────────────────────────────────
C_RATE_MAP = {
    "공격형": 1.8,
    "평균형": 1.0,
    "절약형": 0.6,
}

C_RATE_OPTIMIZED = {
    "공격형": 1.2,
    "평균형": 0.8,
    "절약형": 0.5,
}

STRATEGY_INFO = {
    "공격형": {
        "emoji": "🔴",
        "soc_range": "20~80%",
        "current_soc": "10~100%",
        "fast_charge_limit": "주 2회 이하",
        "charge_type": "완속 충전 우선 권장",
        "life_extension_years": 1.5,
        "details": [
            "충전 구간을 20~80%로 제한 (현재 10~100%)",
            "급속충전 횟수를 주 2회 이하로 줄이기",
            "완속 충전을 우선적으로 사용",
            "주행 속도를 점진적으로 낮추기",
        ],
    },
    "평균형": {
        "emoji": "🟡",
        "soc_range": "20~80%",
        "current_soc": "20~80%",
        "fast_charge_limit": "주 3회 이하",
        "charge_type": "완속 충전 우선",
        "life_extension_years": 1.0,
        "details": [
            "충전 구간 20~80% 유지",
            "완속 충전 우선 사용",
            "급가속·급제동 줄이기",
        ],
    },
    "절약형": {
        "emoji": "🟢",
        "soc_range": "30~80%",
        "current_soc": "30~80%",
        "fast_charge_limit": "현재 수준 유지",
        "charge_type": "현재 습관 유지",
        "life_extension_years": 0.5,
        "details": [
            "충전 구간 30~80% 유지 (현재 양호)",
            "현재 운전 습관 지속",
            "정기적인 배터리 상태 점검 권장",
        ],
    },
}

DAYS_PER_CYCLE = 35              # legacy: 평균 charge_freq=2/주 가정
KM_PER_CYCLE_DEFAULT = 1200
COST_PER_KWH_KRW = 150_000       # 배터리 교체 단가
REFERENCE_POWER_DENSITY = 2.0    # 평균 motor_kw / kwh (보정 기준)

# NCM 셀 수명 기준 (Pelletier 2017 외)
EOL_CYCLES_AT_BASE = 700         # C=1.0, T=25°C 에서 70% SOH 도달 cycle 수
EOL_THRESHOLD = 70.0             # 현대·기아 보증 한계 (%)


# ─────────────────────────────────────────────────────────────────────
# EOL 찾기 (70% 기준)
# ─────────────────────────────────────────────────────────────────────
def find_eol_cycle(soh_curve: np.ndarray, threshold: float = 70.0) -> int:
    below = np.where(soh_curve < threshold)[0]
    return int(below[0]) if len(below) > 0 else len(soh_curve)


# ─────────────────────────────────────────────────────────────────────
# 차종별 보정 함수
# ─────────────────────────────────────────────────────────────────────
def get_car_power_density(spec: dict) -> float:
    """모터 출력 / 배터리 용량 — 부하 잠재력 지표"""
    return spec.get("motor_kw", 100) / max(spec.get("kwh", 50), 1)


def get_car_c_rate_factor(spec: dict) -> float:
    """차종 보정 계수 (1.0 = 평균)"""
    pd = get_car_power_density(spec)
    # 평균 2.0 기준, sqrt로 완만 보정 (0.5 ~ 1.8 범위)
    factor = np.sqrt(pd / REFERENCE_POWER_DENSITY)
    return float(np.clip(factor, 0.5, 1.8))


def get_car_adjusted_c_rate(driver_type: str, spec: dict, optimized: bool = False) -> float:
    """운전자 유형 × 차종 보정 → 실효 C-rate"""
    base_map = C_RATE_OPTIMIZED if optimized else C_RATE_MAP
    base = base_map[driver_type]
    return float(base * get_car_c_rate_factor(spec))


def get_car_replacement_cost(spec: dict) -> int:
    """차종별 교체비 (kWh × 단가 + 셀 타입 보정)"""
    base = spec.get("kwh", 80) * COST_PER_KWH_KRW
    cell = spec.get("cell", "파우치형")
    if cell == "각형":
        base *= 0.92  # 각형 약간 저렴
    return int(base)


def get_car_km_per_cycle(spec: dict) -> int:
    """차종별 주행거리/사이클 환산 (전비 5 km/kWh × 부분충전 환산 3)"""
    return int(spec.get("kwh", 80) * 5 * 3)


def get_cell_thermal_factor(spec: dict) -> float:
    """셀 타입별 열적 부담 보정 (각형은 방열 약함)"""
    cell = spec.get("cell", "파우치형")
    return 1.15 if cell == "각형" else 1.0


# ─────────────────────────────────────────────────────────────────────
# 시간/사이클 환산
# ─────────────────────────────────────────────────────────────────────
def estimate_current_cycle(
    years_used: float, km_driven: int, spec: dict = None, charge_freq: float = None
) -> int:
    """
    캘린더(충전 빈도 기반) + 주행거리 노화 중 더 큰 값.
    charge_freq (주당 충전 횟수)가 주어지면 그 기반으로 정확 환산.
    """
    if charge_freq is None or charge_freq <= 0:
        cycles_from_time = years_used * 365.0 / DAYS_PER_CYCLE
    else:
        cycles_from_time = years_used * 52.0 * float(charge_freq)
    km_per_cycle = get_car_km_per_cycle(spec) if spec else KM_PER_CYCLE_DEFAULT
    cycles_from_km = km_driven / km_per_cycle
    return int(max(cycles_from_time, cycles_from_km))


def cycles_to_years(n_cycles: int, charge_freq: float = None) -> float:
    """사이클 → 년. charge_freq 있으면 정확 환산."""
    if charge_freq is None or charge_freq <= 0:
        return n_cycles * DAYS_PER_CYCLE / 365.0
    return n_cycles / (52.0 * float(charge_freq))


def format_years(years: float) -> str:
    y = int(years)
    m = int(round((years - y) * 12))
    if m == 12:
        y += 1
        m = 0
    return f"{y}년 {m}개월" if y > 0 else f"{m}개월"


def get_strategy(driver_type: str, spec: dict = None) -> dict:
    info = STRATEGY_INFO[driver_type]
    if spec is None:
        c_current = C_RATE_MAP[driver_type]
        c_optimized = C_RATE_OPTIMIZED[driver_type]
    else:
        c_current = get_car_adjusted_c_rate(driver_type, spec, optimized=False)
        c_optimized = get_car_adjusted_c_rate(driver_type, spec, optimized=True)
    return {
        **info,
        "driver_type": driver_type,
        "c_rate_current": c_current,
        "c_rate_optimized": c_optimized,
    }


def estimate_cost_saving(extra_years: float, spec: dict = None) -> int:
    """수명 연장이 가져오는 절감 효과 (교체비 / 10년 × 연장)"""
    if spec is None:
        battery_cost = 12_000_000
    else:
        battery_cost = get_car_replacement_cost(spec)
    cost_per_year = battery_cost / 10.0
    return int(extra_years * cost_per_year)


def temp_effect_label(temp_celsius: float) -> str:
    mult = arrhenius_multiplier(temp_celsius)
    pct = (mult - 1.0) * 100.0
    if abs(pct) < 2:
        return "기준 온도 (영향 없음)"
    direction = "가속" if pct > 0 else "억제"
    return f"25°C 대비 열화 {abs(pct):.0f}% {direction}"


# ─────────────────────────────────────────────────────────────────────
# 맞춤 추천 엔진 (v3.5 핵심)
# ─────────────────────────────────────────────────────────────────────
BASE_EOL_YEARS = 10.0   # 현대·기아 70% / 10년 보증 기준 기대 수명
# 효과 스케일: 단일 행동이 전체 수명을 좌우하지 않도록 보수적으로 적용
# 한 항목이 좌우하는 수명 최대 비중을 25% 수준으로 cap
_PEUKERT_GAIN_SCALE = 0.30
_ARRHENIUS_GAIN_SCALE = 0.30


def _peukert_life_gain(c_current: float, c_target: float, base: float = BASE_EOL_YEARS) -> float:
    """C-rate 개선 → 수명 연장 (년) — Peukert: cycle-life ∝ C^(-n)
    실제 노화는 캘린더 + 사이클 혼합이라 효과의 일부분만 반영"""
    if c_current <= c_target * 1.01:
        return 0.0
    ratio = c_current / c_target
    return base * _PEUKERT_GAIN_SCALE * (1 - (1 / ratio) ** PEUKERT_N)


def _arrhenius_life_gain(t_current: float, t_target: float, base: float = BASE_EOL_YEARS) -> float:
    """온도 개선 → 캘린더 수명 연장 (년)"""
    mc = arrhenius_multiplier(t_current)
    mt = arrhenius_multiplier(t_target)
    if mc <= mt * 1.01:
        return 0.0
    ratio = mc / mt
    return base * _ARRHENIUS_GAIN_SCALE * (1 - 1 / ratio)


def personalized_recommendations(
    features: dict,
    current_soh: float,
    spec: dict,
    avg_temp: float,
    driver_type: str,
    top_k: int = 3,
) -> list:
    """
    6개 입력 + 차종 spec → Peukert/Arrhenius 기반 TOP K 맞춤 추천
    각 항목별 수명 연장 효과 (년) 정량화하여 우선순위 정렬
    """
    recs = []
    car_factor = get_car_c_rate_factor(spec)
    thermal_factor = get_cell_thermal_factor(spec)

    # 1. 평균 주행 속도 → 평균 C-rate (Peukert)
    speed = features["mean_speed"]
    if speed >= 85:
        # 속도 → 평균 C-rate 대략 모델 (40km/h 0.0C 기준, 140km/h 2.5C)
        c_now = max(0.3, (speed - 40) / 40) * car_factor
        c_tgt = max(0.3, (80 - 40) / 40) * car_factor
        gain = _peukert_life_gain(c_now, c_tgt)
        if gain >= 0.1:
            recs.append({
                "icon": "🛣️",
                "category": "평균 주행 속도",
                "current": f"{int(speed)} km/h",
                "target": "80~90 km/h",
                "effect_years": gain,
                "physics": f"Peukert n={PEUKERT_N:.2f}",
                "action": "고속도로 정속 주행, 추월차로 비율 줄이기",
                "difficulty": "쉬움",
            })

    # 2. 급가속 빈도 → 피크 C-rate
    accel = features["accel_events"]
    if accel >= 12:
        c_now = (1.0 + accel / 12) * car_factor
        c_tgt = (1.0 + 5 / 12) * car_factor
        gain = _peukert_life_gain(c_now, c_tgt) * 0.4   # 평균보다 약한 영향
        if gain >= 0.1:
            recs.append({
                "icon": "🚀",
                "category": "급가속 횟수",
                "current": f"{accel}회/h",
                "target": "5회/h 이하",
                "effect_years": gain,
                "physics": "피크 C-rate (Pelletier 2017)",
                "action": "가속 페달 1초 더 천천히, 신호 예측",
                "difficulty": "중간",
            })

    # 3. SOC 폭 (deep discharge → 음극 격자 응력)
    soc_range = features["avg_soc_range"]
    if soc_range >= 75:
        excess = soc_range - 60
        gain = BASE_EOL_YEARS * (excess / 100) * 0.08   # 보수 추정
        if gain >= 0.1:
            recs.append({
                "icon": "🔋",
                "category": "충전 SOC 폭",
                "current": f"{int(soc_range)}%",
                "target": "60% 이하 (20~80% 구간)",
                "effect_years": gain,
                "physics": "음극 응력 (Attia 2020)",
                "action": "20% 이하 방전 피하기, 80% 충전 알림 활용",
                "difficulty": "쉬움",
            })

    # 4. 주차 환경 온도 (Arrhenius — 셀 타입 보정)
    if abs(avg_temp - 25) >= 7:
        target_t = 25
        gain = _arrhenius_life_gain(avg_temp, target_t) * thermal_factor
        if gain >= 0.1:
            direction = "낮추기" if avg_temp > 25 else "올리기"
            action = ("여름철 지하/실내 주차" if avg_temp > 25
                      else "겨울철 실내 차고 또는 보조 배터리 히터")
            recs.append({
                "icon": "🌡️",
                "category": "주차 환경 온도",
                "current": f"{avg_temp}°C",
                "target": "20~25°C",
                "effect_years": gain,
                "physics": f"Arrhenius Ea=31 kJ/mol · 셀:{spec.get('cell','파우치형')}",
                "action": f"{action} → 온도 {direction}",
                "difficulty": "중간",
            })

    # 5. 충전 빈도 (사이클 카운트 직접 영향)
    charge = features["charge_freq"]
    if charge >= 5:
        ratio = charge / 4
        gain = BASE_EOL_YEARS * (1 - 1 / ratio) * 0.15   # 부분 효과
        if gain >= 0.1:
            recs.append({
                "icon": "⚡",
                "category": "충전 빈도",
                "current": f"주 {charge}회",
                "target": "주 3~4회",
                "effect_years": gain,
                "physics": "사이클 수명 (Safari 2011)",
                "action": "한 번 충전을 깊게(20→80%), 짧은 보충 충전 줄이기",
                "difficulty": "쉬움",
            })

    # 6. 급제동 빈도 → 회생제동 미활용
    brake = features["brake_events"]
    if brake >= 10:
        excess = brake - 5
        gain = BASE_EOL_YEARS * (excess / 25) * 0.04   # 부수 효과
        if gain >= 0.1:
            recs.append({
                "icon": "🛞",
                "category": "회생제동 활용도",
                "current": f"급제동 {brake}회/h",
                "target": "i-Pedal/원페달 활용",
                "effect_years": gain,
                "physics": "회생 효율 (Keil 2016)",
                "action": "i-Pedal 모드 활성화, 미리 감속하는 예측 운전",
                "difficulty": "중간",
            })

    # 효과 큰 순 정렬
    recs.sort(key=lambda r: r["effect_years"], reverse=True)

    # 차종 출력밀도 보정 (전체 효과 ±10~20%)
    car_mult = 1.0 + (get_car_power_density(spec) - REFERENCE_POWER_DENSITY) * 0.05
    car_mult = float(np.clip(car_mult, 0.85, 1.30))
    for r in recs:
        r["effect_years"] *= car_mult

    return recs[:top_k]


def total_recommendation_effect(recs: list) -> float:
    """추천을 모두 적용했을 때 수명 연장 (선형 합 — 보수적 추정)"""
    # 효과들이 일부 중복되므로 0.7 factor
    return sum(r["effect_years"] for r in recs) * 0.7
