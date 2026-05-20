# EV BMS 프로젝트 완전 참조 문서
**마감: 2026-05-22**
**실행: `streamlit run app.py`**
**학습: `python train.py`** (전체 4-way) / **`python run_cde_only.py`** (CDE만)

---

## 0. 현재 상태 (2026-05-18 기준)

### 해결된 걱정들 ✅

| # | 걱정 | 상태 |
|---|------|------|
| 1 | R²가 낮다 | ✅ **해결** — soh_norm 적용으로 LSTM R²=0.924 달성 |
| 2 | EVBattery SOH 예측 적합성 | ✅ **해결** — 차량별 soh_norm으로 열화 궤적 학습 |
| 3 | predict_soh_curve()가 LSTM 안 씀 | ✅ **개선** — 2-phase 열화 모델 + 승법 노이즈 도입 |
| 7 | 주제 바꿔야 하나 | ✅ **해결** — EV 배터리 자가진단 유지, 4-way 모델 비교로 강화 |
| 8 | NASA 사용 | ✅ **제거 예정** — 코드에서 NASA 분리 완료 (app.py만 남음) |
| 9 | 발표 설득력 | ✅ **강화** — 논문 기반 물리모델 + 4-way 비교 + PINN 언급 |
| 10 | XGBoost vs LSTM | ✅ **해결** — 둘 다 구현, LSTM 승 (R²=0.924 vs 0.908) |

### 아직 남은 이슈 🔴🟠

| # | 이슈 | 심각도 | 해결 방법 |
|---|------|--------|-----------|
| 4 | app.py 죽음 | 🔴 | NASA 제거, 슬라이더 5개, 혼동행렬 3유형 |
| 5 | 마감 4일 남음 | 🔴 | 아래 우선순위 참고 |
| 6 | GPU 메모리 이슈 | 🟠 | RTX 5060 16GB — batch_size 64로 OK |

---

## 1. 프로젝트 목적

현대·기아 EV 오너용 배터리 자가진단 웹앱.
- 현재 배터리 상태 추정
- 미래 열화 곡선 예측 (2-phase: SEI + 무릎 효과)
- 교체 시점 ("3년 2개월 후")
- 습관 개선 시 수명 연장 효과 ("급속충전 줄이면 +1년 5개월")

---

## 2. 사용 데이터

### EVBattery (메인) ✅
- 출처: Figshare "EVBattery: A Large-Scale Electric Vehicle Dataset"
- 경로: /data/evbattery/battery_dataset1~3/ (15GB, 압축해제 완료)
- 차량: 465대 (NCM 99%, 중국 실차), 유효 191대
- 학습 153대 (85,648세션) / 테스트 38대 (21,629세션)
- 파일형식: pkl → `torch.load(f, weights_only=False)` 로 읽기
- SOH 분포: min 64%, max 100%, mean 97.5%, std 4.27%
- SOH<80% 차량: 14대 / SOH<90% 차량: 79대
- **왜 쓰는가:** NCM 화학 = 현대·기아 EV와 동일

### NASA ❌ 제거됨
- 화학: LCO → NCM과 다름, 한국 EV에 부적합
- app.py에서 아직 사용 중 → EVBattery로 교체 필요

---

## 3. 모델 현황 (4-way 비교)

### 학습 결과 (2026-05-18)

| Model | R² | RMSE(%) | MAE(%) | 상태 |
|-------|-----|---------|--------|------|
| LSTM | **0.9240** | ~1.4% | ~1.0% | ✅ 완료 |
| Transformer | 0.9190 | ~1.5% | ~1.1% | ✅ 완료 |
| XGBoost | 0.9080 | ~1.6% | ~1.2% | ✅ 완료 |
| Neural CDE | 학습 중 | - | - | 🔄 진행 중 |

> comparison.json에 저장됨: `models/saved/comparison.json`

### 각 모델 파일
- `models/saved/lstm_soh.pt` — LSTM 가중치
- `models/saved/transformer_soh.pt` — Transformer 가중치
- `models/saved/xgb_soh.pkl` — XGBoost 모델
- `models/saved/cde_soh.pt` — Neural CDE 가중치 (학습 중)
- `models/saved/feature_scaler.pkl` — LSTM/Transformer용
- `models/saved/feature_scaler_cde.pkl` — CDE용 별도 스케일러

---

## 4. 물리 모델 (predict_soh_curve)

### 2-phase 열화 모델 (논문 근거)

**Phase 1: SEI 성장 (√t 법칙)**
- 수식: SOH = 100 - α·√t
- 근거: Ploehn et al. 2004 (J. Electrochem. Soc.); Safari & Delacourt 2011
- 메커니즘: 음극 표면 SEI 막이 √t 비례 성장 → 용량 손실

**Phase 2: 가속 열화 (무릎 효과, SOH < 85%)**
- 수식: SOH = 100 - (100-knee_SOH) · exp(k·(t - t_knee))
- 근거: Attia et al. 2022 (Nature Energy); Cannarella & Arnold 2014
- 메커니즘: 리튬 도금(Li plating) + 양극 구조 붕괴 → 지수 가속

**승법 노이즈 (불확실성)**
- 수식: σ(t) = 1.2 · √(t/EOL)
- 의미: 쓸수록 열화 불확실성 증가 (실사용 환경 가변성)
- 구현: `soh_lstm.py._two_phase_soh()`

### PINN 손실 함수 (PhysicsInformedLoss)
- Peukert Law: n=0.621 (Safari 2011, Pelletier 2017 피팅)
  → 0.5C: ~780사이클, 1.0C: ~550사이클, 2.0C: ~330사이클 (80% SOH 기준)
- Arrhenius Law: Ea=31,000 J/mol, Q10≈1.5
  → Wang 2014, Waldmann 2014

---

## 5. 파일별 상태

### 핵심 파일 ✅ 완성

| 파일 | 상태 | 비고 |
|------|------|------|
| `src/ev_data_loader.py` | ✅ 완성 | soh_norm, c_rate, thermal_stress, degradation_index 포함 |
| `src/soh_lstm.py` | ✅ 완성 | FEATURES 13개, 2-phase 열화, PhysicsInformedLoss |
| `src/soh_cde.py` | ✅ 완성 | Neural CDE (Kidger 2020), adjoint=False, rk4 |
| `src/clustering.py` | ✅ 완성 | 3유형 (절약/평균/공격), 5개 입력 |
| `src/strategy.py` | ✅ 완성 | 3유형 기준, 수정 불필요 |
| `src/ev_database.py` | ✅ 완성 | 현대 25 + 기아 20 모델 |
| `train.py` | ✅ 완성 | 4-way 학습 + comparison.json |
| `run_cde_only.py` | ✅ 완성 | CDE만 단독 재학습용 |

### 수정 필요 🔴

| 파일 | 문제 | 우선순위 |
|------|------|---------|
| `app.py` | NASA 사용, 슬라이더 불일치, 혼동행렬 하드코딩 | 🔴 마감 전 필수 |

### 확인/정리 필요 🟠

| 파일 | 상태 |
|------|------|
| `ablate_pinn.py` | PINN ablation 실험용 — 발표 자료로 쓸 수 있으면 유지 |
| `cv_train.py` | 교차검증 실험용 — 현재 사용 안 함 |
| `presentation/` | 3-way → 4-way로 업데이트 필요 |
| `figures/` | CDE 결과 나오면 재생성 필요 |

---

## 6. Neural CDE 구현 상세

**논문:** Kidger et al. 2020, "Neural Controlled Differential Equations for Irregular Time Series", NeurIPS 2020

**수식:** dZ(t) = f_θ(Z(t)) · dX(t), t ∈ [0, 1]

**구현 핵심:**
```python
# torchcde 0.2.5 API (NaturalCubicSpline → CubicSpline으로 변경됨)
coeffs   = torchcde.natural_cubic_coeffs(X, t=t)
X_spline = torchcde.CubicSpline(coeffs)   # NOT NaturalCubicSpline

z = torchcde.cdeint(
    X=X_spline, func=self.func, z0=z0, t=t_eval,
    adjoint=False,           # 직접 autograd (adjoint는 수치 발산)
    method="rk4",            # 고정 스텝 (adaptive solver 발산 방지)
    options={"step_size": 1.0 / (WINDOW_SIZE - 1)},
)
```

**하이퍼파라미터:** hidden_dim=64, lr=1e-4, batch_size=64, patience=20

---

## 7. 실행 환경

- OS: Windows 11 Pro
- GPU: RTX 5060 (CUDA 12.8)
- Python: 3.14
- torch: 2.11.0+cu128 (cu128 wheel에서 설치)
- torchcde: 0.2.5

### torch CUDA 설치 주의
Python 3.14 + CUDA 12.8은 PyPI 기본 torch가 CPU 버전 설치됨.
반드시 아래로 설치:
```bash
pip install torch==2.11.0+cu128 --index-url https://download.pytorch.org/whl/cu128
```

---

## 8. 마감 전 우선순위 (2026-05-22)

1. **Neural CDE 결과 확인** — 학습 완료 후 4-way 비교표 확인
2. **app.py 수정** — NASA 제거, EVBattery 연결, 슬라이더 5개, 혼동행렬 3유형
3. **figures/ 업데이트** — CDE 포함 4-way 비교 차트
4. **presentation/ 업데이트** — 4-way 비교 결과 반영
5. **앱 실행 확인** — streamlit run app.py 정상 동작 확인

---

## 9. 발표 핵심 포인트

1. **인포테인먼트와 차이:** 현재 SOH만 → 미래 예측 + 교체 시점
2. **NCM 실차 191대:** NASA LCO 버리고 실제 화학 일치
3. **4-way 모델 비교:** LSTM vs Transformer vs XGBoost vs Neural CDE
4. **2-phase 물리 모델:** SEI(√t) + 무릎효과(exponential) — 논문 근거
5. **PINN:** Peukert n=0.62 / Arrhenius Ea=31kJ/mol — 물리 제약을 손실함수에 내장
6. **승법 노이즈:** 쓸수록 불확실성 증가 (σ = 1.2·√(t/EOL))
7. **운전 습관 → 교체 시점 연장** 구체적 수치 제공

---

## 10. src/strategy.py (수정 불필요)

```python
C_RATE_MAP = {"공격형": 1.8, "평균형": 1.0, "절약형": 0.6}
C_RATE_OPTIMIZED = {"공격형": 1.2, "평균형": 0.8, "절약형": 0.5}
BATTERY_REPLACE_COST_KRW = 12_000_000
DAYS_PER_CYCLE = 35   # Geotab 22,700대 기준
KM_PER_CYCLE = 1200
```

---

## 11. src/ev_database.py (수정 불필요)

현대 25모델 + 기아 20모델. NCM 99% (예외: 레이EV=LFP).
- 아이오닉5 롱레인지 2WD: 84kWh, 697V, SK ON, NCM, 파우치형
- EV6 GT: 84kWh, 697V, SK ON, NCM, 파우치형
- 레이EV: 35.2kWh, CATL, LFP (예외)
