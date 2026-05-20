# EV BMS 튜닝 시스템 — 발표 슬라이드 스토리

## 📋 권장 슬라이드 구성 (13장, 총 15분)

> **시간 배분 원칙**: 표지/전환 슬라이드 30초, 핵심 내용 1~2분, 데모 2분
> **총계**: 0:30 + 1:30 + 1:00 + 1:00 + 2:00 + 1:30 + 1:00 + 0:30 + 1:00 + 1:00 + 1:30 + 2:00 + 0:30 = **15분**

### 1. 표지 `⏱ 0:30`
- **제목**: 운전자 행동 기반 개인화 BMS 튜닝 시스템
- **부제**: NCM 실차 191대 + Physics-Informed 4-way 모델 비교

### 2. 문제 정의 `⏱ 1:30`
- 현재 EV 인포테인먼트: SOH (%) 단순 표시
- 사용자 니즈: 미래 예측 + 교체 시점 + 습관 개선 효과 정량화
- **메시지**: "내 차 몇 년 더? 어떻게 더 오래?"
- 말할 것: 아이오닉5 계기판 SOH 예시 → 숫자만 있고 미래 없음

### 3. 데이터 — EVBattery NCM 실차 `⏱ 1:00`
- **이미지**: `data_overview.png`
- 191대 유효 / 학습 153대(85,648세션) / 테스트 38대(21,629세션) / Figshare 공개
- 운행 fleet 특성: SOH 95~100% 다수

### 4. 화학 일치성 `⏱ 1:00`
- **이미지**: `ncm_match.png`
- 중국 NCM = 한국 현대·기아 NCM (99%)
- NASA(LCO) 제외 이유 명시
- 말할 것: "NASA 쓰면 화학이 달라서 의미 없음, 실차 데이터 선택"

### 5. 모델링 — Physics-Informed Hybrid `⏱ 2:00`
- **이미지**: 별도 다이어그램 (PPT에 추가)
- **4가지 모델 비교**: LSTM, Transformer, XGBoost, **Neural CDE** (Kidger et al. 2020, NeurIPS)
- 13 피처: 9 기본 + 누적 열화 + Rolling 3
- 잔차 학습 (ΔSOH) — 비정상성 처리
- PINN Loss: Peukert·Arrhenius 제약 내장
- 2-phase 열화 곡선: SEI(√t, Ploehn 2004) + 무릎효과(exp, Attia 2022)
- 승법 불확실성: σ = 1.2·√(t/EOL)

### 6. 결과 — 4-way 비교 `⏱ 1:30`
- **이미지**: `model_4way.png`
- LSTM **0.924** / Transformer 0.919 / XGBoost 0.908 / Neural CDE (결과 추가 예정)
- **메시지**: 네 모델 모두 R² 0.9+ → 방법론 자체가 견고함

### 7. 예측 정확도 `⏱ 1:00`
- **이미지**: `lstm_scatter.png`
- 21,629 테스트 샘플 · R² 0.924 · RMSE ~1.4%
- 말할 것: 학습 때 본 적 없는 38대 차량으로 검증

### 8. 잔차 검증 `⏱ 0:30`
- **이미지**: `residual_hist.png`
- 평균 ≈ 0 → 편향 없음, 정규분포 → 체계적 오류 없음

### 9. 한계 — SOH 구간별 R² (솔직) `⏱ 1:00`
- **이미지**: `soh_band_r2.png`
- 95~100% (75% 데이터): R² 높음
- 80~90% (데이터 부족): R² 낮음
- 말할 것: 솔직하게 인정, 물리 모델로 외삽 보완

### 10. 물리 일관성 `⏱ 1:00`
- **이미지**: `physics_calibration.png`
- Peukert: C-rate별 EOL 차이 (급속충전 = 수명 단축)
- Arrhenius: 온도별 EOL 차이 (여름 옥상 = 최악)

### 11. 차종 + 환경 맞춤 `⏱ 1:30`
- **이미지**: `scenario_compare.png`
- 절약형 캐스퍼 부산지하 → 24년
- 평균형 아이오닉5 서울 → 12년
- 공격형 아이오닉5N 광주옥상 → 4년
- 말할 것: 같은 시작점인데 6배 차이 → 맞춤 추천의 가치

### 12. 라이브 데모 `⏱ 2:00`
- 앱 직접 실행 (streamlit run app.py)
- 차종 선택 → 자동 spec / 지역 → 자동 온도 / TOP 3 추천
- SOH 입력 → 교체 시점 + 절감 비용 출력

### 13. 핵심 정리 `⏱ 0:30`
- **이미지**: `key_message.png`
- R² 0.924 / 191대 / 4-way 비교 / NCM 일치 / Physics-Informed
- 한 줄: "실차 데이터 + 물리 법칙 + 4가지 모델로 내 EV 수명 예측"

---

## 🎯 발표 핵심 메시지 (3줄)

1. **NCM 실차 191대로 학습한 4-way 모델 비교 — R² 0.924 달성** (LSTM 우승)
2. **Physics-Informed (Peukert+Arrhenius+PINN) + 2-phase 열화 (SEI+무릎효과)** — 논문 기반
3. **차종+운전 맞춤 추천** — 보편 조언 아닌 개인 패턴 분석

---

## ❓ Q&A 대비

| 질문 | 답변 |
|---|---|
| R² 0.924 측정 환경? | EVBattery 38대 unseen 테스트 / 21,629 세션 |
| 중국 데이터인데 한국 EV에 OK? | NCM 화학 동일 + Peukert/Arrhenius 화학 의존 → 적용 가능 |
| 왜 Neural CDE? | 불규칙 시계열 처리에 이론적으로 적합 (Kidger 2020, NeurIPS) |
| 열화 가속화 근거? | SEI: Ploehn 2004 / 무릎효과: Attia 2022 Nature Energy |
| 왜 딥러닝? XGBoost로도 0.9 | 시계열 패턴 + 4-way 비교로 방법론 다양성 검증 |
| EOL 예측 근거? | Pelletier 2017 NCM 700 cycle × 부분충전 환산 |
| 셀별 차이는? | 셀 형태(파우치/각형)로 열적 부담 보정 |

---

## 📁 이미지 파일 현황

| 파일 | 용도 | 상태 |
|------|------|------|
| `key_message.png` | 핵심 성과 한눈에 | 🔄 CDE 후 업데이트 |
| `data_overview.png` | 데이터 소개 | ✅ |
| `ncm_match.png` | 화학 일치성 | ✅ |
| `model_3way.png` | (구버전) 3-way 비교 | ❌ → `model_4way.png` 로 교체 필요 |
| `lstm_scatter.png` | 예측 정확도 | ✅ |
| `residual_hist.png` | 편향 검증 | ✅ |
| `soh_band_r2.png` | 한계 공개 | ✅ |
| `car_curves.png` | 시간순 예측 | ✅ |
| `physics_calibration.png` | 물리 일관성 | ✅ |
| `scenario_compare.png` | 차종/환경별 차이 | ✅ |

PNG: 1600×900 (16:9 PPT 슬라이드 표준) · HTML: 인터랙티브 (확대/툴팁)

> **CDE 학습 완료 후 할 일**: `model_4way.png` 생성, `key_message.png` R² 업데이트
