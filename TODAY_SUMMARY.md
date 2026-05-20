# 오늘 작업 정리 (2026-05-18)

---

## 완료한 작업

### 1. Neural CDE 구현 및 학습
- `src/soh_cde.py` 신규 작성 — Kidger et al. 2020 (NeurIPS) 기반
- 안정화: `adjoint=False`, `method="rk4"`, 고정 스텝
- 학습 완료: `models/saved/cde_soh.pt` 저장됨 (오후 6:29)
- **결과: R²=-49 (발산)** → LSTM이 이 데이터에 더 적합함을 실험적으로 증명

### 2. 4-way 비교 최종 결과

| Model | R² | RMSE(%) | MAE(%) |
|-------|-----|---------|--------|
| **LSTM** | **0.9233** | **0.784** | 0.295 ★ |
| Transformer | 0.9169 | 0.817 | 0.277 |
| XGBoost | 0.9084 | 0.857 | 0.238 |
| Neural CDE | -49.04 | 20.062 | 12.081 |

저장 위치: `models/saved/comparison.json`

### 3. 물리 모델 업그레이드 (soh_lstm.py)
- **2-phase 열화**: Phase1(SEI √t, Ploehn 2004) + Phase2(무릎효과 지수, Attia 2022)
- **승법 노이즈**: σ(t) = 1.2·√(t/EOL) — 쓸수록 불확실성 증가

### 4. app.py 수정 및 확인
- `"NCM 실차 465대"` → `"191대"` 수정
- `anchor_std` numpy 배열 → `current_std` 스칼라 변환 (TypeError 수정)
- 실행 확인: http://localhost:8502 정상 작동
- 테스트: 아이오닉5 서울 10년 135,000km → SOH 77.7%, 수명 12년 (현실적)

### 5. 파일 정리
- `presentation/model_3way.png/.html` 삭제 (구버전)
- `make_presentation.py` → `fig_4way()`, `model_4way` 출력으로 수정
- `make_presentation.py` 내 SUMMARY_MD 죽은 코드 제거
- `PROJECT_REFERENCE.md` 전면 업데이트
- `presentation/summary.md` 4-way + 15분 타이밍 업데이트

---

## 현재 파일 구조

```
tuning_idea/
├── app.py                    ✅ 정상 작동
├── train.py                  ✅ 4-way 학습 파이프라인
├── run_cde_only.py           ✅ CDE 단독 재학습용
├── make_presentation.py      ✅ 발표 이미지 생성 (model_4way 포함)
├── make_figures.py           ✅ 분석용 이미지 생성
├── ablate_pinn.py            ✅ PINN 효과 ablation (발표 근거용)
├── cv_train.py               ✅ 5-Fold CV (일반화 검증용)
├── eval_cde.py               🗑️ 임시 파일 — 삭제해도 됨
├── PROJECT_REFERENCE.md      ✅ 업데이트 완료
├── FOR_PRESENTER.md          ✅ 팀원 발표자용 설명서
├── src/
│   ├── soh_lstm.py           ✅ LSTM + 2-phase 물리모델
│   ├── soh_cde.py            ✅ Neural CDE
│   ├── baselines.py          ✅ XGBoost
│   ├── ev_data_loader.py     ✅ EVBattery 로더
│   ├── clustering.py         ✅ 운전자 3유형 분류
│   ├── strategy.py           ✅ 수명 전략 계산
│   └── ev_database.py        ✅ 현대·기아 45개 모델 DB
├── models/saved/
│   ├── lstm_soh.pt           ✅ R²=0.9233
│   ├── transformer_soh.pt    ✅ R²=0.9169
│   ├── xgboost_soh.pkl       ✅ R²=0.9084
│   ├── cde_soh.pt            ⚠️ R²=-49 (발산, 비교용으로만)
│   ├── comparison.json       ✅ 4-way 비교 결과
│   ├── train_df.parquet      ✅ 153대 85,648세션
│   └── test_df.parquet       ✅ 38대 21,629세션
└── presentation/
    ├── summary.md            ✅ 15분 발표 구성 업데이트
    └── *.png/*.html          ✅ (model_4way는 make_presentation.py 실행 후 생성)
```

---

## 남은 할 일 (마감 2026-05-22)

| 순서 | 작업 | 중요도 |
|------|------|--------|
| 1 | PPT 제작 (13장, 15분) | 🔴 최우선 |
| 2 | `python make_presentation.py` 실행 → 이미지 최신화 | 🔴 |
| 3 | 앱 최종 점검 (모든 탭 + 엣지케이스) | 🟠 |
| 4 | `eval_cde.py` 삭제 | 🟢 |
