"""
SOH 예측 모델 (v3.5 — 일반화 우선)
─────────────────────────────────────────────────────────────────────
v3 → v3.5 변경 사항 (모두 정공법):
  · 13 피처 (기존 9 + cum_degradation + temp_roll_5 + c_rate_roll_5 + deg_roll_5)
  · 잔차 학습: 절대 SOH 대신 ΔSOH 예측 (비정상성 처리, 일반화에 유리)
  · 멀티태스크: ΔSOH + 보조 rate
  · Physics-Informed Loss: Peukert/Arrhenius 제약을 ΔSOH 부호·크기에 부여
  · MC Dropout: 추론 시 불확실성 정량화
"""

import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

# ─────────────────────────────────────────────────────────────────────
WINDOW_SIZE = 30

FEATURES = [
    # 9 base
    "voltage_mean", "voltage_std", "current_mean", "temp_mean",
    "capacity_fade_rate", "cycle_norm",
    "c_rate", "thermal_stress", "degradation_index",
    # v3.5: 누적 + rolling
    "cum_degradation",
    "temp_roll_5", "c_rate_roll_5", "deg_roll_5",
]
_INPUT_SIZE = len(FEATURES) + 1  # +1 for soh_norm channel
BASE_C_RATE = 1.0

# soh_norm 정규화 범위
SOH_MIN = 60.0
SOH_MAX = 100.0

# Peukert (Safari 2011, Pelletier 2017)
_PEUKERT_FIT_PTS = np.array([(0.5, 780), (1.0, 550), (2.0, 330)], dtype=float)
_slope, _ = np.polyfit(np.log(_PEUKERT_FIT_PTS[:, 0]), np.log(_PEUKERT_FIT_PTS[:, 1]), 1)
PEUKERT_N = float(-_slope)  # ≈ 0.621

# Arrhenius (Wang 2014, Waldmann 2014)
_ARRHENIUS_EA = 31_000.0
_R_GAS = 8.314
_T_REF_K = 298.15


def arrhenius_multiplier(temp_celsius: float) -> float:
    T = temp_celsius + 273.15
    exponent = _ARRHENIUS_EA / _R_GAS * (1.0 / _T_REF_K - 1.0 / T)
    return float(np.exp(exponent))


# ─────────────────────────────────────────────────────────────────────
# Models — 잔차 학습 (Linear output, no sigmoid)
# ─────────────────────────────────────────────────────────────────────
class SOHMultiTaskLSTM(nn.Module):
    """
    LSTM × 2 + LayerNorm + Dropout
    출력: (delta_soh_norm, rate) — 모두 linear (잔차 공간)
    """
    def __init__(
        self,
        input_size: int = _INPUT_SIZE,
        hidden1: int = 128,
        hidden2: int = 64,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size, hidden1, batch_first=True, num_layers=1)
        self.ln1 = nn.LayerNorm(hidden1)
        self.drop1 = nn.Dropout(dropout)

        self.lstm2 = nn.LSTM(hidden1, hidden2, batch_first=True, num_layers=1)
        self.ln2 = nn.LayerNorm(hidden2)
        self.drop2 = nn.Dropout(dropout)

        self.fc_delta = nn.Linear(hidden2, 1)  # ΔSOH (잔차)
        self.fc_rate = nn.Linear(hidden2, 1)   # 보조 열화율

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.ln1(out)
        out = self.drop1(out)
        out, _ = self.lstm2(out)
        out = self.ln2(out)
        out = self.drop2(out)
        last = out[:, -1, :]
        delta = self.fc_delta(last).squeeze(-1)
        rate = self.fc_rate(last).squeeze(-1)
        return delta, rate


SOHLSTMModel = SOHMultiTaskLSTM
SOHTransformerModel = SOHMultiTaskLSTM  # legacy alias


class _LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 200, dropout: float = 0.1):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class SOHTransformerEncoder(nn.Module):
    """Pre-LN Transformer + Learnable PE + 잔차 출력"""
    def __init__(
        self,
        input_size: int = _INPUT_SIZE,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.3,
        max_len: int = 200,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_enc = _LearnablePositionalEncoding(d_model, max_len=max_len, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.ln = nn.LayerNorm(d_model)
        self.fc_delta = nn.Linear(d_model, 1)
        self.fc_rate = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.transformer(x)
        x = self.ln(x[:, -1, :])
        delta = self.fc_delta(x).squeeze(-1)
        rate = self.fc_rate(x).squeeze(-1)
        return delta, rate


# ─────────────────────────────────────────────────────────────────────
# Loss — 잔차(delta) 공간 기반
# ─────────────────────────────────────────────────────────────────────
class PhysicsInformedLoss(nn.Module):
    """
    Loss = MSE(pred_delta, target_delta)
         + λ_aux × MSE(pred_rate, target_rate)
         + λ_pinn × (Peukert + Arrhenius hinge)

    물리 제약:
      pred_delta는 음수(열화)가 자연스럽고, |delta| ∝ c_rate^n × arrhenius(T) 기대.
      위반 시 hinge penalty.
    """
    def __init__(self, lambda_pinn: float = 0.05, lambda_aux: float = 0.1):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lambda_pinn = lambda_pinn
        self.lambda_aux = lambda_aux

    def forward(
        self,
        pred_delta: torch.Tensor,
        target_delta: torch.Tensor,
        pred_rate: torch.Tensor = None,
        c_rate_batch: torch.Tensor = None,
        temp_batch: torch.Tensor = None,
        target_rate: torch.Tensor = None,
    ):
        data_loss = self.mse(pred_delta, target_delta)
        total = data_loss

        if self.lambda_aux > 0 and pred_rate is not None and target_rate is not None:
            aux_loss = self.mse(pred_rate, target_rate)
            total = total + self.lambda_aux * aux_loss

        if self.lambda_pinn > 0 and pred_rate is not None and c_rate_batch is not None:
            peukert_target = -0.001 * torch.pow(torch.clamp(c_rate_batch, min=1e-4), PEUKERT_N)
            arrhenius_target = -0.0001 * torch.exp(0.05 * (temp_batch - 25.0))
            peukert_violation = torch.mean(F.relu(pred_rate - peukert_target))
            arrhenius_violation = torch.mean(F.relu(pred_rate - arrhenius_target))
            total = total + self.lambda_pinn * (peukert_violation + arrhenius_violation)

        return total, data_loss


# ─────────────────────────────────────────────────────────────────────
# Data prep — 잔차 타겟 생성
# ─────────────────────────────────────────────────────────────────────
def _normalize_soh(soh: np.ndarray) -> np.ndarray:
    return (soh - SOH_MIN) / (SOH_MAX - SOH_MIN)


def _denormalize_soh(soh_norm: np.ndarray) -> np.ndarray:
    return soh_norm * (SOH_MAX - SOH_MIN) + SOH_MIN


def make_sequences(
    df: pd.DataFrame,
    feat_scaler: MinMaxScaler = None,
    return_aux: bool = False,
):
    """
    Window sequences with residual targets.

    Returns:
      X: (N, WINDOW, F+1) — 입력 윈도우 (마지막 채널 = soh_norm)
      y_delta: (N,) — 정규화된 ΔSOH 타겟 (soh_n[t] - soh_n[t-1])
      y_abs_norm: (N,) — 정규화된 절대 SOH 타겟 (재구성 용도)
      [선택] c_rate, temp, rate_target — PINN 입력
    """
    X_all, yd_all, ya_all = [], [], []
    c_all, t_all, r_all = [], [], []
    target_col = "soh_norm" if "soh_norm" in df.columns else "soh"

    for bat_id, group in df.groupby("battery_id"):
        group = group.sort_values("cycle").reset_index(drop=True)
        features = group[FEATURES].values
        soh = group[target_col].values
        soh_n = _normalize_soh(soh)

        if feat_scaler is not None:
            features = feat_scaler.transform(features)

        features = np.concatenate([features, soh_n.reshape(-1, 1)], axis=1)
        delta = np.diff(soh_n, prepend=soh_n[0])  # delta[i] = soh_n[i] - soh_n[i-1]

        c_rate = group["c_rate"].values
        temp = group["temp_mean"].values

        for i in range(WINDOW_SIZE, len(group)):
            X_all.append(features[i - WINDOW_SIZE : i])
            yd_all.append(delta[i])
            ya_all.append(soh_n[i])
            c_all.append(float(c_rate[i]))
            t_all.append(float(temp[i]))
            r_all.append(float(delta[i]))  # rate target = delta itself (보조)

    X_arr = np.array(X_all, dtype=np.float32)
    yd_arr = np.array(yd_all, dtype=np.float32)
    ya_arr = np.array(ya_all, dtype=np.float32)

    if return_aux:
        return (
            X_arr, yd_arr, ya_arr,
            np.array(c_all, dtype=np.float32),
            np.array(t_all, dtype=np.float32),
            np.array(r_all, dtype=np.float32),
        )
    return X_arr, yd_arr, ya_arr


# ─────────────────────────────────────────────────────────────────────
# Train / Eval
# ─────────────────────────────────────────────────────────────────────
def _build_model(model_type: str):
    if model_type == "transformer":
        return SOHTransformerEncoder()
    if model_type == "cde":
        from .soh_cde import NeuralCDEModel
        return NeuralCDEModel()
    return SOHMultiTaskLSTM()


def _reconstruct_soh(window_last_soh_n: torch.Tensor, pred_delta: torch.Tensor) -> torch.Tensor:
    """잔차 → 절대 SOH (정규화 공간)"""
    return torch.clamp(window_last_soh_n + pred_delta, 0.0, 1.0)


def train_model(
    train_df: pd.DataFrame,
    model_type: str = "lstm",
    save_dir: str = "models/saved",
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 5e-4,
    weight_decay: float = 1e-4,
    patience: int = 20,
    lambda_pinn: float = 0.05,
    lambda_aux: float = 0.1,
    verbose: bool = True,
):
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    scaler = MinMaxScaler()
    scaler.fit(train_df[FEATURES].values)
    joblib.dump(scaler, f"{save_dir}/feature_scaler_{model_type}.pkl")

    X, yd, ya, c_rate, temp, rate_t = make_sequences(train_df, scaler, return_aux=True)

    n_val = max(int(len(X) * 0.15), 1)
    Xtr, ydtr, yatr = X[:-n_val], yd[:-n_val], ya[:-n_val]
    ctr, ttr, rtr = c_rate[:-n_val], temp[:-n_val], rate_t[:-n_val]
    Xv, ydv, yav = X[-n_val:], yd[-n_val:], ya[-n_val:]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_model(model_type).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
    loss_fn = PhysicsInformedLoss(lambda_pinn=lambda_pinn, lambda_aux=lambda_aux)
    val_mse = nn.MSELoss()

    Xtr_t = torch.tensor(Xtr).to(device)
    ydtr_t = torch.tensor(ydtr).to(device)
    yatr_t = torch.tensor(yatr).to(device)
    ctr_t = torch.tensor(ctr).to(device)
    ttr_t = torch.tensor(ttr).to(device)
    rtr_t = torch.tensor(rtr).to(device)
    Xv_t = torch.tensor(Xv).to(device)
    yav_t = torch.tensor(yav).to(device)

    best_val = float("inf")
    no_improve = 0
    ckpt = f"{save_dir}/{model_type}_soh.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        idx = torch.randperm(len(Xtr_t))
        total_loss = 0.0
        total_data = 0.0
        for start in range(0, len(Xtr_t), batch_size):
            bi = idx[start:start + batch_size]
            xb = Xtr_t[bi]
            yd_b, ya_b = ydtr_t[bi], yatr_t[bi]
            cb, tb, rb = ctr_t[bi], ttr_t[bi], rtr_t[bi]

            pred_delta, pred_rate = model(xb)
            loss, data_loss = loss_fn(pred_delta, yd_b, pred_rate, cb, tb, rb)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * len(xb)
            total_data += data_loss.item() * len(xb)

        scheduler.step()

        # Val: 절대 SOH 공간에서 MSE 측정 (R²와 일관성)
        model.eval()
        with torch.no_grad():
            pred_delta_v, _ = model(Xv_t)
            soh_n_prev_v = Xv_t[:, -1, -1]
            pred_soh_n_v = _reconstruct_soh(soh_n_prev_v, pred_delta_v)
            val_loss = val_mse(pred_soh_n_v, yav_t).item()

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), ckpt)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                if verbose:
                    print(f"  [{model_type}] Early stop ep{epoch} (best val {best_val:.6f})")
                break

        if verbose and epoch % 10 == 0:
            tr_rmse = math.sqrt(total_data / len(Xtr_t)) * (SOH_MAX - SOH_MIN)
            v_rmse = math.sqrt(val_loss) * (SOH_MAX - SOH_MIN)
            print(f"  [{model_type}] Ep{epoch:3d} | train_ΔRMSE {tr_rmse:.3f}% | val_RMSE {v_rmse:.3f}% | lr {optimizer.param_groups[0]['lr']:.2e}")

    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False))
    return model


def train_lstm(train_df, save_dir="models/saved", epochs=100, batch_size=32, lr=5e-4, patience=20):
    model = train_model(train_df, model_type="lstm", save_dir=save_dir,
                        epochs=epochs, batch_size=batch_size, lr=lr, patience=patience)
    import shutil
    shutil.copy2(f"{save_dir}/feature_scaler_lstm.pkl", f"{save_dir}/feature_scaler.pkl")
    return model


def evaluate_model(test_df: pd.DataFrame, model_type: str = "lstm", save_dir: str = "models/saved"):
    scaler = joblib.load(f"{save_dir}/feature_scaler_{model_type}.pkl")
    X, _, y_abs_n = make_sequences(test_df, scaler)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_model(model_type).to(device)
    model.load_state_dict(torch.load(f"{save_dir}/{model_type}_soh.pt", map_location=device, weights_only=False))
    model.eval()

    X_t = torch.tensor(X).to(device)
    with torch.no_grad():
        pred_delta, _ = model(X_t)
        soh_n_prev = X_t[:, -1, -1]
        pred_soh_n = _reconstruct_soh(soh_n_prev, pred_delta).cpu().numpy()

    y_true = _denormalize_soh(y_abs_n)
    y_pred = _denormalize_soh(pred_soh_n)

    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    mae = float(np.mean(np.abs(y_pred - y_true)))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    return {"rmse": rmse, "mae": mae, "r2": r2, "y_true": y_true, "y_pred": y_pred}


def evaluate_lstm(test_df: pd.DataFrame, save_dir: str = "models/saved"):
    return evaluate_model(test_df, model_type="lstm", save_dir=save_dir)


# ─────────────────────────────────────────────────────────────────────
# MC Dropout
# ─────────────────────────────────────────────────────────────────────
def predict_with_uncertainty(
    X_tensor: torch.Tensor,
    model_type: str = "lstm",
    save_dir: str = "models/saved",
    n_samples: int = 50,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_model(model_type).to(device)
    model.load_state_dict(torch.load(f"{save_dir}/{model_type}_soh.pt", map_location=device, weights_only=False))
    model.train()  # dropout 활성화

    X = X_tensor.to(device)
    soh_n_prev = X[:, -1, -1]
    preds = []
    with torch.no_grad():
        for _ in range(n_samples):
            d, _ = model(X)
            preds.append(_reconstruct_soh(soh_n_prev, d).cpu().numpy())
    preds = np.stack(preds, axis=0)
    mean_n = preds.mean(axis=0)
    std_n = preds.std(axis=0)
    return _denormalize_soh(mean_n), std_n * (SOH_MAX - SOH_MIN)


# ─────────────────────────────────────────────────────────────────────
# SOH curve prediction (LSTM 앵커링)
# ─────────────────────────────────────────────────────────────────────
def _averaged_normalized_soh(train_df: pd.DataFrame, n_cycles: int) -> np.ndarray:
    target_x = np.arange(1, n_cycles + 1, dtype=float)
    curves = []
    target_col = "soh_norm" if "soh_norm" in train_df.columns else "soh"
    for _, group in train_df.groupby("battery_id"):
        group = group.sort_values("cycle").reset_index(drop=True)
        soh_raw = group[target_col].values
        if soh_raw[0] <= 0:
            continue
        soh_norm = soh_raw / soh_raw[0] * 100.0
        src_x = np.arange(1, len(soh_norm) + 1, dtype=float)
        interp = np.interp(target_x, src_x, soh_norm, left=soh_norm[0], right=soh_norm[-1])
        last_n = min(30, len(soh_norm) - 1)
        if last_n > 0:
            slope = (soh_norm[-1] - soh_norm[-last_n - 1]) / last_n
            beyond = target_x > src_x[-1]
            interp[beyond] = soh_norm[-1] + slope * (target_x[beyond] - src_x[-1])
        interp = np.clip(interp, 0.0, 100.0)
        curves.append(interp)
    return np.mean(curves, axis=0) if curves else np.full(n_cycles, 100.0)


def _lstm_current_soh(
    seed_df: pd.DataFrame,
    model_type: str = "lstm",
    save_dir: str = "models/saved",
    return_uncertainty: bool = False,
):
    try:
        scaler = joblib.load(f"{save_dir}/feature_scaler_{model_type}.pkl")
    except FileNotFoundError:
        return (None, None) if return_uncertainty else None

    g = seed_df.sort_values("cycle").reset_index(drop=True)
    if len(g) < WINDOW_SIZE or not all(f in g.columns for f in FEATURES):
        return (None, None) if return_uncertainty else None

    feat = scaler.transform(g[FEATURES].values[-WINDOW_SIZE:])
    target_col = "soh_norm" if "soh_norm" in g.columns else "soh"
    soh_n = _normalize_soh(g[target_col].values[-WINDOW_SIZE:])
    x = np.concatenate([feat, soh_n.reshape(-1, 1)], axis=1)
    x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(0)

    if return_uncertainty:
        mean_pct, std_pct = predict_with_uncertainty(x_t, model_type=model_type, save_dir=save_dir)
        return float(mean_pct[0]), float(std_pct[0])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_model(model_type).to(device)
    model.load_state_dict(torch.load(f"{save_dir}/{model_type}_soh.pt", map_location=device, weights_only=False))
    model.eval()
    with torch.no_grad():
        d, _ = model(x_t.to(device))
        soh_n_prev = x_t[:, -1, -1].to(device)
        pred_soh_n = _reconstruct_soh(soh_n_prev, d)
    return float(_denormalize_soh(pred_soh_n.cpu().numpy()[0]))


def _two_phase_soh(cycles: np.ndarray, alpha: float, eol_user: float) -> np.ndarray:
    """
    3단계 NCM 배터리 열화 모형 (문헌 기반):

    Phase 1 (SOH > 85%): Q(t) = Q0 - alpha·√t
      - SEI 확산 성장이 지배: 막이 두꺼워질수록 반응 차단 → 열화 감속
      - Ref: Ploehn et al. 2004 (J. Electrochem. Soc.)
             Safari & Delacourt 2011 (J. Electrochem. Soc.)

    Phase 2 (SOH ≤ 85%): dSOH/dt = -k·(100 - SOH)  → 지수 가속 (무릎 효과)
      - 리튬 석출(Li plating) + NCM 양극 입자 균열 누적 → 열화 재가속
      - 무릎 임계점(85% SOH): NCM 셀 전형값 (Attia et al. 2022, Nature Energy)
      - 지수 가속 형태: Cannarella & Arnold 2014 (J. Power Sources)
      - k는 EOL(SOH=70%)에서 곡선 수렴 조건으로 결정

    불확실성 (승법 노이즈):
      σ(t) ∝ √(t/EOL)
      - 배터리 상태가 나빠질수록 셀 간 분산 증가 → 예측 불확실성 증대
      - 개념: Saha & Goebel 2009 (IEEE Trans. Instrum. Meas.)
      - σ₀=1.2%: EVBattery 데이터 std(SOH)=4.27%에서 말기 불확실성 역산한 근사값
    """
    soh = 100.0 - alpha * np.sqrt(cycles)

    # 무릎 효과: SOH < 85% 구간에서 지수 가속
    # Attia et al. 2022: NCM 무릎 전형적으로 75~85% SOH
    KNEE_SOH = 85.0
    knee_mask = soh < KNEE_SOH
    if knee_mask.any():
        knee_idx   = int(np.argmax(knee_mask))
        knee_cycle = cycles[knee_idx]
        knee_soh   = soh[knee_idx]

        remaining = eol_user - knee_cycle
        if remaining > 0:
            # k: SOH=70%(EOL)에서 수렴하도록 역산
            k = np.log(max((100.0 - 70.0) / max(100.0 - knee_soh, 1e-4), 1.001)) / remaining
            future = cycles[knee_mask] - knee_cycle
            soh[knee_mask] = 100.0 - (100.0 - knee_soh) * np.exp(k * future)

    soh = soh.clip(0.0, 100.0)

    # 승법 노이즈 σ(t) = 1.2 × √(t/EOL)
    sigma = 1.2 * np.sqrt(np.clip(cycles / eol_user, 0, 1))
    return soh, sigma


def predict_soh_curve(
    seed_df: pd.DataFrame,
    n_future_cycles: int = 500,
    c_rate: float = BASE_C_RATE,
    temp_celsius: float = 25.0,
    save_dir: str = "models/saved",
    train_df: pd.DataFrame = None,
    model_type: str = "lstm",
    return_uncertainty: bool = False,
) -> np.ndarray:
    """
    Hybrid SOH 곡선 (3단계 열화 + 승법 노이즈):
      1. Peukert + Arrhenius → 사용자 조건 EOL cycle 계산
      2. Phase1(√t) + Phase2(선형) + Phase3(지수 가속, 무릎 효과)
      3. 승법 노이즈: σ(t) ∝ √t/EOL → 쓸수록 신뢰구간 확대
      4. LSTM 앵커: 현재 SOH로 곡선 시프트
    """
    EOL_AT_BASE = 2000.0
    peukert_factor = (BASE_C_RATE / max(c_rate, 1e-4)) ** PEUKERT_N
    arrhenius_mult = arrhenius_multiplier(temp_celsius)
    eol_user = EOL_AT_BASE * peukert_factor / max(arrhenius_mult, 1e-4)
    eol_user = max(eol_user, 100.0)

    alpha  = 30.0 / np.sqrt(eol_user)
    cycles = np.arange(1, n_future_cycles + 1, dtype=float)
    soh_curve, sigma = _two_phase_soh(cycles, alpha, eol_user)

    # LSTM 앵커
    if return_uncertainty:
        anchor_soh, anchor_std = _lstm_current_soh(
            seed_df, model_type=model_type, save_dir=save_dir, return_uncertainty=True
        )
    else:
        anchor_soh = _lstm_current_soh(seed_df, model_type=model_type, save_dir=save_dir)
        anchor_std = None

    if anchor_soh is not None and anchor_soh < 100.0:
        delta_soh  = max(0.0, 100.0 - anchor_soh)
        equiv_cycle = (delta_soh / alpha) ** 2
        soh_curve, sigma = _two_phase_soh(cycles + equiv_cycle, alpha, eol_user)

    if return_uncertainty:
        # anchor MC Dropout 불확실성 + 승법 시간 불확실성 합산
        combined_std = np.sqrt((anchor_std or 0.0) ** 2 + sigma ** 2)
        return soh_curve, combined_std
    return soh_curve
