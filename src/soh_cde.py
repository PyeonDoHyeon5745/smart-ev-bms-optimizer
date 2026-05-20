"""
Neural Controlled Differential Equation (Neural CDE) for EV Battery SOH
────────────────────────────────────────────────────────────────────────
Reference: Kidger et al. 2020
  "Neural Controlled Differential Equations for Irregular Time Series"
  NeurIPS 2020  |  https://arxiv.org/abs/2005.08926

수식: dZ(t) = f_θ(Z(t)) · dX(t),   t ∈ [0, 1]

LSTM 과 같은 FEATURES 14채널 + WINDOW_SIZE=30 사용.
train.py Step 6에서 호출. 4-way 비교에 참여.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torchcde
import joblib
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

from .soh_lstm import (
    FEATURES, WINDOW_SIZE, SOH_MIN, SOH_MAX,
    _normalize_soh, _denormalize_soh,
    PhysicsInformedLoss,
)

_INPUT_CHANNELS = len(FEATURES) + 1   # 13 피처 + soh_norm = 14
_HIDDEN_DIM     = 64

# 모든 윈도우에서 공유하는 [0, 1] 시간 격자
_T_GRID = torch.linspace(0.0, 1.0, WINDOW_SIZE)


# ─────────────────────────────────────────────────────────────────────
# 벡터장 f_θ
# ─────────────────────────────────────────────────────────────────────
class _CDEFunc(nn.Module):
    """
    f_θ : (batch, hidden) → (batch, hidden, input_channels)
    dZ_i = Σ_j f_{ij}(Z) · dX_j
    """
    def __init__(self, hidden_dim: int, input_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.Tanh(),
            nn.Linear(128, hidden_dim * input_channels),
            nn.Tanh(),
        )
        self.hidden_dim     = hidden_dim
        self.input_channels = input_channels

    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        out = self.net(z)
        return out.view(*z.shape[:-1], self.hidden_dim, self.input_channels)


# ─────────────────────────────────────────────────────────────────────
# Neural CDE 모델
# ─────────────────────────────────────────────────────────────────────
class NeuralCDEModel(nn.Module):
    """
    Neural CDE SOH 예측 모델.

    LSTM 과 동일한 입출력:
      forward(X) → (delta_soh_norm, rate)   — 잔차 공간 출력

    X: (batch, WINDOW_SIZE, input_channels)
       윈도우 내 30개 세션을 자연 삼차 스플라인으로 연속 경로 보간 후 CDE 적분.
    """
    def __init__(
        self,
        input_channels: int = _INPUT_CHANNELS,
        hidden_dim: int     = _HIDDEN_DIM,
    ):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Linear(input_channels, hidden_dim),
            nn.Tanh(),
        )
        self.func     = _CDEFunc(hidden_dim, input_channels)
        self.fc_delta = nn.Linear(hidden_dim, 1)
        self.fc_rate  = nn.Linear(hidden_dim, 1)

    def forward(self, X: torch.Tensor):
        device = X.device
        t = _T_GRID.to(device)

        # 자연 삼차 스플라인 → 연속 경로 X(t)
        coeffs   = torchcde.natural_cubic_coeffs(X, t=t)
        X_spline = torchcde.CubicSpline(coeffs)

        # z_0 = tanh(W · X(0))
        z0 = self.initial(X_spline.evaluate(t[0]))   # (batch, hidden)

        # CDE 적분: t=0 → t=1
        # adjoint=False: 직접 autograd 사용 (adjoint 역전파 수치 불안정 방지)
        # method=rk4 + 고정 스텝: adaptive solver 발산 방지
        t_eval = t[[0, -1]]                           # [0.0, 1.0]
        z = torchcde.cdeint(
            X=X_spline,
            func=self.func,
            z0=z0,
            t=t_eval,
            adjoint=False,
            method="rk4",
            options={"step_size": 1.0 / (WINDOW_SIZE - 1)},
        )                                             # (batch, 2, hidden)

        z_T   = z[:, -1]                             # 최종 상태
        delta = self.fc_delta(z_T).squeeze(-1)
        rate  = self.fc_rate(z_T).squeeze(-1)
        return delta, rate


# ─────────────────────────────────────────────────────────────────────
# 시퀀스 생성 (soh_lstm.make_sequences 와 동일 구조)
# ─────────────────────────────────────────────────────────────────────
def make_cde_sequences(df, feat_scaler=None, return_aux: bool = False):
    X_all, yd_all, ya_all = [], [], []
    c_all, t_all, r_all   = [], [], []
    target_col = "soh_norm" if "soh_norm" in df.columns else "soh"

    for bat_id, group in df.groupby("battery_id"):
        group    = group.sort_values("cycle").reset_index(drop=True)
        features = group[FEATURES].values
        soh      = group[target_col].values
        soh_n    = _normalize_soh(soh)

        if feat_scaler is not None:
            features = feat_scaler.transform(features)

        features = np.concatenate([features, soh_n.reshape(-1, 1)], axis=1)
        delta    = np.diff(soh_n, prepend=soh_n[0])
        c_rate   = group["c_rate"].values
        temp     = group["temp_mean"].values

        for i in range(WINDOW_SIZE, len(group)):
            X_all.append(features[i - WINDOW_SIZE : i])
            yd_all.append(delta[i])
            ya_all.append(soh_n[i])
            c_all.append(float(c_rate[i]))
            t_all.append(float(temp[i]))
            r_all.append(float(delta[i]))

    X_arr  = np.array(X_all,  dtype=np.float32)
    yd_arr = np.array(yd_all, dtype=np.float32)
    ya_arr = np.array(ya_all, dtype=np.float32)

    if return_aux:
        return (X_arr, yd_arr, ya_arr,
                np.array(c_all, dtype=np.float32),
                np.array(t_all, dtype=np.float32),
                np.array(r_all, dtype=np.float32))
    return X_arr, yd_arr, ya_arr


def _reconstruct_soh(soh_n_prev: torch.Tensor, pred_delta: torch.Tensor) -> torch.Tensor:
    return torch.clamp(soh_n_prev + pred_delta, 0.0, 1.0)


# ─────────────────────────────────────────────────────────────────────
# 학습
# ─────────────────────────────────────────────────────────────────────
def train_cde(
    train_df,
    save_dir: str       = "models/saved",
    epochs: int         = 100,
    batch_size: int     = 64,
    lr: float           = 5e-4,
    weight_decay: float = 1e-4,
    patience: int       = 20,
    lambda_pinn: float  = 0.05,
    lambda_aux: float   = 0.1,
    verbose: bool       = True,
):
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    scaler = MinMaxScaler()
    scaler.fit(train_df[FEATURES].values)
    joblib.dump(scaler, f"{save_dir}/feature_scaler_cde.pkl")

    X, yd, ya, c_rate, temp, rate_t = make_cde_sequences(train_df, scaler, return_aux=True)

    n_val  = max(int(len(X) * 0.15), 1)
    Xtr,  ydtr,  yatr  = X[:-n_val],  yd[:-n_val],  ya[:-n_val]
    ctr,  ttr,   rtr   = c_rate[:-n_val], temp[:-n_val], rate_t[:-n_val]
    Xv,   ydv,   yav   = X[-n_val:],  yd[-n_val:],  ya[-n_val:]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = NeuralCDEModel().to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
    loss_fn   = PhysicsInformedLoss(lambda_pinn=lambda_pinn, lambda_aux=lambda_aux)
    val_mse   = nn.MSELoss()

    Xtr_t  = torch.tensor(Xtr).to(device)
    ydtr_t = torch.tensor(ydtr).to(device)
    ctr_t  = torch.tensor(ctr).to(device)
    ttr_t  = torch.tensor(ttr).to(device)
    rtr_t  = torch.tensor(rtr).to(device)
    Xv_t   = torch.tensor(Xv).to(device)
    yav_t  = torch.tensor(yav).to(device)

    best_val   = float("inf")
    no_improve = 0
    ckpt       = f"{save_dir}/cde_soh.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        idx        = torch.randperm(len(Xtr_t))
        total_loss = 0.0
        total_data = 0.0

        for start in range(0, len(Xtr_t), batch_size):
            bi  = idx[start : start + batch_size]
            xb  = Xtr_t[bi]
            yd_b = ydtr_t[bi]
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

        model.eval()
        with torch.no_grad():
            pred_delta_v, _ = model(Xv_t)
            soh_n_prev_v    = Xv_t[:, -1, -1]
            pred_soh_n_v    = _reconstruct_soh(soh_n_prev_v, pred_delta_v)
            val_loss        = val_mse(pred_soh_n_v, yav_t).item()

        if val_loss < best_val:
            best_val   = val_loss
            torch.save(model.state_dict(), ckpt)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                if verbose:
                    print(f"  [CDE] Early stop ep{epoch}  (best val {best_val:.6f})")
                break

        if verbose and epoch % 10 == 0:
            tr_rmse = math.sqrt(total_data / len(Xtr_t)) * (SOH_MAX - SOH_MIN)
            v_rmse  = math.sqrt(val_loss)               * (SOH_MAX - SOH_MIN)
            print(
                f"  [CDE] Ep{epoch:3d} | "
                f"train_ΔRMSE {tr_rmse:.3f}% | val_RMSE {v_rmse:.3f}% | "
                f"lr {optimizer.param_groups[0]['lr']:.2e}"
            )

    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False))
    return model


# ─────────────────────────────────────────────────────────────────────
# 평가
# ─────────────────────────────────────────────────────────────────────
def evaluate_cde(test_df, save_dir: str = "models/saved"):
    scaler = joblib.load(f"{save_dir}/feature_scaler_cde.pkl")
    X, _, y_abs_n = make_cde_sequences(test_df, scaler)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = NeuralCDEModel().to(device)
    model.load_state_dict(torch.load(f"{save_dir}/cde_soh.pt", map_location=device, weights_only=False))
    model.eval()

    X_t = torch.tensor(X).to(device)
    with torch.no_grad():
        pred_delta, _ = model(X_t)
        soh_n_prev    = X_t[:, -1, -1]
        pred_soh_n    = _reconstruct_soh(soh_n_prev, pred_delta).cpu().numpy()

    y_true = _denormalize_soh(y_abs_n)
    y_pred = _denormalize_soh(pred_soh_n)

    rmse   = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    mae    = float(np.mean(np.abs(y_pred - y_true)))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2     = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    return {"rmse": rmse, "mae": mae, "r2": r2, "y_true": y_true, "y_pred": y_pred}
