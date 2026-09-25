"""LSTM-AD: next-step multivariate forecasting residual.

Train an LSTM to predict ``x[t]`` from ``x[t-w:t]``. Anomaly score is
``−MSE`` (lower = worse). Threshold = train residual percentile.
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "LSTM-AD requires torch. Install OmniAnomaly requirements / torch first."
    ) from exc


class _LSTMForecaster(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64, layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=0.1 if layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden, n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, W, D)
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


def _make_windows(
    x: np.ndarray, window: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return (inputs W, targets) with target = next step after window."""
    x = np.asarray(x, dtype=np.float32)
    n, d = x.shape
    if n <= window:
        raise ValueError(f"need len>{window}, got {n}")
    xs, ys = [], []
    for i in range(window, n):
        xs.append(x[i - window : i])
        ys.append(x[i])
    return np.stack(xs), np.stack(ys)


def train_lstm_ad(
    train: np.ndarray,
    *,
    window: int = 32,
    hidden: int = 64,
    layers: int = 2,
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str | None = None,
) -> dict[str, Any]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    x = np.asarray(train, dtype=np.float32)
    xs, ys = _make_windows(x, window)
    ds = TensorDataset(torch.from_numpy(xs), torch.from_numpy(ys))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)
    model = _LSTMForecaster(x.shape[1], hidden=hidden, layers=layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    model.train()
    history: list[float] = []
    for ep in range(int(epochs)):
        total = 0.0
        n = 0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(xb)
            n += len(xb)
        history.append(total / max(n, 1))
        print(f"  LSTM-AD epoch {ep + 1}/{epochs}  loss={history[-1]:.6f}", flush=True)
    return {
        "model": model,
        "device": device,
        "window": int(window),
        "history": history,
        "n_features": int(x.shape[1]),
    }


@torch.no_grad()
def predict_mse(
    bundle: dict[str, Any],
    series: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (mse_per_step length n, aligned boolean valid_mask).

    First ``window`` steps have no score (nan / False).
    """
    model: _LSTMForecaster = bundle["model"]
    device = bundle["device"]
    window = int(bundle["window"])
    x = np.asarray(series, dtype=np.float32)
    n, d = x.shape
    mse = np.full(n, np.nan, dtype=np.float64)
    if n <= window:
        return mse, np.zeros(n, dtype=bool)
    xs, ys = _make_windows(x, window)
    model.eval()
    # batch for speed
    bs = 256
    preds = []
    for i in range(0, len(xs), bs):
        xb = torch.from_numpy(xs[i : i + bs]).to(device)
        preds.append(model(xb).cpu().numpy())
    pred = np.concatenate(preds, axis=0)
    err = ((pred - ys) ** 2).mean(axis=1)
    # err[j] corresponds to time index window+j
    mse[window:] = err
    ok = np.isfinite(mse)
    return mse, ok


def scores_from_mse(mse: np.ndarray) -> np.ndarray:
    """UI polarity: lower = more anomalous → ``−mse``."""
    out = np.full_like(mse, np.nan, dtype=np.float64)
    ok = np.isfinite(mse)
    out[ok] = -mse[ok]
    return out
