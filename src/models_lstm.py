"""
LSTM Recurrent Neural Network for 24-Hour Smart Grid Power Demand Forecasting.

Implements an encoder-decoder sequence model in PyTorch:
- Input: Sequence of past L=24 hours of grid state (scaled demand, wind, solar, and calendar harmonics)
- Recurrent Encoder: 2-layer LSTM with dropout
- Output Projection: Dense multi-horizon decoder mapping hidden representation to all 24 forward hours
- Target Scaling: Normalized to zero mean and unit variance during training, inversely scaled for evaluation
"""

from typing import Dict, Any, Tuple
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

from src.config import MODELS_DIR, FORECAST_HORIZON_HOURS, COL_TIMESTAMP, COL_DEMAND
from src.features import create_forecasting_dataset, split_chronological_data
from src.evaluate import calculate_metrics, analyze_peak_demand_errors


class TimeSeriesSequenceDataset(Dataset):
    """PyTorch Dataset yielding (features_sequence, 24h_targets)."""
    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int = 24):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.X) - self.seq_len + 1

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x_seq = self.X[idx : idx + self.seq_len]  # Shape: [seq_len, num_features]
        # Target corresponding to the last step of the sequence (origin t)
        target = self.y[idx + self.seq_len - 1]     # Shape: [24]
        return x_seq, target


class LSTM24hForecaster(nn.Module):
    """2-Layer LSTM with Dense Multi-Horizon 24-step Output Head."""
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, horizon: int = 24, dropout: float = 0.15):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, seq_len, input_dim]
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]  # [batch_size, hidden_dim]
        out = self.fc(last_hidden)        # [batch_size, horizon=24]
        return out


def train_lstm_model(
    splits: Dict[str, Any],
    seq_len: int = 24,
    epochs: int = 25,
    batch_size: int = 64,
    lr: float = 0.003,
    save_model: bool = True,
) -> Tuple[Dict[str, Any], np.ndarray]:
    """
    Trains the LSTM model with early stopping on the validation set.
    Evaluates on both validation and untouched test sets.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")

    X_train_raw = splits["X_train"]
    y_train_raw = splits["y_train"]
    X_val_raw = splits["X_val"]
    y_val_raw = splits["y_val"]
    X_test_raw = splits["X_test"]
    y_test_raw = splits["y_test"]

    # Target Scaler fitted STRICTLY on y_train
    target_scaler = StandardScaler()
    y_train_scaled = target_scaler.fit_transform(y_train_raw)
    y_val_scaled = target_scaler.transform(y_val_raw)
    y_test_scaled = target_scaler.transform(y_test_raw)

    train_ds = TimeSeriesSequenceDataset(X_train_raw, y_train_scaled, seq_len=seq_len)
    val_ds = TimeSeriesSequenceDataset(X_val_raw, y_val_scaled, seq_len=seq_len)
    test_ds = TimeSeriesSequenceDataset(X_test_raw, y_test_scaled, seq_len=seq_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    input_dim = X_train_raw.shape[1]
    model = LSTM24hForecaster(input_dim=input_dim, hidden_dim=64, num_layers=2, horizon=FORECAST_HORIZON_HOURS).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    best_val_loss = float("inf")
    best_weights_path = MODELS_DIR / "lstm_demand_forecaster.pt"

    print("Training PyTorch LSTM Forecaster...")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            preds = model(x_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * len(x_batch)
        train_loss /= len(train_ds)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                preds = model(x_batch)
                loss = criterion(preds, y_batch)
                val_loss += loss.item() * len(x_batch)
        val_loss /= len(val_ds)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if save_model:
                torch.save(model.state_dict(), best_weights_path)

        if epoch % 5 == 0 or epoch == epochs:
            print(f"  Epoch [{epoch:02d}/{epochs:02d}] - Train Loss (MSE): {train_loss:.4f} | Val Loss (MSE): {val_loss:.4f}")

    # Load best weights
    model.load_state_dict(torch.load(best_weights_path, map_location=device))
    model.eval()

    def predict_dataset(loader, true_raw, ds):
        preds_scaled = []
        with torch.no_grad():
            for x_batch, _ in loader:
                preds_batch = model(x_batch.to(device))
                preds_scaled.append(preds_batch.cpu().numpy())
        preds_scaled = np.vstack(preds_scaled)
        preds_orig = target_scaler.inverse_transform(preds_scaled)
        # Matching ground truth slice corresponding to index >= seq_len - 1
        true_slice = true_raw[seq_len - 1 :]
        return preds_orig, true_slice

    val_preds_orig, y_val_eval = predict_dataset(val_loader, y_val_raw, val_ds)
    test_preds_orig, y_test_eval = predict_dataset(test_loader, y_test_raw, test_ds)

    results = {
        "LSTM": {
            "val_metrics": calculate_metrics(y_val_eval, val_preds_orig),
            "val_peak": analyze_peak_demand_errors(y_val_eval, val_preds_orig),
            "test_metrics": calculate_metrics(y_test_eval, test_preds_orig),
            "test_peak": analyze_peak_demand_errors(y_test_eval, test_preds_orig),
            "eval_offset": seq_len - 1,
        }
    }

    return results, test_preds_orig


if __name__ == "__main__":
    df_feat, feats, targets = create_forecasting_dataset()
    splits = split_chronological_data(df_feat, feats, targets)
    lstm_res, lstm_preds = train_lstm_model(splits, epochs=15)
    print("LSTM Test Results:", lstm_res["LSTM"]["test_metrics"])
    print("LSTM Peak Analysis:", lstm_res["LSTM"]["test_peak"])
