from fastapi import FastAPI
import uvicorn
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import json, os

app = FastAPI()

# ── Constants ──
SEQ_LEN   = 168
PRED_LEN  = 24
TRAIN_END = "2013-12-31 23:00"
TEST_START = "2014-01-01 00:00"
TEST_END   = "2014-12-31 23:00"
CLUSTER_ID = 3
MODEL_DIR  = "./models/cluster_3"
DATA_PATH  = "./electricity.txt"
device     = torch.device("cpu")

# ── Paste your full PatchTST class definition here from Cell 53 ──
class PatchTST(nn.Module):
    """
    Channel-independent Patch Time Series Transformer.
    Nie et al., ICLR 2023  (https://github.com/yuqinie98/PatchTST)

    Input  (B, M, L)  →  output  (B, M, T)
      B = batch size,  M = number of channels (clients),
      L = look-back window,  T = forecast horizon

    Steps (Figure 1b of the paper):
      1. Instance norm per channel
      2. Reshape (B,M,L) → (B*M, L)   [channel-independence]
      3. Pad stride copies of last value → unfold into patches  (B*M, N, P)
      4. Linear projection + learnable positional encoding  (B*M, N, D)
      5. Transformer encoder
      6. Flatten + linear head  (B*M, T)
      7. Reshape → (B, M, T)  then denorm
    """

    def __init__(
        self,
        seq_len   = 96,    # look-back window L
        pred_len  = 24,    # forecast horizon T
        patch_len = 16,    # patch length P
        stride    =  8,    # stride S  →  N = floor((L-P)/S)+2 = 12 patches
        d_model   = 128,   # hidden dim D
        n_heads   =   8,   # attention heads  (d_model must be divisible by n_heads)
        e_layers  =   3,   # encoder layers
        d_ff      = 256,   # feedforward inner dim
        dropout   = 0.2,
    ):
        super().__init__()
        self.seq_len   = seq_len
        self.pred_len  = pred_len
        self.patch_len = patch_len
        self.stride    = stride
        # N = floor((L-P)/S) + 2  (paper formula, pads stride values to end before patching)
        self.n_patches = (seq_len - patch_len) // stride + 2

        # Patch projection + learnable positional encoding
        self.patch_proj = nn.Linear(patch_len, d_model)
        self.pos_enc    = nn.Parameter(torch.randn(1, self.n_patches, d_model) * 0.02)
        self.in_dropout = nn.Dropout(dropout)

        # Transformer encoder (standard PyTorch, batch_first=True)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_ff, dropout=dropout,
            batch_first=True, norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=e_layers)

        # Flatten → linear head
        self.head = nn.Linear(self.n_patches * d_model, pred_len)

    def _instance_norm(self, x):
        """Normalize each (batch, channel) series: zero mean, unit std."""
        mean = x.mean(dim=-1, keepdim=True)         # (B, M, 1)
        std  = x.std(dim=-1, keepdim=True) + 1e-8  # (B, M, 1)
        return (x - mean) / std, mean, std

    def _make_patches(self, x):
        """Pad stride copies of last value then unfold into patches.
        x: (B*M, L)  →  (B*M, n_patches, patch_len)"""
        pad = x[:, -1:].expand(-1, self.stride)           # (B*M, stride)
        x   = torch.cat([x, pad], dim=-1)                 # (B*M, L + stride)
        return x.unfold(-1, self.patch_len, self.stride)  # (B*M, n_patches, patch_len)

    def forward(self, x):
        """x: (B, M, L)  →  (B, M, pred_len)"""
        B, M, L = x.shape

        # 1. Instance normalisation per channel
        x, mean, std = self._instance_norm(x)

        # 2. Channel-independent: (B, M, L) → (B*M, L)
        x = x.reshape(B * M, L)

        # 3. Patch: (B*M, n_patches, patch_len)
        x = self._make_patches(x)

        # 4. Project + pos enc: (B*M, n_patches, d_model)
        x = self.in_dropout(self.patch_proj(x) + self.pos_enc)

        # 5. Transformer encoder
        x = self.transformer(x)     # (B*M, n_patches, d_model)

        # 6. Flatten + head: (B*M, pred_len)
        x = self.head(x.flatten(1))

        # 7. Reshape + denorm: (B, M, pred_len)
        return x.reshape(B, M, -1) * std + mean

# ── Load model once at startup ──
checkpoint = torch.load(f"{MODEL_DIR}/patchtst.pt", map_location=device)
cfg        = checkpoint["config"]
model      = PatchTST(**cfg)
model.load_state_dict(checkpoint["state_dict"])
model.eval()

with open(f"{MODEL_DIR}/clients.json") as f:
    cluster_members = json.load(f)

df_raw    = pd.read_csv(DATA_PATH, sep=";", decimal=",", index_col=0, parse_dates=True)
df_hourly = df_raw.resample("h").mean().ffill(limit=3).dropna(axis=1)
train_mat = df_hourly[cluster_members].loc[:TRAIN_END].values
test_mat  = df_hourly[cluster_members].loc[TEST_START:TEST_END].values
train_idx = df_hourly.loc[:TRAIN_END].index
test_idx  = df_hourly.loc[TEST_START:TEST_END].index
full_mat  = np.vstack([train_mat, test_mat])
full_index = train_idx.append(test_idx)

# ── Forecast endpoint ──
@app.post("/forecast")
def forecast(item: dict):
    client_id = item["client_id"]

    # Use end of training data as the forecast starting point
    end_ts = pd.Timestamp(TRAIN_END)
    pos    = int(full_index.searchsorted(end_ts, side="right")) - 1
    start  = pos - SEQ_LEN + 1

    window = full_mat[start: pos + 1]
    x = torch.FloatTensor(window.T).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(x)  # (1, M, pred_len)

    forecast_values = pred[0].mean(dim=0).cpu().numpy().tolist()

    # Build hourly timestamps for 2014
    forecast_index = pd.date_range(start=TEST_START, periods=PRED_LEN, freq="h")

    return {
        "client_id": client_id,
        "cluster": CLUSTER_ID,
        "model": "PatchTST",
        "forecast_start": TEST_START,
        "forecast_hours": forecast_index.strftime("%Y-%m-%d %H:%M").tolist(),
        "forecast_values": forecast_values
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5004)