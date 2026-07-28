"""Minimal self-contained TimesNet (https://github.com/thuml/TimesNet).

Vendored from the notebook experiment (damin/notebooks/TimesNet_test.ipynb) that trained
the bundled checkpoints, so module/attribute names match the saved `state_dict` exactly.
Time-feature (month/day/weekday/hour) embeddings are omitted, matching the simplification
already used for `backend/models/itransformer.py`'s `ITransformerNet` — value + position
embedding only.

Core idea: a 1D series mixes several periods. FFT picks the top-k dominant frequencies,
each period reshapes the sequence into 2D (rows = across-period, cols = within-period),
an Inception-style 2D conv extracts features per period, and an amplitude-weighted sum
folds the periods back into 1D (TimesBlock). The forecast head extends the time axis from
seq_len to seq_len+pred_len with a linear layer before the TimesBlocks run (as in the
reference implementation).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, n_vars, d_model):
        super().__init__()
        self.conv = nn.Conv1d(n_vars, d_model, kernel_size=3, padding=1, padding_mode="circular", bias=False)
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_in", nonlinearity="leaky_relu")

    def forward(self, x):
        # x: (batch, seq_len, n_vars) -> (batch, seq_len, d_model)
        return self.conv(x.permute(0, 2, 1)).transpose(1, 2)


class DataEmbedding(nn.Module):
    def __init__(self, n_vars, d_model, dropout=0.1):
        super().__init__()
        self.value_embedding = TokenEmbedding(n_vars, d_model)
        self.position_embedding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.value_embedding(x) + self.position_embedding(x))


def FFT_for_period(x, k=5):
    """x: (batch, T, d_model). Top-k dominant frequencies -> period (T//freq) and amplitude
    (used as the blending weight)."""
    xf = torch.fft.rfft(x, dim=1)
    amp = abs(xf).mean(0).mean(-1)  # (freq,)
    amp[0] = 0  # exclude the DC (trend) component
    _, top_idx = torch.topk(amp, k)
    top_idx = top_idx.detach().cpu().numpy()
    period = np.clip(x.shape[1] // top_idx, 1, None)  # guard against a zero period
    period_weight = abs(xf).mean(-1)[:, top_idx]  # (batch, k)
    return period, period_weight


class InceptionBlock(nn.Module):
    """Averages conv2d kernels of different sizes -- captures within/across-period
    patterns at multiple receptive fields."""
    def __init__(self, in_ch, out_ch, num_kernels=6):
        super().__init__()
        self.kernels = nn.ModuleList([
            nn.Conv2d(in_ch, out_ch, kernel_size=2 * i + 1, padding=i) for i in range(num_kernels)
        ])
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        return torch.stack([k(x) for k in self.kernels], dim=-1).mean(-1)


class TimesBlock(nn.Module):
    def __init__(self, seq_len, pred_len, d_model, d_ff, top_k, num_kernels=6):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.k = top_k
        self.conv = nn.Sequential(
            InceptionBlock(d_model, d_ff, num_kernels),
            nn.GELU(),
            InceptionBlock(d_ff, d_model, num_kernels),
        )

    def forward(self, x):
        # x: (batch, seq_len+pred_len, d_model)
        B, T, D = x.shape
        period_list, period_weight = FFT_for_period(x, self.k)

        res = []
        total = self.seq_len + self.pred_len
        for i in range(self.k):
            period = int(period_list[i])
            if total % period != 0:
                length = ((total // period) + 1) * period
                pad = torch.zeros(B, length - total, D, device=x.device, dtype=x.dtype)
                out = torch.cat([x, pad], dim=1)
            else:
                length = total
                out = x
            # 1D -> 2D: (batch, d_model, length//period, period) -- across-period x within-period
            out = out.reshape(B, length // period, period, D).permute(0, 3, 1, 2).contiguous()
            out = self.conv(out)
            out = out.permute(0, 2, 3, 1).reshape(B, -1, D)
            res.append(out[:, :total, :])
        res = torch.stack(res, dim=-1)  # (B, T, D, k)

        weight = F.softmax(period_weight, dim=1)  # (B, k) -- larger amplitude, more trust
        weight = weight.unsqueeze(1).unsqueeze(1).repeat(1, T, D, 1)
        res = torch.sum(res * weight, dim=-1)
        return res + x  # residual


class TimesNet(nn.Module):
    def __init__(self, seq_len, pred_len, n_vars, d_model=16, d_ff=32,
                 e_layers=2, top_k=5, num_kernels=6, dropout=0.1, **_):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.embedding = DataEmbedding(n_vars, d_model, dropout)
        self.predict_linear = nn.Linear(seq_len, seq_len + pred_len)
        self.blocks = nn.ModuleList([
            TimesBlock(seq_len, pred_len, d_model, d_ff, top_k, num_kernels) for _ in range(e_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.projection = nn.Linear(d_model, n_vars)

    def forward(self, x):
        # x: (batch, seq_len, n_vars) -> out: (batch, pred_len, n_vars), raw units.
        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x = x / stdev

        enc_out = self.embedding(x)  # (B, seq_len, d_model)
        enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(0, 2, 1)  # (B, seq_len+pred_len, d_model)
        for block in self.blocks:
            enc_out = self.norm(block(enc_out))
        dec_out = self.projection(enc_out)  # (B, seq_len+pred_len, n_vars)

        dec_out = dec_out * stdev.repeat(1, self.seq_len + self.pred_len, 1)
        dec_out = dec_out + means.repeat(1, self.seq_len + self.pred_len, 1)
        return dec_out[:, -self.pred_len:, :]
