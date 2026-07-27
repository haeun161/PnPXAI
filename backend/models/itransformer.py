"""Minimal self-contained iTransformer (https://arxiv.org/abs/2310.06625).

Vendored from the reference implementation in model_test/iTransformer so the backend
doesn't need that tree on sys.path — its `layers.SelfAttention_Family` pulls in
`reformer_pytorch`, which isn't installed (and is only needed by other model variants).
Only the modules the ETTh1_96_96 checkpoint actually uses are kept; module and
attribute names are identical to the reference so `state_dict` keys load unchanged.

The architecture is "inverted": each *variate* becomes a token whose embedding is a
linear map over the whole time axis. So `seq_len`/`pred_len` are fixed by the weights
(96/96 here) but the number of channels is not — the same weights run on any channel
count, and timestamp covariates simply appear as extra tokens.
"""
from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F


class _EncoderLayer(nn.Module):
    """Post-norm block over variate tokens, using fused-QKV MultiheadAttention."""

    def __init__(self, d_model, n_heads, d_ff, dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.norm1(x + self.attn(x, x, x, need_weights=False)[0])
        return self.norm2(x + self.ff(x))


class ITransformerNet(nn.Module):
    """Config-driven iTransformer matching checkpoints saved as {"state_dict", "config"}.

    Same inverted idea as `Model` above — one token per variate, embedded across the whole
    time axis — but a different implementation lineage: fused-QKV `nn.MultiheadAttention`
    and reversible instance normalization (RevIN) around the network, with no timestamp
    covariates. Layer shapes all come from the checkpoint's own `config`.

    RevIN here is non-affine: the checkpoint carries no scale/shift parameters for it, so
    normalization is plain per-instance mean/std removal, undone on the output.
    """

    def __init__(self, seq_len, pred_len, d_model, d_ff, e_layers, n_heads, **_):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.embedding = nn.Module()
        self.embedding.proj = nn.Linear(seq_len, d_model)
        self.layers = nn.ModuleList(
            [_EncoderLayer(d_model, n_heads, d_ff) for _ in range(e_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.projection = nn.Linear(d_model, pred_len)

    def forward(self, x):
        """x: (batch, seq_len, channels) -> (batch, pred_len, channels), raw units."""
        mean = x.mean(1, keepdim=True)
        std = x.std(1, keepdim=True, unbiased=False) + 1e-5
        h = self.embedding.proj(((x - mean) / std).permute(0, 2, 1))
        for layer in self.layers:
            h = layer(h)
        out = self.projection(self.norm(h)).permute(0, 2, 1)
        return out * std + mean


class DataEmbedding_inverted(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1):
        super().__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = x.permute(0, 2, 1)  # [B, Variate, Time]
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            # Timestamp covariates ride along as extra variate tokens.
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1))
        return self.dropout(x)


class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1,
                 output_attention=False):
        super().__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        scale = self.scale or 1. / sqrt(E)
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)
        return (V.contiguous(), A if self.output_attention else None)


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super().__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)
        out, attn = self.inner_attention(queries, keys, values, attn_mask, tau=tau, delta=delta)
        return self.out_projection(out.view(B, L, -1)), attn


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask, tau=tau, delta=delta)
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y), attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super().__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        attns = []
        for attn_layer in self.attn_layers:
            x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
            attns.append(attn)
        if self.norm is not None:
            x = self.norm(x)
        return x, attns


class Model(nn.Module):
    def __init__(self, seq_len=96, pred_len=96, d_model=256, d_ff=256, n_heads=8,
                 e_layers=2, dropout=0.1, factor=1, activation="gelu",
                 output_attention=False, use_norm=True):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.output_attention = output_attention
        self.use_norm = use_norm
        self.enc_embedding = DataEmbedding_inverted(seq_len, d_model, dropout)
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, factor, attention_dropout=dropout,
                                      output_attention=output_attention),
                        d_model, n_heads),
                    d_model, d_ff, dropout=dropout, activation=activation,
                ) for _ in range(e_layers)
            ],
            norm_layer=nn.LayerNorm(d_model),
        )
        self.projector = nn.Linear(d_model, pred_len, bias=True)

    def forecast(self, x_enc, x_mark_enc):
        if self.use_norm:
            # Non-stationary Transformer instance normalization. Note this makes any
            # dataset-level z-scoring of the input a no-op end to end: it is divided out
            # here and multiplied back in below, so raw units in == raw units out.
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc = x_enc / stdev

        _, _, N = x_enc.shape
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        # B N E -> B N S -> B S N, dropping the covariate tokens
        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N]

        if self.use_norm:
            dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
            dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        return dec_out, attns

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        dec_out, attns = self.forecast(x_enc, x_mark_enc)
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        return dec_out[:, -self.pred_len:, :]
