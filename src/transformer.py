from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


def subsequent_mask(size: int, device=None) -> torch.Tensor:
    # (1, 1, size, size)
    return torch.tril(torch.ones((size, size), dtype=torch.bool, device=device)).unsqueeze(0).unsqueeze(0)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask: torch.Tensor | None = None):
        # q,k,v: (B, h, Tq, d_k)
        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)  # (B,h,Tq,Tk)
        if mask is not None:
            # mask must be broadcastable to (B,h,Tq,Tk) and boolean True=keep
            scores = scores.masked_fill(~mask, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        return torch.matmul(attn, v), attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.attn = ScaledDotProductAttention(dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask: torch.Tensor | None = None):
        # q,k,v: (B,T,d_model)
        B = q.size(0)

        q = self.w_q(q).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)

        if mask is not None:
            # allow masks of (B,1,1,Tk) or (B,1,Tq,Tk) etc.
            # Broadcast across heads.
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)

        x, attn = self.attn(q, k, v, mask=mask)
        x = x.transpose(1, 2).contiguous().view(B, -1, self.d_model)
        x = self.w_o(x)
        return self.dropout(x), attn


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class SublayerConnection(nn.Module):
    """Post-norm residual: LayerNorm(x + Dropout(sublayer(x)))"""

    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return self.norm(x + self.dropout(sublayer(x)))


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.sublayer1 = SublayerConnection(d_model, dropout)
        self.sublayer2 = SublayerConnection(d_model, dropout)

    def forward(self, x, src_mask: torch.Tensor | None = None):
        x = self.sublayer1(x, lambda t: self.self_attn(t, t, t, mask=src_mask)[0])
        x = self.sublayer2(x, self.ff)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.src_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.sublayer1 = SublayerConnection(d_model, dropout)
        self.sublayer2 = SublayerConnection(d_model, dropout)
        self.sublayer3 = SublayerConnection(d_model, dropout)

    def forward(self, x, memory, src_mask: torch.Tensor | None, tgt_mask: torch.Tensor | None):
        x = self.sublayer1(x, lambda t: self.self_attn(t, t, t, mask=tgt_mask)[0])
        x = self.sublayer2(x, lambda t: self.src_attn(t, memory, memory, mask=src_mask)[0])
        x = self.sublayer3(x, self.ff)
        return x


class Encoder(nn.Module):
    def __init__(self, layer: EncoderLayer, N: int):
        super().__init__()
        self.layers = nn.ModuleList([layer if i == 0 else type(layer)(
            d_model=layer.self_attn.d_model,
            n_heads=layer.self_attn.n_heads,
            d_ff=layer.ff.w_1.out_features,
            dropout=layer.sublayer1.dropout.p,
        ) for i in range(N)])

    def forward(self, x, src_mask: torch.Tensor | None = None):
        for layer in self.layers:
            x = layer(x, src_mask=src_mask)
        return x


class Decoder(nn.Module):
    def __init__(self, layer: DecoderLayer, N: int):
        super().__init__()
        self.layers = nn.ModuleList([layer if i == 0 else type(layer)(
            d_model=layer.self_attn.d_model,
            n_heads=layer.self_attn.n_heads,
            d_ff=layer.ff.w_1.out_features,
            dropout=layer.sublayer1.dropout.p,
        ) for i in range(N)])

    def forward(self, x, memory, src_mask: torch.Tensor | None = None, tgt_mask: torch.Tensor | None = None):
        for layer in self.layers:
            x = layer(x, memory, src_mask=src_mask, tgt_mask=tgt_mask)
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_len: int = 512,
        tie_embeddings: bool = True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model, dropout=dropout, max_len=max_len)

        enc_layer = EncoderLayer(d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout)
        dec_layer = DecoderLayer(d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout)
        self.encoder = Encoder(enc_layer, num_layers)
        self.decoder = Decoder(dec_layer, num_layers)

        self.generator = nn.Linear(d_model, vocab_size, bias=False)
        if tie_embeddings:
            self.generator.weight = self.tok_embed.weight

        self.dropout = nn.Dropout(dropout)

    def encode(self, src, src_mask=None):
        x = self.tok_embed(src) * math.sqrt(self.d_model)
        x = self.pos(x)
        return self.encoder(x, src_mask=src_mask)

    def decode(self, tgt_in, memory, src_mask=None, tgt_mask=None):
        x = self.tok_embed(tgt_in) * math.sqrt(self.d_model)
        x = self.pos(x)
        return self.decoder(x, memory, src_mask=src_mask, tgt_mask=tgt_mask)

    def forward(self, src, tgt_in, src_mask=None, tgt_mask=None):
        memory = self.encode(src, src_mask=src_mask)
        out = self.decode(tgt_in, memory, src_mask=src_mask, tgt_mask=tgt_mask)
        logits = self.generator(out)
        return logits
