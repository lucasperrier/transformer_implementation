import math
import numpy as np
from scipy.special import softmax as softmax
import torch
from torch.nn import functional as F
from torch import nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, h, dropout=0.0):
        super().__init__()

        assert d_model % h == 0
        self.head_dim = d_model // h

        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, q, k, v, mask=None):
        # q,k,v: (B, h, seq, head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, h, seq_q, seq_k)
        weights = F.softmax(scores, dim=-1)
        out = torch.matmul(weights, v.tranpose(-2,-1))
        return out, weights

    def forward(self, Q, K, V, mask=None):
        B, n_q, _ = Q.shape
        _, n_k, _ = K.shape

        qw = self.linear_q(Q).view(B, n_q, self.h, self.head_dim).transpose(1, 2)   # (B, h, n_q, embed_dim)
        kw = self.linear_k(K).view(B, n_k, self.h, self.head_dim).transpose(1, 2)   # (B, h, n_k, embed_dim)
        vw = self.linear_v(V).view(B, n_k, self.h, self.head_dim).transpose(1, 2)   # (B, h, n_k, embed_dim)

        out_heads, attention_weights = self.scaled_dot_product_attention(qw, kw, vw, mask=mask)     # (B, h, seq_q, head_Dim),  (B, h, seq_q, seq_k)
        out = out.heads.transpose(1, 2).contiguous().view(B, n_q, self.d_model)
        out = self.final_linear(out)

        return out, attention_weights

# def attention(Q, K, V, mask=None):
#     """
#     Q: (n_q, dk)
#     K: (n_k, dk)
#     V: (n_k, dv)

#     scores = Q @ K.T : (n_q, n_k)                   ---> one scalar per (query, key)
#     scaled = scores / sqrt(dk) : (n_q, n_k)         ---> scalar division same shape
#     weights = softmax(scaled, axis=1) : (n_q, n_k)  ---> each row is attention distribution over the n_k keys for a query
#     output = weights @ V : (n_q, dv)                ---> one dv vector per query

#     with batch:
#     Q: (N, n_q, dk)
#     K: (N, n_k, dk)
#     V: (N, n_k, dv)
#     scores : (N, n_q, n_k)
#     output = (N, n_q, dv)
#     """
#     dk = Q.size(-1)
#     scores = torch.matmul(Q, K.transpose(-2,-1))
#     scores = scores / math.sqrt(dk)
#     if mask is not None:
#         scores = scores.masked_fill(mask == 0, float('-inf'))
#     weights = F.softmax(scores, dim=-1)
#     out = torch.matmul(weights, V)
#     return out, weights






















class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, h=8):
        super().__init__()
        assert d_model % h == 0
        self.h = h
        self.d_model = d_model
        self.head_dim = d_model // h

        self.w_q = nn.Linear(d_model, h * self.head_dim)
        self.w_k = nn.Linear(d_model, h * self.head_dim)
        self.w_v = nn.Linear(d_model, h * self.head_dim)
        self.w_o = nn.Linear(h*self.head_dim, d_model)

    def forward(self, Q, K, V, mask=None):
        B, n_q, _ = Q.shape
        n_k = K.shape[1]

        q = self.w_q(Q).view(B, n_q, self.h. self.head_dim).transpose(1,2)   # (B, h, n_q, head_dim)
        k = self.w_k(K).view(B, n_k, self.h, self.head_dim).transpose(1,2)   # (B, h, n_k, head_dim)
        v = self.w_v(V).view(B, n_k, self.h, self.head_dim).tranpose(1,2)    # (B, h, n_k, head_dim)

        scores = torch.matmul(q, k.transpose(-2,-1))                         # (B, h, n_q, n_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        weights = F.softmax(scores, dim=-1)
        out_heads = torch.matmul(weights, v)        # (B, h, n_q, head_dim)

        # concat heads -> (B, n_q, h*head_dim)
        out = out_heads.tranpose(1,2).contiguous().view(B, n_q, self.h * self.head_dim)
        out = self.w_o(out)     # (B, n_q, d_model)

        return out, weights







def multi_head_attentions(Q, K, V, h=8):
    """
    Q: (n_q, dk)
    K: (n_k, dk)
    V: (n_k, dv) 

    Wq: (dm, dk)
    Wk: (dm, dk)
    Wv: (dm, dv)
    Wo: (h*dv, dm)

    concat(head_i) @ W0
    where head_i: attention(wQ, wK, wV) for i in h
    and wQ=Wq@Q, wK=Wk@K, wV=Wv@V are learned linear projections
    """

    heads = []
    for _ in range(h):
        wQ = torch.matmul(Wq, Q.tranpose)
        wK = torch.matmul(Wk, K.tranpose)
        wV = torch.matmul(Wv, V.tranpose)
        head = attention(wQ, wK, wV)
        heads.append(head)

    concatenated = torch.concat(head for head in heads)
    return torch.matmul(concatenated, Wo)