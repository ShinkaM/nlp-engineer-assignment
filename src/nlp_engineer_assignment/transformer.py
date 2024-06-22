import torch
import torch.nn as nn
import math
from typing import Optional
from torch.nn import functional as F
from dataclasses import dataclass

class MLP(nn.Module):
    def __init__(self, input_dim: int, ff_dim: int, dropout:int = 0.0):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, ff_dim),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(ff_dim, input_dim)
        )
    def forward(self, x: torch.Tensor):
        return self.model(x)


class CausalSelfAttention(nn.Module):
    def __init__(self, seq_len: int, hidden_dim: int, num_heads: int = 1):
        super().__init__()
        self.qkv_dim = hidden_dim
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.num_heads = num_heads

        self.qkv_proj = nn.Linear(hidden_dim, 3 * num_heads * self.qkv_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.qkv_dim, hidden_dim, bias=False)
        self.temp = 1
        self.causal_mask = torch.tril(torch.ones(seq_len, seq_len)).view( 1, seq_len, seq_len)


    def forward(self, x: torch.Tensor):
        B, S, H = x.shape

        M = self.qkv_proj(x)
        q,k,v = M.split(self.hidden_dim, dim=2)
    
        qk = torch.einsum('bih,bjh->bij', q, k) * (self.temp / math.sqrt(H))
        qk_masked = qk.masked_fill(self.causal_mask.repeat(B, 1, 1) == 0, float('-inf'))
        scores = F.softmax(qk_masked, dim=-1)
        output = torch.einsum('bij,bjh->bih',scores,v)

        return output