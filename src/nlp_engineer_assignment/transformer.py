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
        assert hidden_dim % num_heads == 0, "num_heads must be a factor of hidden_dim."

        self.qkv_proj = nn.Linear(hidden_dim, 3 * self.qkv_dim, bias=False)
        self.o_proj = nn.Linear(self.qkv_dim, hidden_dim, bias=False)
        self.temp = 1
        self.causal_mask = torch.tril(torch.ones(seq_len, seq_len)).view(
            1, 1, seq_len, seq_len
        )

    def forward(self, x: torch.Tensor):
        B, S, D = x.shape
        M = self.qkv_proj(x)
        q, k, v = M.split(self.hidden_dim, dim=-1)
        q = q.view(B, S, self.num_heads, -1)  # (B, S, H D)
        k = k.view(B, S, self.num_heads, -1)
        v = v.view(B, S, self.num_heads, -1)

        qk = torch.einsum("bihd,bjhd->bhij", q, k) * (
            self.temp / math.sqrt(D // self.num_heads)
        )
        qk_masked = qk.masked_fill(self.causal_mask[:, :, :S, :S] == 0, float("-inf"))
        scores = F.softmax(qk_masked, dim=-1)
        output = torch.einsum("bhij,bjhd->bihd", scores, v)
        output = output.contiguous().view(B, S, D)
        return output
    

class PositionalEncoding(torch.nn.Module):
    def __init__(self, hidden_dim:int, seq_len:int):
        super().__init__()
        pe = torch.zeros(seq_len, hidden_dim)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_dim, 2).float() * (-math.log(10000.0) / hidden_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x):
        # import IPython; IPython.embed()
        x = x + self.pe[:, : x.size(1)]
        return x

if __name__ == "__main__":
    # Simple tests.
    # Unit test for MLP
    mlp = MLP(input_dim = 1, ff_dim = 3, dropout= 0.0)
    out = mlp(torch.zeros(10, 1))
    assert out.shape == (10, 1), f"Failed!{out.shape}"

    # Unit test for Self Attention
    s = CausalSelfAttention(seq_len = 32, hidden_dim = 16, num_heads = 1)
    inp = torch.zeros(10, 32, 16)
    out = s(inp)
    assert out.shape == (10, 32, 16), f"Failed!{out.shape}"
    pe = PositionalEncoding(hidden_dim=16, seq_len = 32)
    inp = torch.zeros(10, 32, 16)
    out = pe(inp)
    assert out.shape == inp.shape, f"Failed!{out.shape, inp.shape}"
