import torch
import torch.nn as nn
import math
from typing import Optional
from torch.nn import functional as F
from dataclasses import dataclass
import numpy as np
"""
MLP to be used in block
"""
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
    

class Block(nn.Module):
    def __init__(self, seq_len:int, hidden_dim:int, ff_dim:int, num_heads:int, dropout:float):
        super().__init__()
        self.attn = CausalSelfAttention(seq_len=seq_len, hidden_dim=hidden_dim, num_heads=num_heads)
        self.mlp = MLP(hidden_dim,ff_dim,dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x:torch.Tensor):
        attn_out = self.attn(x)
        x  = x + self.dropout(attn_out)
        x = self.norm1(x)

        linear_out = self.mlp(x)
        x = x + self.dropout(linear_out)
        x = self.norm2(x)
        return x
    
class Transformer(nn.Module):
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        embed_dim: int,
        vocab_size: int,
        ff_dim: int,
        output_vocab_size: int,
        seq_len: int = 512,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.output_vocab_size = output_vocab_size
        self.hyperparameters = dict(
            num_layers=num_layers,
            num_heads=num_heads,
            embed_dim=embed_dim,
            vocab_size=vocab_size,
            ff_dim=ff_dim,
            seq_len=seq_len,
            dropout=dropout,
            output_vocab_size=output_vocab_size,
        )
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.position_embed = PositionalEncoding(embed_dim, seq_len)

        self.decode_to_vocab = nn.Linear(embed_dim, output_vocab_size)
        self.dropout = nn.Dropout(p=dropout)

        self.layers = nn.ModuleList(
            [
                Block(
                    seq_len=seq_len,
                    hidden_dim=embed_dim,
                    ff_dim=ff_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.layer_norm = nn.LayerNorm(embed_dim)
        self.seq_len = seq_len

        def init_weights(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                torch.nn.init.zeros_(module.bias)
                torch.nn.init.ones_(module.weight)

        self.apply(init_weights)
        for name, param in self.named_parameters():
            if name.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    param, mean=0.0, std=0.02 / math.sqrt(2 * num_layers)
                )

    def forward(self, idx:torch.Tensor):
        B,S = idx.shape
        position_embed = self.position_embed(torch.arrange(S, device = idx.device).unsqueeze(0))
        token_embed = self.token_embed(idx)

        embed = self.dropout(token_embed + position_embed)

        for block in self.layers:
            embed = block(embed)
        embed = self.layer_norm(embed)
        logits = self.decode_to_vocab(embed)
        return logits   
    
    def step(self, idx:torch.Tensor, labels: torch.Tensor = None):
        logits = self(idx)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.reshape(-1, self.output_vocab_size), labels.flatten())
        return logits, loss
    

    @torch.no_grad()
    def generate(self, inp) -> torch.Tensor:
        if isinstance(inp, (list, np.ndarray)):
            inp = torch.longTensor(inp)
        elif not isinstance(inp, torch.Tensor):
            raise ValueError(
                "Expected inp to be list, np.array or torch.tensor. Found {typ}".format(
                    typ=type(inp)
                )
            )
        logits = self(inp.unsqueeze(0)).squeeze()
        idxs = logits.argmax(-1)
        return logits, idxs 
    
    

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
