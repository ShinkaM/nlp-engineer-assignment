import torch
import torch.nn as nn
import math
from typing import Optional
from torch.nn import functional as F
from dataclasses import dataclass
import numpy as np
from .dataset import CharTokenizedDataset
from .utils import count_letters, read_inputs, score
from tqdm import tqdm
import os
import pickle
"""
MLP to be used in block

input_dim = dimension f inout tensor
ff_dim = hidden dimension
dropout = parameter for nn.Dropout
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

"""
Implementation of self attention from scratch

"""
class CausalSelfAttention(nn.Module):
    def __init__(self, seq_len: int, hidden_dim: int, num_heads: int = 1):
        super().__init__()
        self.qkv_dim = hidden_dim 
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.num_heads = num_heads
        assert hidden_dim % num_heads == 0, "num_heads must be a factor of hidden_dim."

        self.qkv_proj = nn.Linear(hidden_dim, 3 * self.qkv_dim, bias=False)#key, query, value projections in a batch
        self.o_proj = nn.Linear(self.qkv_dim, hidden_dim, bias=False)#output projection
        self.temp = 1
        self.causal_mask = torch.tril(torch.ones(seq_len, seq_len)).view(
            1, 1, seq_len, seq_len
        )#apply masking so that the model does not see ahead. 

    def forward(self, x: torch.Tensor):
        B, S, D = x.shape
        M = self.qkv_proj(x)
        q, k, v = M.split(self.hidden_dim, dim=-1)
        q = q.view(B, S, self.num_heads, -1)  # (B, S, H, D)
        k = k.view(B, S, self.num_heads, -1)
        v = v.view(B, S, self.num_heads, -1)

        qk = torch.einsum("bihd,bjhd->bhij", q, k) * (
            self.temp / math.sqrt(D // self.num_heads)
        )#multiplication of q times k
        qk_masked = qk.masked_fill(self.causal_mask[:, :, :S, :S] == 0, float("-inf"))
        scores = F.softmax(qk_masked, dim=-1)
        output = torch.einsum("bhij,bjhd->bihd", scores, v)#multiplication of qk * v
        output = output.contiguous().view(B, S, D)
        return output
    
"""
seq_leb, embedding_dim

"""
class PositionalEncoding(torch.nn.Module):
    def __init__(self, seq_len:int, hidden_dim:int):
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
    
"""
Implementation of nn.LayerNorm from scratch
"""
class CustomLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(CustomLayerNorm, self).__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.gamma = nn.Parameter(torch.ones(normalized_shape))
            self.beta = nn.Parameter(torch.zeros(normalized_shape))
        else:
            self.register_parameter('gamma', None)
            self.register_parameter('beta', None)

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_normalized = (x - mean) / torch.sqrt(var + self.eps)

        if self.elementwise_affine:
            x_normalized = x_normalized * self.gamma + self.beta
        
        return x_normalized
    

class Block(nn.Module):
    def __init__(self, seq_len:int, hidden_dim:int, ff_dim:int, num_heads:int, dropout:float):
        super().__init__()
        self.attn = CausalSelfAttention(seq_len=seq_len, hidden_dim=hidden_dim, num_heads=num_heads)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(ff_dim, hidden_dim),
        )
        self.norm1 = CustomLayerNorm(hidden_dim)
        self.norm2 = CustomLayerNorm(hidden_dim)
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
        # self.position_embed = PositionalEncoding(seq_len, embed_dim)
        self.position_embed = nn.Embedding(seq_len, embed_dim)
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

        self.layer_norm = CustomLayerNorm(embed_dim)
        self.seq_len = seq_len

        def init_weights(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, CustomLayerNorm):
                torch.nn.init.zeros_(module.beta)
                torch.nn.init.ones_(module.gamma)

        self.apply(init_weights)
        for name, param in self.named_parameters():
            if name.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    param, mean=0.0, std=0.02 / math.sqrt(2 * num_layers)
                )
    def configure_optimizers(self, lr, weight_decay):
        #select parameters that does/does not experience weight decay
        decay = set()
        no_decay = set()
        w_modules = (torch.nn.Linear,)
        b_modules = (CustomLayerNorm, torch.nn.Embedding)
        for mod_name, m in self.named_modules():
            for param_name, p in m.named_parameters():
                fpn = "%s.%s" % (mod_name, param_name) if mod_name else param_name
                if param_name.endswith("bias"):#bias will not be decayed
                    no_decay.add(fpn)
                elif param_name.endswith("weight") and isinstance(m, w_modules): #weights of w_modules will be decayed
                    decay.add(fpn)
                elif param_name.endswith("weight") and isinstance(m, b_modules):#weights of b_modules will not be decayed
                    no_decay.add(fpn)

        param_dict = dict(self.named_parameters())
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(optim_groups, lr=lr, weight_decay=weight_decay)
        return optimizer


    def forward(self, idx:torch.Tensor):
        B,S = idx.shape
        position_embed = self.position_embed(torch.arange(S, device = idx.device).unsqueeze(0))
        token_embed = self.token_embed(idx)
        # position_embed = self.position_embed(token_embed)
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
    """
    Used to verify outputs manually
    """
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
    
    def save_checkpoint(self, path):
        assert os.path.exists(
            os.path.dirname(path)
        ), f"Path {os.path.dirname(path)} not found."
        artifact = dict(
            state_dict=self.state_dict(), hyperparameters=self.hyperparameters
        )
        with open(path, "wb") as fp:
            pickle.dump(artifact, fp)

    @classmethod
    def from_pretrained(cls, model_path):
        assert os.path.exists(model_path), f"Model not found at {model_path}"
        with open(model_path, "rb") as fp:
            artifact = pickle.load(fp)

        model = cls(**artifact["hyperparameters"])
        model.load_state_dict(artifact["state_dict"], strict=True)
        return model


def train_classifier(
    vocabs,
    train_inputs,
    batch_size: int = 10,
    num_workers: int = 1,
    n_epochs: int = 10,
    device: str = "cpu",
    eval_every_n_epochs: int = 1,
):
    output_vocab_size = 3  # 0, 1, or 2

    # @TODO: Analysis on the minimum number of layers we need "in theory"
    # to modl the "counting" function. From RASP: ICML 2021.
    model = Transformer(
        ff_dim=64,
        num_layers=5,
        num_heads=4,
        embed_dim=512,
        vocab_size=len(vocabs),
        output_vocab_size=output_vocab_size,
        dropout=0.01,
    )
    optimizer = model.configure_optimizers(lr=3e-4, weight_decay=0.02)
    # Reduce LR 0.1x after every epoch.
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1)


    train_dataset = CharTokenizedDataset(sentences=train_inputs, vocab=vocabs)
    train_loader = train_dataset.get_dataloader(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=(num_workers > 0),
    )

    for epoch in range(n_epochs):
        model.train()
        batch_iter = tqdm(train_loader, desc=f"Epoch {epoch}, Loss: NaN")
        for i, batch in enumerate(batch_iter):
            idxs, counts = batch
            idxs = idxs.to(device)
            counts = counts.to(device)
            logits, loss = model.step(idx=idxs, labels=counts)
            model.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            batch_iter.set_description(
                "Epoch {epoch}, Loss: {loss:.4f}".format(
                    epoch=epoch,
                    loss=loss.item(),
                )
            )
        scheduler.step()
        #print sample after epochs
        if epoch % eval_every_n_epochs == 0:
            model.eval()
            sample_sentence = train_inputs[0]
            sample_input, sample_count = train_dataset[0]
            pred_logits, generated_count = model.generate(sample_input)
            for i, c in enumerate(sample_sentence):
                print(
                    "{i:>2}, {c}, {count:>3}, {pred_count:>3}".format(
                        i=i, c=c, count=sample_count[i], pred_count=generated_count[i]
                    )
                )
    return model


if __name__ == "__main__":
    # Simple tests.
    # Unit test for MLP
    # mlp = MLP(input_dim = 1, ff_dim = 3, dropout= 0.0)
    # out = mlp(torch.zeros(10, 1))
    # assert out.shape == (10, 1), f"Failed!{out.shape}"

    # # Unit test for Self Attention
    # s = CausalSelfAttention(seq_len = 32, hidden_dim = 16, num_heads = 1)
    # inp = torch.zeros(10, 32, 16)
    # out = s(inp)
    # assert out.shape == (10, 32, 16), f"Failed!{out.shape}"
    # pe = PositionalEncoding(hidden_dim=16, seq_len = 32)
    # inp = torch.zeros(10, 32, 16)
    # out = pe(inp)
    # assert out.shape == inp.shape, f"Failed!{out.shape, inp.shape}"
    input_tensor = torch.randn(20, 5, 10, 10)  # Example input tensor
    layer_norm = CustomLayerNorm(10)  # Normalize over the last dimension
    output_tensor = layer_norm(input_tensor)
    assert output_tensor.shape == input_tensor.shape, f"Failed!{output_tensor.shape, input_tensor.shape}"
