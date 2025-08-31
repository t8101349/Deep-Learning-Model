import torch
import torch.nn as nn
from mamba_ssm import Mamba
#使用官方keernel省略ssmkernel實作

class Mambablock(nn.Module):
    def __init__(self, d_model, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.mamba = Mamba(d_model, d_state=16, expand=2, conv_kernel=15)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        x = x + self.dropout(self.mamba(self.norm1(x)))
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x 
    # pre-norm, dropout應在mambablock里實作

class Mambabackbone(nn.Module):
    def __init__(self, d_model = 128, n_layers=4, vocab_size=10000, dropout=0.1):
        super().__init__()
        self.embedding =nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            Mambablock(d_model, dropout) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        logits = self.head(x)
        return logits