import torch
import torch.nn as nn
from mamba_ssm import Mamba

class Mambablock(nn.Module):
    def __init__(self, d_model, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba(d_model, d_state=16, expand=2, conv_kernel=15)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        return x + self.dropout(self.mamba(self.norm(x)))
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