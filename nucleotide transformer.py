import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math
import numpy as np

class KmerEmbedding(nn.Module):
    def __init__(self, vocab_size:int, embed_dim:int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)

    def forward(self, x):
        return self.embedding(x)
    
class PositionEncoding(nn.Module):
    def __init__(self, embed_dim:int, max_len:int=10000):
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)
        pos = torch.arange(0,max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-torch.log(torch.tensor(10000))/embed_dim))
        pe[:,0::2]= torch.sin(pos * div_term)
        pe[:,1::2]= torch.cos(pos * div_term)
        
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1),:]
        return x

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1 ):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim*4),
            nn.GELU(),
            nn.Linear(embed_dim*4, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        # self.proj_res = nn.Linear()
    
    def forward(self, x):
        res = x
        x,weights = self.attn(x,x,x)
        x = self.dropout(x)
        x = self.norm1(res + x)

        res = x
        x = self.ffn(x)
        x = self.dropout(x)
        x = self.norm2(res + x)
        return x
    
class NucleotideTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        embed_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        num_classes: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed = KmerEmbedding(vocab_size, embed_dim)
        self.pe = PositionEncoding(embed_dim, max_seq_len)
        self.transformers =nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim,embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self,x):
        x = self.embed(x)
        x = self.pe(x)
        for layer in self.transformers:
            x = layer(x)
        
        x = x.transpose(1,2)
        x = self.pool(x).squeeze(-1)
        return self.fc((x))

if __name__ == "__main__":
    model = NucleotideTransformer(
        vocab_size=64,  # depends on 3-mer tokenizer
        max_seq_len=512,
        embed_dim=256,
        num_heads=4,
        num_layers=4,
        num_classes=2
    )
    x = torch.randint(0, 64, (8, 512))  # [batch, seq_len]
    logits = model(x)
    print(logits.shape)  # [8, 2]