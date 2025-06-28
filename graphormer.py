import torch
import torch.nn as nn
import torch.nn.functional as F

'''
# MultiheadAttention不支援attn_bias, 只能做線性注意力
class GraphormerLayer(nn.Module):
    def __init__(self, dim, num_heads, dropout =0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim*4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim*4, dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_bias=None):
        res = x
        if attn_bias is not None:
            raise NotImplementedError("nn.MultiheadAttention doesn't support attn_bias directly.")
        else:
            attn_output, _ = self.attn(x,x,x)
        x = self.dropout(attn_output)
        x = self.norm1( res + x)
        
        res = x
        x = self.ffn(x)
        x = self.dropout(x)
        x = self.norm2(res + x)
        return x
'''

class GraphormerLayer(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.1, **kwargs):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.attn_bias_type = kwargs.get('attn_bias_type', 'none')
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"

        self.qkv_proj = nn.Linear(dim, dim * 3)
        self.out_proj = nn.Linear(dim, dim)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_bias=None):
        B, N, D = x.shape
        # pre-norm
        x = self.norm1(x)
        qkv = self.qkv_proj(x)  # [B, N, 3D]
        q, k, v = qkv.chunk(3, dim=-1)

        # reshape to [B, num_heads, N, head_dim]
        q = q.view(B, N, self.num_heads, -1).transpose(1, 2)  # [B, H, N, d]
        k = k.view(B, N, self.num_heads, -1).transpose(1, 2)
        v = v.view(B, N, self.num_heads, -1).transpose(1, 2)

        attn_scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [B, H, N, N]

        if attn_bias is not None:
            attn_scores = attn_scores + attn_bias.unsqueeze(1)  # broadcasting over heads

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = attn_weights @ v  # [B, H, N, d]
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, N, D)  # concat heads
        attn_output = self.out_proj(attn_output)

        # residual 
        x = x + self.dropout(attn_output)

        # pre-norm FFN block
        x = self.norm2(x)
        x = x + self.dropout(self.ffn(x))

        return x
    
    #attn_bias 是用來把「圖的結構資訊」融入自注意力中的方法
    # 假設已計算好距離矩陣 [B, N, N]
    #edge_dist = torch.randint(0, 512, (B, N, N))  # 離散距離
    #attn_bias = edge_distance_embedding(edge_dist)  # [B, N, N, H]
    #attn_bias = attn_bias.mean(dim=-1)  # [B, N, N]

class GraphormerEncoder(nn.Module):
    def __init__(self, num_layers, dim, num_heads, dropout=0.1, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList([
            GraphormerLayer(dim=dim, num_heads=num_heads, dropout=dropout) for _ in range(num_layers)
        ])

    def forward(self, x, attn_bias=None):
        for layer in self.layers:
            x = layer(x, attn_bias= attn_bias)

        return x

class GraphormerModel(nn.Module):
    def __init__(self, node_feat_dim, edge_feat_dim, num_layers=6, hidden_dim=256, num_heads=8, dropout=0.1, **kwargs):
        super().__init__()
        self.node_feat_emb = nn.Linear(node_feat_dim, hidden_dim)
        self.edge_feat_emb = nn.Embedding(edge_feat_dim, num_heads)  #邊的類型或特徵進行編碼
        self.edge_emb = nn.Embedding(512, num_heads)  # 節點對的離散距離進行編碼
        self.encoder = GraphormerEncoder(
            num_layers=num_layers, 
            dim=hidden_dim, 
            num_heads=num_heads, 
            dropout=dropout,
            **kwargs)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.pred_head = nn.Linear(hidden_dim, 1)
    
    def forward(self, node_feat, edge_dist):