import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_k = d_model // num_heads
        self.num_heads = num_heads

        self.q_linear = nn.Linear(d_model,d_model)
        self.k_linear = nn.Linear(d_model,d_model)
        self.v_linear = nn.Linear(d_model,d_model)
        self.out_proj = nn.Linear(d_model,d_model)

        self.dropout = nn.Dropout(0.1)

    def forward(self, x, q, k, v, mask=None):
        B, T_q, _ = q.size()
        B, T_k, _ = k.size()
        B, T_v, _ = v.size()

        def transform(x, linear, T):
            x = linear(x)
            return x.view(B, T, self.num_heads, self.d_k).transpose(1, 2)

        q = transform(q, self.q_linear, T_q)
        k = transform(k, self.k_linear, T_k)
        v = transform(v, self.v_linear, T_v)

        scores = torch.matmul(q, k.transpose(-2,-1))/ math.sqrt(self.d_k)
        # a = q*k/(d_k**1/2)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        output = torch.matmul(attn, v)
        output = output.transpose(1, 2).contiguous().view(B, T, -1)

        return self.out_proj(output)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.LayerNorm(d_ff),
            nn.ReLU(inplace = True),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
    def forward(self, x):
        return self.net(x)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.ffl = FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        
        dx = self.attn(x, x, x, mask)
        dx = self.dropout(dx)
        x = self.norm1(x + dx)
        dx = self.ffl(x)
        dx = self.dropout(dx)
        x = self.norm2(x + dx)
        return x

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, max_len=5000, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.encoding_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        #x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.embedding(x)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        for layer in self.encoding_layers:
            x = layer(x, mask)
        return x

class DecoderLayer(nn.Module):
    def __init__(self,d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.ffl = FeedForward(d_model, d_ff)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encode_out, tgt_mask=None, memory_mask=None):
        x2 = self.self_attn(x,x,x,tgt_mask)
        x2 = self.dropout(x2)
        x = self.norm1(x + x2)

        x2 = self.cross_attn(x, encode_out, encode_out, memory_mask)
        x2 = self.dropout(x2)
        x = self.norm2(x + x2)

        x2 = self.ffl(x)
        x = self.norm3(x + self.dropout(x2))
        return x

class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, max_len=5000, dropout = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encode_out, tgt_mask=None, memory_mask=None):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, encode_out, tgt_mask, memory_mask, dropout)

        return x

class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder
        self.decoder = Decoder
        self.out = nn.Linear

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        encode_out = self.encoder(src, src_mask)
        decode_out = self.decoder(tgt, encode_out, tgt_mask, src_mask)

        out = self.out(decode_out)

        return out