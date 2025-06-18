import torch
import torch.nn as nn
import torch.nn.functional as F

class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = nn.Parameter(torch.randn(1, 512, embed_dim))

    def forward(self, x):
        emb = self.embedding(x)
        emb = emb + self.pos_encoding[:, :x.size(1), :]
        return emb

class SegmentEncoder(nn.Module):
    def __init__(self, emb_dim, num_layers=2, num_heads=8, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=num_heads,dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    
    def forward(self,x):
        x = x.permute(1,0,2)
        out = self.transformer(x)
        return x.permute(1,0,2)

class InteractionModule(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim = embed_dim, num_heads = 8, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim*2, 512),
            nn.ReLU(inplace = True),
            nn.Linear(512, 1)
        )

    def forward(self, drug, protein):
        attn_output ,_ = self.attn(drug, protein, protein)
        combined = torch.cat([attn_output, drug], dim = -1)
        pooled = combined.mean(dim=1)
        return self.fc(pooled)

class  MolTrans(nn.Module):
    def __init__(self, drug_vocab_size, protein_vocab_size, emb_dim = 128):
        super().__init__()
        self.drug_emb = EmbeddingLayer(drug_vocab_size, emb_dim)
        self.protein_emb = EmbeddingLayer(protein_vocab_size, emb_dim)

        self.drug_encoder = SegmentEncoder(emb_dim)
        self.protein_encoder = SegmentEncoder(emb_dim)

        self.interaction = InteractionModule(emb_dim)
    
    def forward(self, drug_seq, pro_seq):
        drug = self.drug_emb(drug_seq)
        protein = self.protein_emb(pro_seq)

        drug_encoder = self.drug_encoder(drug)
        protein_encoder = self.protein_encoder(protein)

        score = self.interaction(drug_encoder, protein_encoder)
        return torch.sigmoid(score)
    
