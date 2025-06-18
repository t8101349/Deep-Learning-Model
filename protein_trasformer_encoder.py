import torch
import torch.nn as nn



AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"  # 20種常見氨基酸
aa_to_id = {aa: idx + 1 for idx, aa in enumerate(AMINO_ACIDS)}  # 0留給padding

def tokenize_sequence(seq, max_len=512):
    token_ids = [aa_to_id.get(aa, 0) for aa in seq]
    if len(token_ids) < max_len:
        token_ids += [0] * (max_len - len(token_ids))
    else:
        token_ids = token_ids[:max_len]
    return token_ids



class ProteinFunctionTransformer(nn.Module):
    def __init__(self, vocab_size=21, embed_dim=128, num_heads=4, num_layers=2, num_classes=10, max_len=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_len, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand_as(x)
        x = self.embedding(x) + self.pos_embedding(positions)

        x = self.transformer_encoder(x)  # (B, L, D)
        x = x.mean(dim=1)  # Average pooling
        return self.classifier(x)
