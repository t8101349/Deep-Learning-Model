import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_gepmetric.nn import GATConv, global_mean_pool

class DrugEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, heads = 4):
        super().__init__()
        self.norm1 = nn.BatchNorm1d(input_dim)
        self.gat1 = GATConv(input_dim, hidden_dim // heads, heads = heads, concat=True)

        self.norm2 = nn.BatchNorm1d(input_dim)
        self.gat2 = GATConv(input_dim, hidden_dim, heads = 1, concat=False)

    def forward(self, x, edge_index, batch):
        x = self.norm1(x)
        x = self.gat1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)

        x = self.norm2(x)
        x = self.gat2(x, edge_index)

        x = global_mean_pool(x, batch)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0,d_model, 2)*(-torch.log(torch.tensor(10000.0))/d_model))
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[: ,1::2] = torch.cos(pos * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self,x):
        x = x +self.pe[:, :x.size(1)].to(x.device)
        return x

class TargetEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, hidden_dim, max_len=1000):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, max_len)
        encoder_layer = nn.TransformerEncoderLayer(embed_dim,num_heads, hidden_dim, dropout=0.1,batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, seq):
        x = self.embedding(seq)              # [B, L, D]
        x = self.pos_encoder(x)              # [B, L, D]
        x = self.transformer(x)              # [B, L, D]
        x = x.permute(0, 2, 1)               # [B, D, L]
        x = self.pool(x).squeeze(-1)         # [B, D]
        return x

class FusionModule(nn.Module):
    def __init__(self, drug_dim, target_dim, fusion_dim):
        super(FusionModule, self).__init__()
        self.fc1 = nn.Linear(drug_dim + target_dim, fusion_dim)
        self.fc2 = nn.Linear(fusion_dim, 1)

    def forward(self, drug_feat, target_feat):
        fused = torch.cat((drug_feat, target_feat), dim=1)
        x = F.relu(self.fc1(fused))
        x = torch.sigmoid(self.fc2(x))
        return x

class FusionDTI(nn.Module):
    def __init__(self, drug_input_dim, drug_hidden_dim, drug_output_dim,
                 target_vocab_size, target_embed_dim, ,target_heads, target_layers, target_hidden_dim,
                 fusion_dim):
         
        super().__init__()
        self.Drug_Encoder = DrugEncoder(drug_input_dim, drug_hidden_dim, drug_output_dim)
        self.Target_Encoder = TargetEncoder(target_vocab_size, target_embed_dim,target_heads, target_layers, target_hidden_dim)
        self.fusion = FusionModule(drug_output_dim, target_embed_dim, fusion_dim)

    def forward(self, drug_data, target_seq):
        x_drug = self.Drug_Encoder(drug_data.x, drug_data.edge_index)
        x_target = self.Target_Encoder(target_seq)
        x = self.fusion(x_drug.unsqueeze(0), x_target)
        return x