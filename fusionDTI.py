import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GCNConv, global_mean_pool, GraphNorm

# ======== Drug Encoder (GCN) ========
class DrugEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DrugEncoder, self).__init__()
        self.norm1 = GraphNorm(input_dim)
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.norm2 = GraphNorm(hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index, batch):
        x = self.norm1(x)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.norm2(x)
        x = self.conv2(x, edge_index)
        return global_mean_pool(x, batch)

# ======== Target Encoder (Transformer) ========
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        # self.pe = pe.unsqueeze(0)  # shape: [1, max_len, d_model]
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].to(x.device)
        return x

class TargetEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_heads, num_layers, max_len=1000):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pe = PositionalEncoding(embed_dim, max_len)
        encoderlayer = nn.TransformerEncoderLayer(embed_dim, num_heads, hidden_dim, dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoderlayer, num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, seq):
        x = self.embedding(seq)
        x = self.pe(x)
        x = self.transformer(x)
        x = x.permute(0,2,1)
        x = self.pool(x).squeeze(-1)
        return x

class FusionModule(nn.Module):
    def __init__(self, drug_dim, target_dim, fusion_dim):
        super().__init__()
        self.fc1 = nn.Linear(drug_dim + target_dim , fusion_dim)
        self.fc2 = nn.Linear(fusion_dim , 1)

    def forward(self, drug_feature, target_feature):
        fused = torch.cat([drug_feature,target_feature], dim=1)
        x = self.fc1(fused)
        x = F.relu(x, inplace=True)
        x = self.fc2(x)
        return torch.sigmoid(x)
    
class FusionDTI(nn.Module):
    def __init__(self,
                 drug_input_dim, drug_hidden_dim, drug_output_dim,
                 target_vocab_size, target_embed_dim, target_heads, target_layers, target_hidden_dim,
                 fusion_dim):
        super().__init__()
        self.drug_encoder = DrugEncoder(drug_input_dim, drug_hidden_dim, drug_output_dim)
        self.target_encoder = TargetEncoder(target_vocab_size, target_embed_dim, target_hidden_dim, target_heads, target_layers)
        self.fusion = FusionModule(drug_output_dim, target_embed_dim, fusion_dim)
    
    def forward(self, drug_data, target_seq):
        x_drug = self.drug_encoder(drug_data.x, drug_data.edge_index, drug_data.batch)
        x_target = self.target_encoder(target_seq)
        x = self.fusion(x_drug, x_target)
        return x