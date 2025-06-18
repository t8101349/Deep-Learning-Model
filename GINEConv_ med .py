import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import GINEConv, global_mean_pool
from torch_geometric.nn import Sequential as GNNSequential
import torch.nn.functional as F

class protein_encoder(nn.Module):
    def __init__(self, seq_len, vocab_size=26, embed_dim=128, hidden_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_dim, embed_dim, padding_idx=0)
        self.conv = nn.Sequential(
            nn.Conv1d(embed_dim, hidden_dim, kernel_size=5, padding=2),
            nn.ReLU(inplace = True),
            nn.AdaptiveMaxPool1d(1),
        )

    def forward(self, x):
        x = self.embedding(seq)          # (B, L, E)
        x = x.transpose(1, 2)            # (B, E, L)
        x = self.conv(x)                 # (B, H, 1)
        return x.squeeze(-1)             # (B, H)

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim= None):
        super().__init__()
        if not hidden_dim:
            hidden_dim = out_dim

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace = True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace = True),
            nn.Linear(hidden_dim, out_dim),
        )
    def forward(self, x):
        return self.net(x)

class GINE(nn.Module):
    def __init__(self, node_feat_dim, edge_feat_dim, prot_vocab_size=26, prot_embed_dim=128, hidden_dim=256, out_dim=1):
        super().__init__()
        self.res_proj = nn.Linear(node_feat_dim, hidden_dim)
        self.edge_encoder = nn.Linear(edge_feat_dim, hidden_dim)
        self.protein_encoder = protein_encoder(pro_vocab_size, prot_embed_dim, hidden_dim)

        self.nn1 = MLP(node_feat_dim, hidden_dim)
        self.nn2 = MLP(hidden_dim, hidden_dim)
        self.nn3 = MLP(hidden_dim, hidden_dim)
        self.nn4 = MLP(hidden_dim, hidden_dim)
        self.nn5 = MLP(hidden_dim, hidden_dim)

        self.conv1 = GINEConv(self.nn1)
        self.conv2 = GINEConv(self.nn2)
        self.conv3 = GINEConv(self.nn3)
        self.conv4 = GINEConv(self.nn4)
        self.conv5 = GINEConv(self.nn5)

        self.norm1 = nn.BatchNorm1d(hidden_dim)
        self.norm2 = nn.BatchNorm1d(hidden_dim)
        self.norm3 = nn.BatchNorm1d(hidden_dim)
        self.norm4 = nn.BatchNorm1d(hidden_dim)
        self.norm5 = nn.BatchNorm1d(hidden_dim)

        self.drop = nn.Dropout(0.3)


        self.final_pred = MLP(hidden_dim, out_dim, hidden_dim=128)
    
    def forward(self, h, edge_index, edge_attr, batch):
        res = self.res_proj(h)
        edge_attr = self.edge_encoder(edge_attr)

        h = self.conv1(h,edge_index, edge_attr)
        h = self.norm1(h)
        h = F.relu(h)
        h = self.drop(h)

        res = h
        h = self.conv2(h,edge_index, edge_attr)
        h = self.norm2(h)
        h = F.relu(res + h)
        h = self.drop(h)

        res = h
        h = self.conv3(h,edge_index, edge_attr)
        h = self.norm3(h)
        h = F.relu(res + h)
        h = self.drop(h)

        res = h
        h = self.conv4(h,edge_index, edge_attr)
        h = self.norm4(h)
        h = F.relu(res + h)
        h = self.drop(h)

        
        res = h
        h = self.conv5(h,edge_index, edge_attr)
        h = self.norm5(h)
        h = F.relu(res + h)
        h = self.drop(h)


        g_feat = global_mean_pool(h, batch)        # (B, H)
        #h = global_mean_pool(h, batch)
        #h = self.final_pred(h)
        # h = F.sigmoid(h)
        p_feat = self.protein_encoder(prot_seq)    # (B, H)
        feat = torch.cat([g_feat, p_feat], dim=1)  # (B, 2H)
        out = self.final_pred(feat)
        return out


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        pt = torch.where(targets==1, probs, 1-probs)
        loss = self.alpha*(1-pt) ** self.gamma * BCE_loss
        if self.reduction == 'mean':
            return loss.mean()
        else:
            return loss.sum()

class WeightedFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # torch.tensor([w_neg, w_pos])

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets.float(), reduction='none')
        pt = torch.exp(-BCE_loss)

        focal_loss = (1 - pt) ** self.gamma * BCE_loss

        if self.alpha is not None:
            alpha_t = self.alpha[targets.long()]  # shape: (batch,)
            focal_loss = alpha_t * focal_loss

        return focal_loss.mean()


# === 第 1 部分：資料前處理（RDKit 分子圖 + 序列 embedding 整合） ===
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Data
import torch
import numpy as np

atom_vocab = ["C", "N", "O", "S", "F", "Cl", "Br", "I", "P"]
bond_vocab = [Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE, Chem.BondType.AROMATIC]

def atom_features(atom):
    return [
        atom_vocab.index(atom.GetSymbol()) if atom.GetSymbol() in atom_vocab else len(atom_vocab),
        atom.GetAtomicNum(),
        atom.GetDegree(),
        atom.GetTotalNumHs(),
        atom.GetFormalCharge(),
        int(atom.GetIsAromatic())
    ]

def bond_features(bond):
    bt = bond.GetBondType()
    return [
        bond_vocab.index(bt) if bt in bond_vocab else len(bond_vocab),
        int(bond.GetIsConjugated()),
        int(bond.IsInRing())
    ]

def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    Chem.Kekulize(mol, clearAromaticFlags=True)

    atom_feats = [atom_features(atom) for atom in mol.GetAtoms()]
    x = torch.tensor(atom_feats, dtype=torch.float)

    edge_index = []
    edge_attr = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_index += [[i, j], [j, i]]
        edge_attr += [bond_features(bond), bond_features(bond)]

    edge_index = torch.tensor(edge_index).t().long()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    return x, edge_index, edge_attr

# === 第 2 部分：訓練與優化設定 ===
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch_geometric.loader import DataLoader as GeoDataLoader
from torch.optim import AdamW
import os

# === Dataset example (需配合實際資料類型調整) ===
class MoleculeProteinDataset(torch.utils.data.Dataset):
    def __init__(self, data_list):  # data_list 為 list of Data (PyG + prot_seq)
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]

# === Oversampling ===
def get_sampler(data_list):
    labels = torch.tensor([d.y.item() for d in data_list])
    class_counts = torch.bincount(labels.long())
    class_weights = 1. / class_counts.float()
    sample_weights = class_weights[labels.long()]
    return WeightedRandomSampler(sample_weights, len(sample_weights))

# === 訓練迴圈 ===
def train(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        prot_seq = batch.prot_seq  # tensor of shape (B, L)
        logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, prot_seq)
        loss = loss_fn(logits.view(-1), batch.y.float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0
    correct = 0
    for batch in loader:
        batch = batch.to(device)
        prot_seq = batch.prot_seq
        logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, prot_seq)
        loss = loss_fn(logits.view(-1), batch.y.float())
        total_loss += loss.item() * batch.num_graphs

        preds = (logits.view(-1) > 0).long()
        correct += (preds == batch.y).sum().item()

    acc = correct / len(loader.dataset)
    return total_loss / len(loader.dataset), acc

# === 優化器設定 ===
def get_optimizer(model, lr=5e-5, weight_decay=1e-4):
    return AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

# === Checkpoint 與 Early Stopping ===
def save_checkpoint(model, optimizer, epoch, path):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }, path)

def load_checkpoint(model, optimizer, path, device):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch']

def early_stopping(val_loss_list, patience):
    if len(val_loss_list) < patience:
        return False
    recent = val_loss_list[-patience:]
    return recent[-1] > min(recent[:-1])



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GINE(
    node_feat_dim=NODE_FEAT_DIM,
    edge_feat_dim=EDGE_FEAT_DIM,
    hidden_dim=256,
    out_dim=1
).to(device)
train_loader = GeoDataLoader(MoleculeProteinDataset, batch_size=32, sampler=get_sampler(train_dataset))
optimizer = get_optimizer(model, lr=5e-5, weight_decay=1e-4)
loss_fn = FocalLoss

best_val_loss = float('inf')
patience, patience_counter = 5, 0

for epoch in range(1,31):
    model.train()
    total_loss = 0

    for batch in train_loader:
        out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        loss = loss_fn(out.view(-1), batch.y.float())
        loss.backward()
        optimizer.step()

        total_loss += loss.items()

    avg_loss = total_loss/len(train_loader)
    print(f"Epoch {epoch} | Train Loss: {avg_loss:.4f}")

    val_loss = evaluate(model, val_loader, loss_fn, device)
    print(f"Epoch {epoch} | Val Loss: {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), "best_model.pt")
        print("✅ Model saved.")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("⏹️ Early stopping triggered.")
            break