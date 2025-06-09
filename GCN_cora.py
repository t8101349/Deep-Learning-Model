import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import GCNConv
from torch_geometric.nn import Sequential as GNNSequential
import torch.nn.functional as F


# ---- Load Dataset ----
dataset = Planetoid(root='data/Cora', name='Cora')
data = dataset[0].to('cuda' if torch.cuda.is_available() else 'cpu')

# ---- Model ----
class GCN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim = 64):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, output_dim)
        self.norm1 = nn.BatchNorm1d(hidden_dim)
        self.norm2 = nn.BatchNorm1d(hidden_dim)
        self.norm3 = nn.BatchNorm1d(output_dim)
        self.drop = nn.Dropout(0.5)

        self.res_proj = nn.Linear(input_dim, hidden_dim)
        self.res_proj2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        res = self.res_proj(x)

        x = self.conv1(x, edge_index)
        x = self.norm1(x)
        x = F.relu(res + x)
        x = self.drop(x)

        res = x
        x = self.conv2(x, edge_index)
        x = self.norm2(x)
        x = F.relu(res + x)
        x = self.drop(x)
        
        res = self.res_proj2(x)
        x = self.conv3(x, edge_index)
        x = self.norm3(x)
        x = F.log_softmax(res + x, dim = 1)
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN(dataset.num_node_features, dataset.num_classes, 64).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = nn.NLLLoss()

'''
# ---- Train Function ----
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

# ---- Evaluation ----
def evaluate(mask):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        correct = (pred[mask] == data.y[mask]).sum()
        return int(correct) / int(mask.sum())
'''
# ---- Training Loop ----

model.train()
for epoch in range(1, 31):
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)  
    loss = criterion(out[data.train_mask], data.y[data.train_mask])  
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch:02d}, Loss: {loss.item():.4f}')