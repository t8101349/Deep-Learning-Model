import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import GINConv
from torch_geometric.nn import Sequential as GNNSequential
import torch.nn.functional as F


# ---- Load Dataset ----
dataset = Planetoid(root='data/Cora', name='Cora')
data = dataset[0].to('cuda' if torch.cuda.is_available() else 'cpu')

'''
# ---- Sampling Module ----
train_loader = NeighborLoader(
    data,
    num_neighbors=[10, 10],
    batch_size=64,
    input_nodes=data.train_mask,
)
'''

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim= None):
        super().__init__()
        hidden_dim = hidden_dim or output_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), 
            nn.ReLU(), 
            nn.Linear(hidden_dim, output_dim),
            )
    def forward(self, x):
        return self.net(x)

# ---- GIN Model ----
class GIN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GIN, self).__init__()
        self.nn1 = MLP(in_channels,hidden_channels)
        self.nn2 = MLP(hidden_channels,hidden_channels)
        self.nn3 = MLP(hidden_channels, out_channels, hidden_channels)

        
        self.conv1 = GINConv(self.nn1)
        self.conv2 = GINConv(self.nn2)
        self.conv3 = GINConv(self.nn3)

        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.drop = nn.Dropout(0.5)

        self.res_proj1 = nn.Linear(in_channels, hidden_channels)
        self.res_proj2 = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        res = self.res_proj1(x)
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(res + x)
        x = self.drop(x)

        res = x
        x = self.conv2(x, edge_index)
        x = self.bn1(x)
        x = F.relu(res + x)
        x = self.drop(x)

        res = self.res_proj2(x)
        x = self.conv3(x, edge_index)
        x = self.bn2(x)
        x = F.log_softmax(res + x, dim=1)
        return x

# ---- Training Setup ----
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GIN(dataset.num_node_features, 64, dataset.num_classes).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()

# ---- Training Loop ----
model.train()
for epoch in range(1, 51):
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)  
    loss = criterion(out[data.train_mask], data.y[data.train_mask])  
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch:02d}, Loss: {loss.item():.4f}')