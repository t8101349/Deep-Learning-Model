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

# ---- GIN Model ----
class GIN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GIN, self).__init__()
        self.nn1 = nn.Sequential(
            nn.Linear(in_channels, hidden_channels), 
            nn.ReLU(), 
            nn.Linear(hidden_channels, hidden_channels),
            )
        self.nn2 = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels), 
            nn.ReLU(), 
            nn.Linear(hidden_channels, out_channels),
            )
        
        self.conv1 = GINConv(self.nn1)
        self.conv2 = GINConv(self.nn2)

        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.log_softmax(x, dim=1)
        return x

# ---- Training Setup ----
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GIN(dataset.num_node_features, 64, dataset.num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

# ---- Training Loop ----
model.train()
for epoch in range(1, 31):
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)  
    loss = criterion(out[data.train_mask], data.y[data.train_mask])  
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch:02d}, Loss: {loss.item():.4f}')