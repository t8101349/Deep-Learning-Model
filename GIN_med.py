import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import GINConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import Sequential as GNNSequential
import torch.nn.functional as F

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

class GIN(nn.Module):
    def __init__(self, in_dim, hidden_dim=256, out_dim=1):
        super().__init__()
        self.nn1 = MLP(in_dim, hidden_dim)
        self.nn2 = MLP(hidden_dim, hidden_dim)
        self.nn3 = MLP(hidden_dim, hidden_dim)
        self.nn4 = MLP(hidden_dim, hidden_dim)
        self.nn5 = MLP(hidden_dim, hidden_dim)

        self.conv1 = GINConv(self.nn1)
        self.conv2 = GINConv(self.nn2)
        self.conv3 = GINConv(self.nn3)
        self.conv4 = GINConv(self.nn4)
        self.conv5 = GINConv(self.nn5)

        self.norm1 = nn.BatchNorm1d(hidden_dim)
        self.norm2 = nn.BatchNorm1d(hidden_dim)
        self.norm3 = nn.BatchNorm1d(hidden_dim)
        self.norm4 = nn.BatchNorm1d(hidden_dim)
        self.norm5 = nn.BatchNorm1d(hidden_dim)

        self.drop = nn.Dropout(0.3)

        self.res_proj1 = nn.Linear(in_dim, hidden_dim)
        #self.res_proj2 = nn.Linear(hidden_dim, out_dim)

        self.final_pred = MLP(hidden_dim, out_dim, hidden_dim=128)
    
    def forward(self, h, edge_index, batch):
        res = self.res_proj1(h)

        h = self.conv1(h,edge_index)
        h = self.norm1(h)
        h = F.relu(h)
        h = self.drop(h)

        res = h
        h = self.conv2(h,edge_index)
        h = self.norm2(h)
        h = F.relu(res + h)
        h = self.drop(h)

        res = h
        h = self.conv3(h,edge_index)
        h = self.norm3(h)
        h = F.relu(res + h)
        h = self.drop(h)

        res = h
        h = self.conv4(h,edge_index)
        h = self.norm4(h)
        h = F.relu(res + h)
        h = self.drop(h)

        
        res = h
        h = self.conv5(h,edge_index)
        h = self.norm5(h)
        h = F.relu(res + h)
        h = self.drop(h)

        h = global_mean_pool(h, batch)
        h = self.final_pred(h)
        # h = F.sigmoid(h)
        return h