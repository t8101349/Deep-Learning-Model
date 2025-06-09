import torch
import torch.nn as nn
from torch_scatter import scatter_sum

class GNNLayer(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim):
        super().__init__()
        self.f_edge = MLP(input_dim=2*node_dim + edge_dim, output_dim=hidden_dim)
        self.f_node = MLP(input_dim=node_dim + hidden_dim + node_dim, output_dim=hidden_dim)

    def forward(self, x, pre_h_node, edge_index, edge_attr):
        # x: [num_nodes, node_dim]
        # edge_index: [2, num_edges]
        # edge_attr: [num_edges, edge_dim]
        h_i = x[edge_index[0]]
        h_j = x[edge_index[1]]
        h_edge = self.f_edge(torch.cat([h_i, h_j, edge_attr], dim=-1))
        
        # Aggregate messages 
        h_msg = scatter_sum(h_edge, edge_index[1], dim=0, dim_size=x.size(0))
        
        h_node = self.f_node(torch.cat([pre_h_node, h_msg, x], dim=-1))

        return h_node

class MlP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or output_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
    def forward(self, x):
        return self.net(x)

class MPNN(nn.Module):
    def __init__(self, node_dim,edge_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.f_edge = MLP(input_dim = 2*node_dim + edge_dim, output_dim = hidden_dim)
        self.f_node = MLP(input_dim = 2*node_dim + hidden_dim, output_dim = hidden_dim)

        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # 若 node_dim 與 hidden_dim 不同，需要升維以便做 residual
        self.residual_project = nn.Linear(node_dim, hidden_dim) if node_dim != hidden_dim else nn.Identity()

    def forward(self, x, pre_node, edge_index, edge_attr):
        if pre_node is None:
            pre_node = x
        
        h_i = pre_node[edge_index[0]]
        h_j = pre_node[edge_index[1]]
        h_edge = self.f_edge(torch.cat([h_i, h_j, edge_attr], dim=-1))

        h_msg = scatter_sum(h_edge, edge_index[1], dim=0, dim_size= x.size(0))

        h_node = self.f_node(torch.cat([pre_node, h_msg, x], dim=-1))

        res = self.residual_project(pre_node)
        out = self.norm(res+self.dropout(h_node))
        return out

class MPNNStack(nn.Module):
    def __init__(self, num_layers, node_dim, edge_dim, hidden_dim):
        super().__init__()
        self.layers = nn.ModuleList([
            MPNN(node_dim, edge_dim, hidden_dim) for _ in range(num_layers)
        ])

    def forward(self, x, h, edge_index, edge_attr):
        if not h:
            h = x
        for layer in self.layers:
            h = layer(x, h, edge_index, edge_attr)
        return h