class MPNNLayer(MessagePassing):
    def __init__(self, emb_dim=64, edge_dim=4, aggr='add'):
        super().__init__(aggr=aggr)
    
        self.emb_dim = emb_dim
        self.edge_dim = edge_dim
        self.mlp_msg = nn.Sequential(
            nn.Linear(2 * emb_dim + edge_dim, emb_dim), nn.BatchNorm1d(emb_dim), nn.ReLU(),
            nn.Linear(emb_dim, emb_dim), nn.BatchNorm1d(emb_dim), nn.ReLU()
        )

        self.mlp_upd = nn.Sequential(
            nn.Linear(2 * emb_dim, emb_dim), nn.BatchNorm1d(emb_dim), nn.ReLU(),
            nn.Linear(emb_dim, emb_dim), nn.BatchNorm1d(emb_dim), nn.ReLU()
        )

        self.res_proj = nn.Linear(2 * emb_dim + edge_dim, emb_dim)
        self.norm = nn.BatchNorm1d(emb_dim)
        self.dropout = nn.Dropout(0.2)

    
    def message(self, h_i, h_j, edge_attr):
        msg = torch.cat([h_i, h_j, edge_attr], dim=-1)
        out = self.mlp_msg(msg)
        return out
    
    def aggregate(self, inputs, index):
        return scatter_sum(inputs, index, dim=self.node_dim, reduce=self.aggr)
    
    def update(self, aggr_out, h):
        upd_out = torch.cat([h, aggr_out], dim=-1)
        out = self.mlp_upd(upd_out)
        return out
    
    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(emb_dim={self.emb_dim}, aggr={self.aggr})')

    
    def forward(self, h, edge_index, edge_attr):
        res = self.res_proj(h)

        h_i = h[edge_index[0]]
        h_j = h[edge_index[1]]
        h_edge = self.message(h_i, h_j, edge_attr)
        h_msg = self.aggregate(h_edge,edge_index[1])
        h_node = self.update(h_msg, h)
        out = self.norm(res+self.dropout(h_node))
        # out = self.propagate(edge_index, h=h, edge_attr=edge_attr)

        return out
    

class MPNNModel(nn.Module):
    def __init__(self, num_layers=4, input_dim=11, edge_dim=4, out_dim=1, emb_dim=64):
        super().__init__()
    
        self.nn_in = nn.Linear(input_dim, emb_dim)
    
        # Stack of MPNN layers
        self.convs = torch.nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(MPNNLayer(emb_dim, edge_dim, aggr='add'))
    
        self.pool = global_mean_pool
    
    def forward(self, data): #PyG.Data - batch of PyG graphs
    
        #h = self.nn_in(torch.from_numpy(np.unpackbits(data.x,-1).float()) )
        h = self.nn_in(F_unpackbits(data.x, -1).float())

        for conv in self.convs:
            res = h
            h = res + conv(h, data.edge_index.long(), F_unpackbits(data.edge_attr,-1).float())  # (n, d) -> (n, d)
    
        h_graph = self.pool(h, data.batch)  
        return h_graph

# our prediction model here 
class Net(nn.Module):
    def __init__(self, ):
        super().__init__()
    
        self.output_type = ['infer', 'loss']
    
        graph_dim=96
        input_dim = 11
        edge_dim = 4
        self.smile_encoder = MPNNModel(
            num_layers=4, input_dim=input_dim, edge_dim=edge_dim, emb_dim=graph_dim,
        )
        self.bind = nn.Sequential(
            nn.Linear(graph_dim, 1024),
            #nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(1024, 1024),
            #nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            #nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, 3),
        )
    
    def forward(self, batch):
        graph = batch['graph']
        x = self.smile_encoder(graph) 
        bind = self.bind(x)
    
        # --------------------------
        output = {}
        if 'loss' in self.output_type:
            target = batch['bind']
            output['bce_loss'] = F.binary_cross_entropy_with_logits(bind.float(), target.float())
    
        if 'infer' in self.output_type:
            output['bind'] = torch.sigmoid(bind)
    
        return output