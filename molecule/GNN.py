import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import  global_mean_pool
from torch_geometric.nn.models import MLP
from torch_geometric.nn import GINEConv


class GINE_Model(torch.nn.Module):
    def __init__(self, node_feature_dim, hidden_dim, out_dim, edge_dim, num_layers=2):
        super(GINE_Model, self).__init__()
        self.out_dim = out_dim
        self.atom_encoder = nn.Linear(node_feature_dim, hidden_dim)
        self.edge_encoder = nn.Linear(edge_dim, hidden_dim)

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            mlp = MLP([hidden_dim, hidden_dim, hidden_dim], act="ReLU", batch_norm=True)
            # self.convs.append(Gt_GINE_Conv(hidden_dim, edge_dim))
            self.convs.append(GINEConv(nn=mlp, train_eps=False))
            
        self.final_mlp = MLP([hidden_dim, hidden_dim, out_dim], act="ReLU", batch_norm=True)

    def forward(self, data):
        edge_index, edge_attr, batch = data.edge_index, data.edge_attr, data.batch
        if data.dataset == "QM7b":
            x = data.f_d
        else:
            x = data.x

        x = self.atom_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)
        
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = F.relu(x)
            
        x = global_mean_pool(x, batch)
        x = self.final_mlp(x)
        return x




class Gt_GINE_Conv(nn.Module):
    def __init__(self, hidden_dim, edge_dim):
        super().__init__()

        self.gine1 = GINEConv(MLP([hidden_dim, hidden_dim, hidden_dim]))
        self.gine2 = GINEConv(MLP([hidden_dim, hidden_dim, hidden_dim]))
        self.gine12 = GINEConv(MLP([hidden_dim, hidden_dim, hidden_dim]))
        self.gine21 = GINEConv(MLP([hidden_dim, hidden_dim, hidden_dim]))

        self.c1 = nn.Parameter(torch.tensor(0.8))
        self.c2 = nn.Parameter(torch.tensor(0.1))
        self.c12 = nn.Parameter(torch.tensor(0.05))
        self.c21 = nn.Parameter(torch.tensor(0.05))
        
        self.edge_encoder1 = nn.Linear(edge_dim, hidden_dim) 
        self.edge_encoder2 = nn.Linear(edge_dim, hidden_dim) 

    def forward(self, x, data):
        x_initial = x

        edge_index1, edge_attr1 = data.edge_index_A1, data.edge_attr_A1
        edge_index2, edge_attr2 = data.edge_index_A2, data.edge_attr_A2
        
        enc_edge_attr1 = self.edge_encoder1(edge_attr1)
        enc_edge_attr2 = self.edge_encoder2(edge_attr2)

        
        h_path1 = self.gine1(x, edge_index1, enc_edge_attr1) 
        h_path2 = self.gine2(x, edge_index2, enc_edge_attr2) 
        h_path12 = self.gine12(F.relu(h_path1), edge_index2, enc_edge_attr2)
        h_path21 = self.gine21(F.relu(h_path2), edge_index1, enc_edge_attr1)


        h_final = x_initial + self.c1 * h_path1 + self.c2 * h_path2 + self.c12 * h_path12 + self.c21 * h_path21

        return h_final


class Gt_GINE_Model(torch.nn.Module):
    def __init__(self, node_feature_dim, hidden_dim, out_dim, edge_dim, num_layers=2):
        super(Gt_GINE_Model, self).__init__()
        self.out_dim = out_dim
        self.atom_encoder = nn.Linear(node_feature_dim, hidden_dim) 

        self.convs = torch.nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(num_layers):
            # mlp = MLP([hidden_dim, hidden_dim, hidden_dim], act="ReLU", batch_norm=True)
            self.convs.append(Gt_GINE_Conv(hidden_dim, edge_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))
            
        self.final_mlp = MLP([hidden_dim, hidden_dim, out_dim], act="ReLU", batch_norm=True)

    def forward(self, data):
        batch = data.batch
        x = data.f_d
        
        x = self.atom_encoder(x)
        
        for i, conv in enumerate(self.convs):
            x = conv(x, data)
            x = self.norms[i](x)
            x = F.relu(x) 

        x = global_mean_pool(x, batch)
        x = self.final_mlp(x)
        return x






