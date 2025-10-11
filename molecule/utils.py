import torch
import numpy as np
from torch_geometric.utils import to_dense_adj
from torch_geometric.utils import scatter
from torch_geometric.transforms import BaseTransform
import torch.nn.functional as F
from torch_geometric.data import Dataset 

class ListDataset(Dataset):
    def __init__(self, data_list):
        super(ListDataset, self).__init__()
        self.data_list = data_list

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        if isinstance(idx, (list, tuple, torch.Tensor)):
            return ListDataset([self.data_list[i] for i in idx])
        else:
            return self.data_list[idx]


def get_fs(data):
  '''
  input: Data (graph) from pytorch geometric dataset
    edge_index=[2, num_edges]
    edge_attr=[num_edges]
    y=[num_targets]
    x=[num_nodes, 1]

  output: updated DataBatch with the following additional invariant features
   f_d is the set of diagonal edge attributes,
   f_o is the set of off-diagonal (upper) edge attributes,
   f_star is \sum_{i \neq j} X_ii X_ij where X is the n by n edge attribute graph
  '''
  #get f_d, f_o
  loop_mask = data.edge_index[0] == data.edge_index[1] #🌟diagonal elements(self-loop)
  data.f_d = data.edge_attr[loop_mask].unsqueeze(1) #🌟shape: [num_diag, 1]
  data.f_o = data.edge_attr[~loop_mask].unsqueeze(1) #🌟shape: [num_offdiag, 1]

  #get f_star
  sum_fo = scatter(data.edge_attr, data.edge_index[1], dim=0) #sum over rows
  data.f_star = (data.f_d.squeeze(1) @ ( sum_fo - data.f_d.squeeze(1))).reshape((1,1))
  return data

#The kernel trick that projects a scalar to higher-dimensional space
#src: https://arxiv.org/pdf/1305.7074.pdf, Appendix B
def binary_expansion(f, num_radial, theta=1):
  '''
  input: f (bs x 1); num_radial - number of basis expansion
  output: phi(f) (bs x num_radial)
  phi(f) = [..., sigmoid(f-theta/theta), sigmoid(f/theta), sigmoid(f+theta/thera),...]
  '''
  bs = f.shape[0]
  out = torch.zeros((bs, num_radial))
  max_val = (num_radial - 1)//2
  offsets = np.arange(start=-max_val, stop=max_val+1) #symmetric construction
  for i, offset in enumerate(offsets):
    out[:, i:(i+1)] = F.sigmoid( (f - theta*offset) / theta )
  return out

def get_binary_expansion(data, num_radial=100, theta=1):
  '''
  Apply binary_expansion for all fs
  try: num_radial \in {100, 1000}
  '''
  #print(f"num_basis={num_radial}")
  data.f_d = binary_expansion(data.f_d, num_radial, theta)
  data.f_o = binary_expansion(data.f_o, num_radial, theta)
  data.f_star = binary_expansion(data.f_star, num_radial, theta)
  return data

class ExtractTarget(BaseTransform):
    def __init__(self, target):
        self.target = target
    def forward(self, data):
        data.y = data.y[:, self.target:(self.target+1)]
        return data


def get_graph_tuple_QM7b(raw_ds_with_features, thresh=2):
    processed_set = []
    for data in raw_ds_with_features:
        n = data.num_nodes
        feature_dim = data.f_d.shape[1] 
        
        feature_adj = torch.zeros(n, n, feature_dim, device=data.f_d.device)
        
        for i in range(n):
            feature_adj[i, i] = data.f_d[i]

        off_diag_mask = ~torch.eye(n, dtype=torch.bool)
        feature_adj[off_diag_mask] = data.f_o.view(-1, feature_dim)

        C = to_dense_adj(
            data.edge_index, 
            edge_attr=data.edge_attr, 
            max_num_nodes=n
            )[0]     
        C = 0.5 * (C + C.T); C.fill_diagonal_(0.)
        
        mask = (torch.abs(C) >= thresh)
        row, col = torch.nonzero(mask, as_tuple=True)
        data.edge_index_A1 = torch.stack([row, col], dim=0)
        data.edge_attr_A1 = feature_adj[row, col]
        data.edge_attr1 = C[row, col]
        
        mask = (torch.abs(C) < thresh)
        row, col = torch.nonzero(mask, as_tuple=True)
        data.edge_index_A2 = torch.stack([row, col], dim=0)
        data.edge_attr_A2 = feature_adj[row, col]
        data.edge_attr2 = C[row, col]

        processed_set.append(data)

    processed_set = ListDataset(processed_set)
    return processed_set


