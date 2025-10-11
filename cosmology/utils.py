import torch, h5py, math
from torch_cluster import radius_graph
from torch_geometric.data import Dataset, Data
from torch_cluster import radius_graph


def pbc_delta(delta, L):
    if L is None: return delta
    return delta - L * torch.round(delta / L)

def rbf_encode(dist, Rc, K, device):
    centers = torch.linspace(0., Rc, K, device=device)            # [K]
    delta   = centers[1] - centers[0]
    gamma   = 1.0 / (delta * delta)
    return torch.exp(-gamma * (dist - centers)**2) 

def rbf_encode_unit(dist, Rc, K=32, eps=1e-8):
    if torch.is_tensor(Rc):
        Rc = float(Rc.detach().cpu().item())
    Rc = max(float(Rc), eps)

    dist01 = (dist / Rc).clamp(0.0, 1.0)
    dev, dt = dist01.device, dist01.dtype

    centers = torch.linspace(0.0, 1.0, steps=K, device=dev, dtype=dt)
    delta   = centers[1] - centers[0] if K > 1 else torch.tensor(1.0, device=dev, dtype=dt)
    gamma   = 1.0 / (delta * delta)
    return torch.exp(-gamma * (dist01 - centers)**2)

def pbc_center_and_wrap(pos, L):
    """
    pos: [N,3] in [0,L)
    return: centered & wrapped to [-L/2, L/2)
    """
    out = []
    for d in range(3):
        x = pos[:, d]                         # [0, L)
        theta = 2*math.pi * x / L
        s, c = torch.sin(theta).mean(), torch.cos(theta).mean()
        mu_theta = torch.atan2(s, c)         # ∈ (-π, π]
        mu = (mu_theta % (2*math.pi)) * L / (2*math.pi)  # back to [0, L)
        xc = x - mu
        # wrap to [-L/2, L/2)
        xc = xc - L*torch.round(xc / L)
        out.append(xc)
    return torch.stack(out, dim=1)


def build_radius_graph(pos, L, R, max_nn=10**9, num_rbf=32):
    edge_index = radius_graph(pos, r=R, loop=False, max_num_neighbors=max_nn)  # [2, E]
    row, col = edge_index
    delta = pos[row] - pos[col]
    delta = pbc_delta(delta, L)               
    dist  = delta.norm(dim=1)                 
    eattr = rbf_encode(dist, R, num_rbf=num_rbf)  
    return edge_index, eattr

@torch.no_grad()
def build_graph(pos, L, R, self_loops=False, device=None):
    """
    PBC radius graph.
    Returns:
      node_x  : [N,3] coordinates (optionally centered)
      edge_index: [2,E]
      edge_attr : [E,3] -> [ d_ij/R, <x_i,x_j>, <x_i, x_i-x_j> ]
    """
    if device is None:
        device = pos.device
    x = pos

    N = x.size(0)
    delta = x.unsqueeze(1) - x.unsqueeze(0)   # [N,N,3]
    delta = pbc_delta(delta, L)
    dist  = delta.norm(dim=-1)                # [N,N]

    mask = dist <= R
    if not self_loops:
        mask &= ~torch.eye(N, dtype=torch.bool, device=device)

    row, col = mask.nonzero(as_tuple=True)
    if row.numel() == 0:
        return x, torch.empty(2,0, dtype=torch.long, device=device), torch.empty(0,3, dtype=torch.float32, device=device)

    d_hat  = (dist[row, col] / R).unsqueeze(-1)
    dp_xx  = (x[row] * x[col]).sum(-1, keepdim=True)
    dp_xdx = (x[row] * delta[row, col]).sum(-1, keepdim=True)

    edge_attr  = torch.cat([d_hat, dp_xx, dp_xdx], dim=-1)  # [E,3]
    edge_index = torch.stack([row, col], dim=0)              # [2,E]
    return edge_index, edge_attr

def MSE_loss(ypred, y):
    return torch.mean((ypred - y)**2)

def variance(y):
    #compute mean vector (per feat), then average over variance per element
    mean = y.mean(axis=0)
    return torch.mean((y - mean)**2) 

def compute_R2(ypred, y):
    mse = MSE_loss(ypred, y)
    var = variance(y)
    return 1 - mse/var

def bootstrap_r2(Y_pred_test, Y_test, num_bootstrap=1000, seed=42):
    """
    Compute bootstrapped R² scores from predictions and ground truth.

    Args:
        Y_pred_test (torch.Tensor): Predicted values, shape [n]
        Y_test (torch.Tensor): Ground truth values, shape [n]
        num_bootstrap (int): Number of bootstrap samples
        seed (int, optional): Random seed for reproducibility

    Returns:
        torch.Tensor: Bootstrap R² scores mean and std
    """
    if seed is not None:
        torch.manual_seed(seed)

    n = Y_test.shape[0]
    r2_scores = torch.zeros(num_bootstrap)

    for i in range(num_bootstrap):
        idx = torch.randint(0, n, (n,), device=Y_test.device)  # sample with replacement
        y_pred_sample = Y_pred_test[idx]
        y_true_sample = Y_test[idx]
        r2_scores[i] = compute_R2(y_pred_sample, y_true_sample)

    return r2_scores.mean(), r2_scores.std()


class CamelsPointCloud(Dataset):
    def __init__(self, h5_path, group='LH', params_h5=None, target='om',
                 L=100.0, R1=3.0, R2=None, num_rbf=32, center=True, transform=None):
        super().__init__(None, transform)
        self.h5_path = h5_path
        self.group = group
        self.params_h5 = params_h5 or h5_path
        self.target = target
        self.L = float(L)
        self.R1 = float(R1)
        self.R2 = float(R2) if R2 is not None else None
        self.num_rbf = int(num_rbf)
        self.center = center
        with h5py.File(h5_path, 'r') as f:
            self.keys = sorted(list(f[self.group].keys()))  # e.g. LH_0 ... LH_{N-1}
        with h5py.File(self.params_h5, 'r') as f:
            p = f['params']
            if target == 'om':
                self.y_all = torch.tensor(p['Omega_m'][:], dtype=torch.float32)
            else:
                self.y_all = torch.tensor(p['sigma_8'][:], dtype=torch.float32)

    def len(self): return len(self.keys)

    def get(self, idx):
        with h5py.File(self.h5_path, 'r') as f:
            g = f[self.group][self.keys[idx]]
            x = torch.tensor(g['X'][:], dtype=torch.float32)
            y = torch.tensor(g['Y'][:], dtype=torch.float32)
            z = torch.tensor(g['Z'][:], dtype=torch.float32)
            pos = torch.stack([x, y, z], dim=1)  # [N,3]

        pos_c = pbc_center_and_wrap(pos, self.L) if self.center else pos

        # ei1, ea1 = build_radius_graph(pos, self.L, self.R1, max_nn=64, num_rbf=self.num_rbf)
        ei1, ea1 = build_graph(pos_c, self.L, self.R1, self_loops=False)
        
        data = Data(x=pos_c, pos=pos, y=self.y_all[idx].unsqueeze(0), L=self.L)  
        data.edge_index = ei1
        data.edge_attr  = ea1

        if self.R2 is not None:
            ei2, ea2 = build_graph(pos_c, self.L, self.R2, self_loops=False)
            # ei2, ea2 = build_radius_graph(pos_c, self.L, self.R2, max_nn=64, num_rbf=self.num_rbf)
            data.edge_index2 = ei2
            data.edge_attr2  = ea2

        return data
