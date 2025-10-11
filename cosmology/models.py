import torch, torch.nn as nn, torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.nn import global_mean_pool
from utils import pbc_delta, rbf_encode, rbf_encode_unit

class EGCL(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim,
                 use_pbc=True, Rc=3, rbf_k=32, coord_norm=True, tanh_out=True):
        super().__init__()
        self.use_pbc = use_pbc
        self.coord_norm = coord_norm
        self.tanh_out = tanh_out
        self.Rc, self.rbf_k = Rc, rbf_k

        self.edge_mlp = nn.Sequential(
            nn.Linear(2*in_dim + rbf_k, hid_dim),
            nn.SiLU(),
            nn.Linear(hid_dim, hid_dim),
            nn.SiLU(),
        )
        self.coord_mlp = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.SiLU(),
            nn.Linear(hid_dim, 1, bias=False),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(in_dim + hid_dim, hid_dim),
            nn.SiLU(),
            nn.Linear(hid_dim, out_dim),
        )

    def forward(self, h, pos, edge_index, L=None, chunk_size=1_000_000):
        N = h.size(0); E = edge_index.size(1)
        agg_h = h.new_zeros((N, self.edge_mlp[-2].out_features))  # hid
        dpos  = pos.new_zeros((N, 3))

        for s in range(0, E, chunk_size):
            sl  = slice(s, min(s+chunk_size, E))
            row = edge_index[0, sl]; col = edge_index[1, sl]

            cdiff = pos[row] - pos[col]
            if self.use_pbc and (L is not None):
                cdiff = pbc_delta(cdiff, L)

            dist = cdiff.norm(dim=-1, keepdim=True)                 # d = ||Δx||
            geom = rbf_encode(dist, self.Rc, self.rbf_k, device=pos.device)  # [E,K]
            e_in = torch.cat([h[row], h[col], geom], dim=1)

            emsg = self.edge_mlp(e_in)

            trans = self.coord_mlp(emsg)
            if self.tanh_out: trans = torch.tanh(trans)
            scatter_add(cdiff * trans, row, dim=0, out=dpos)
            scatter_add(emsg,          row, dim=0, out=agg_h)

        if self.coord_norm:
            deg = scatter_add(
                torch.ones_like(edge_index[0], dtype=pos.dtype, device=pos.device),
                edge_index[0], dim=0, dim_size=N
            ).clamp(min=1).view(-1, 1)
            dpos = dpos / deg

        pos = pos + (pbc_delta(dpos, L) if (self.use_pbc and (L is not None)) else dpos)
        h   = h + self.node_mlp(torch.cat([h, agg_h], dim=1))
        return h, pos


class EGNN(nn.Module):
    def __init__(self, node_feature_dim=1, hid_dim=128, out_dim=1, n_layers=3, Rc=3, use_pbc=True):
        super().__init__()
        self.embed  = nn.Linear(node_feature_dim, hid_dim)
        self.layers = nn.ModuleList(
            [EGCL(hid_dim, hid_dim, hid_dim, Rc=Rc, use_pbc=use_pbc) for _ in range(n_layers)]
        )
        self.readout = nn.Sequential(
            nn.Linear(hid_dim, hid_dim), nn.SiLU(), nn.Linear(hid_dim, out_dim)
        )

    def forward(self, data):
        h   = self.embed(data.x)
        pos = data.pos
        ei  = data.edge_index
        L   = float(data.L) if not torch.is_tensor(data.L) else float(
            data.L.item() if data.L.ndim == 0 else data.L[0].item()
        )
        for layer in self.layers:
            h, pos = layer(h, pos, ei, L=L)
        g = global_mean_pool(h, data.batch)
        return self.readout(g)




class EGCL_GT(nn.Module):
    def __init__(self, hid_dim, Rc1, Rc2, rbf_k=32,
                 use_pbc=True, coord_norm=True, tanh_out=True, dropout=0.0):
        super().__init__()
        self.use_pbc, self.coord_norm, self.tanh_out = use_pbc, coord_norm, tanh_out
        self.Rc1, self.Rc2, self.rbf_k = Rc1, Rc2, rbf_k

        in_edge = 2*hid_dim + rbf_k
        self.edge_norm = nn.LayerNorm(in_edge)
        self.edge_mlp  = nn.Sequential(
            nn.Linear(in_edge, hid_dim), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hid_dim, hid_dim), nn.SiLU(), nn.Dropout(dropout),
        )
        self.coord_mlp = nn.Sequential(
            nn.Linear(hid_dim, hid_dim), nn.SiLU(),
            nn.Linear(hid_dim, 1, bias=False),
        )
        nn.init.constant_(self.coord_mlp[-1].weight, 0.0)

        self.node_mlp  = nn.Sequential(
            nn.Linear(hid_dim + hid_dim, hid_dim),
            nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hid_dim, hid_dim),
        )

        self.alpha_h_logits = nn.Parameter(torch.tensor([0.6, 0.3, 0.05, 0.05]))  
        self.alpha_x_logits = nn.Parameter(torch.tensor([0.8, 0.15, 0.025, 0.025])) 

    def _one_pass(self, h, pos, ei, Rc, L, chunk_size=1_000_000):
        N, E = h.size(0), ei.size(1)
        agg_h = h.new_zeros((N, self.edge_mlp[0].out_features))
        dpos  = pos.new_zeros((N, 3))
        for s in range(0, E, chunk_size):
            sl   = slice(s, min(s+chunk_size, E))
            row, col = ei[0, sl], ei[1, sl]
            cdiff = pos[row] - pos[col]
            if self.use_pbc and (L is not None):
                cdiff = pbc_delta(cdiff, L)
            dist = cdiff.norm(dim=-1, keepdim=True)
            geom = rbf_encode(dist, Rc, self.rbf_k, device=pos.device)

            e_in = torch.cat([h[row], h[col], geom], dim=1)
            e_in = self.edge_norm(e_in)
            emsg = self.edge_mlp(e_in)

            trans = self.coord_mlp(emsg)
            if self.tanh_out: trans = torch.tanh(trans)
            scatter_add(cdiff * trans, row, dim=0, out=dpos)
            scatter_add(emsg,          row, dim=0, out=agg_h)

        if self.coord_norm:
            deg = scatter_add(torch.ones_like(ei[0], dtype=pos.dtype, device=pos.device),
                              ei[0], dim=0, dim_size=N).clamp(min=1).view(-1,1)
            dpos = dpos / deg
        dh = self.node_mlp(torch.cat([h, agg_h], dim=1))
        return dh, dpos

    def forward(self, h, pos, ei1, ei2, L):
        # --- H1 ---
        dh1, dpos1 = self._one_pass(h, pos, ei1, self.Rc1, L)
        if self.use_pbc:
            dpos1 = pbc_delta(dpos1, L)     
        h1   = h   + dh1
        pos1 = pos + dpos1

        # --- H2 ---
        dh2, dpos2 = self._one_pass(h, pos, ei2, self.Rc2, L)
        if self.use_pbc:
            dpos2 = pbc_delta(dpos2, L)    
        h2   = h   + dh2
        pos2 = pos + dpos2

        # --- H1→H2 ---
        dh12, dpos12 = self._one_pass(h1, pos1, ei2, self.Rc2, L)
        if self.use_pbc:
            dpos12 = pbc_delta(dpos12, L)  

        # --- H2→H1 ---
        dh21, dpos21 = self._one_pass(h2, pos2, ei1, self.Rc1, L)
        if self.use_pbc:
            dpos21 = pbc_delta(dpos21, L)   

        # --- fuse ---
        alpha_h = torch.softmax(self.alpha_h_logits, dim=0)
        alpha_x = torch.softmax(self.alpha_x_logits, dim=0)

        h_next   = h   + alpha_h[0]*dh1   + alpha_h[1]*dh2   + alpha_h[2]*dh12   + alpha_h[3]*dh21
        dpos_f   =       alpha_x[0]*dpos1 + alpha_x[1]*dpos2 + alpha_x[2]*dpos12 + alpha_x[3]*dpos21
        if self.use_pbc:
            dpos_f = pbc_delta(dpos_f, L)  
        pos_next = pos + dpos_f
        return h_next, pos_next



class EGNN_GT(nn.Module):
    def __init__(self, node_feature_dim=1, hid_dim=128, out_dim=1,
                 n_layers=3, Rc1=2, Rc2=4, rbf_k=32, use_pbc=True, dropout=0.0):
        super().__init__()
        self.embed  = nn.Linear(node_feature_dim, hid_dim)
        self.layers = nn.ModuleList([
            EGCL_GT(hid_dim, Rc1=Rc1, Rc2=Rc2, rbf_k=rbf_k,
                    use_pbc=use_pbc, coord_norm=True, tanh_out=True, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.readout = nn.Sequential(nn.Linear(hid_dim, hid_dim), nn.SiLU(),
                                     nn.Linear(hid_dim, out_dim))

    def forward(self, data):
        h   = self.embed(data.x)
        pos = data.pos
        ei1 = data.edge_index1
        ei2 = data.edge_index2
        L   = float(data.L) if not torch.is_tensor(data.L) else float(data.L.flatten()[0])

        for layer in self.layers:
            h, pos = layer(h, pos, ei1, ei2, L)

        g = global_mean_pool(h, data.batch)
        return self.readout(g)
