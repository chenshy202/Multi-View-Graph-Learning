import os, h5py, torch
from torch_geometric.data import Data
from torch_cluster import radius_graph
from pathlib import Path
import math
from utils import pbc_delta, pbc_center_and_wrap
import numpy as np

path = "/data/schen355/cosmodata/"
h5_root = "CosmoBench_CAMELS" #"CosmoBench_CAMELS-SAM" 
target = "om" #"sigma_8"
group  = "LH"
L = 25
num = "ALL"
# Rlist = np.arange(10, 61, 5)
# Rlist = np.arange(0.2, 1, 0.2) #np.arange(1, 10.5, 1) 
Rlist = [0.5, 1.5, 2.5, 3.5]
for R in Rlist:
    print(f"Processing R = {float(R):.1f}")
    data_dir   = f"{path}{h5_root}"
    out_dir    = Path(data_dir) / f"{target}/{float(R):.1f}"
    out_dir.mkdir(parents=True, exist_ok=True)
    if h5_root == "CosmoBench_CAMELS-SAM":
        L = 100
        num = "top5000"
    for split in ["train","val","test"]:
        h5_file = f"{data_dir}/{num}_galaxies_{split}.hdf5"
        print("Processing", h5_file, "...")
        with h5py.File(h5_file, "r") as f:
            keys  = sorted(list(f[group].keys()))
            p     = f["params"]
            y_all = torch.tensor(p["Omega_m" if target=="om" else "sigma_8"][:], dtype=torch.float32)

            for idx, k in enumerate(keys):
                g   = f[group][k]
                pos = torch.stack([
                    torch.tensor(g["X"][:], dtype=torch.float32),
                    torch.tensor(g["Y"][:], dtype=torch.float32),
                    torch.tensor(g["Z"][:], dtype=torch.float32),
                ], dim=1)                         

                ei = radius_graph(pos, r=R, loop=False)

                sid = int(k.split("_")[-1])
                y   = y_all[sid].unsqueeze(0)        

                x = torch.ones((pos.size(0), 1), dtype=torch.float32)
                data = Data(x=x, pos=pos, y=y, L=L, edge_index=ei)
                torch.save(data, out_dir / f"data_{split}_{idx}.pt")
        print(f"{split} done.")
    print("Preprocessing finished!")
