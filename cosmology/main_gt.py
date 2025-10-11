# dataset_camels.py
import os, glob, re, torch
from torch_geometric.data import Dataset, Data
from utils import compute_R2, bootstrap_r2, pbc_delta, rbf_encode
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.nn import global_mean_pool
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from models import EGNN, EGNN_GT
import os, csv, json, time
from datetime import datetime
import pandas as pd
from pathlib import Path

def set_seed(seed: int = 0):
    import random, numpy as np
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _natural_key(s):
    return [int(t) if t.isdigit() else t for t in re.findall(r'\d+|\D+', s)]



@torch.no_grad()
def _band_minus(ei_big: torch.Tensor, ei_small: torch.Tensor, N: int) -> torch.Tensor:
    """
    Return edges in ei_big (≤R2) that are NOT in ei_small (≤R1), treating edges as undirected.
    """
    if ei_big.numel() == 0:
        return ei_big

    device = ei_big.device
    ei_big   = ei_big.long().contiguous()
    ei_small = ei_small.long().to(device).contiguous()

    # undirected ids: (min(i,j), max(i,j)) → i*N + j
    b0 = torch.minimum(ei_big[0],  ei_big[1])
    b1 = torch.maximum(ei_big[0],  ei_big[1])
    if ei_small.numel() > 0:
        s0 = torch.minimum(ei_small[0], ei_small[1])
        s1 = torch.maximum(ei_small[0], ei_small[1])
        small_id = s0 * N + s1
    else:
        return ei_big

    big_id = b0 * N + b1
    keep   = ~torch.isin(big_id, small_id)
    ei_band = ei_big[:, keep]


    return ei_band


class CamelsTupleDataset(Dataset):
    """
    Load two single-threshold graphs (≤R1 and ≤R2) and return complementary pair:
      edge_index1 = ≤R1
      edge_index2 = (R1, R2]  (ei_R2 minus ei_R1)
    """
    def __init__(self, root: str, split: str = "train", R1: str = "2", R2: str = "4", target="om"):
        super().__init__(root)
        dir1 = os.path.join(root, f"{target}/{R1}")
        dir2 = os.path.join(root, f"{target}/{R2}")
        ptn  = f"data_{split}_*.pt"

        self.files1 = sorted(glob.glob(os.path.join(dir1, ptn)), key=_natural_key)
        self.files2 = sorted(glob.glob(os.path.join(dir2, ptn)), key=_natural_key)
        if not self.files1:
            raise FileNotFoundError(f"No files in {dir1}/{ptn}")
        if not self.files2:
            raise FileNotFoundError(f"No files in {dir2}/{ptn}")
        if len(self.files1) != len(self.files2):
            raise RuntimeError("Mismatched sample counts between R1 and R2 folders")

        self.r_inner = float(R1)
        self.r_outer = float(R2)

    def len(self):
        return len(self.files1)

    def get(self, idx):
        d1: Data = torch.load(self.files1[idx], weights_only=False)  # ≤R1
        d2: Data = torch.load(self.files2[idx], weights_only=False)  # ≤R2

        # basic sanity checks (optional)
        assert d1.pos.size(0) == d2.pos.size(0), "Different node counts"
        assert float(d1.L if not torch.is_tensor(d1.L) else d1.L.flatten()[0]) == \
               float(d2.L if not torch.is_tensor(d2.L) else d2.L.flatten()[0]), "Different box size L"

        N   = d1.pos.size(0)
        ei1 = d1.edge_index
        ei2_band = _band_minus(d2.edge_index, ei1, N)

        # optional: verify disjointness
        # with torch.no_grad():
        #     s1 = set(map(tuple, torch.stack([torch.minimum(ei1[0], ei1[1]),
        #                                      torch.maximum(ei1[0], ei1[1])], dim=0).t().tolist()))
        #     s2 = set(map(tuple, torch.stack([torch.minimum(ei2_band[0], ei2_band[1]),
        #                                      torch.maximum(ei2_band[0], ei2_band[1])], dim=0).t().tolist()))
        #     assert len(s1 & s2) == 0, "edge sets still overlap"

        return Data(
            x=d1.x, pos=d1.pos, y=d1.y, L=d1.L,
            edge_index1=ei1,          # 0..R1
            edge_index2=ei2_band      # (R1, R2]
        )


def train_one_epoch(model, loader, device, opt):
    model.train()
    loss_sum, n = 0.0, 0
    y_all, yhat_all = [], []
    for data in loader:
        data = data.to(device, non_blocking=True)
        y    = data.y.view(-1)
        yhat = model(data).view(-1)

        loss = F.mse_loss(yhat, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        loss_sum += loss.item() * data.num_graphs
        n += data.num_graphs
        y_all.append(y.detach().cpu())
        yhat_all.append(yhat.detach().cpu())

    y    = torch.cat(y_all, dim=0)
    yhat = torch.cat(yhat_all, dim=0)
    r2   = compute_R2(yhat, y).item()
    return loss_sum/max(n,1), r2

@torch.no_grad()
def eval_metrics(model, loader, device, return_preds=False):
    model.eval()
    loss_sum, n = 0.0, 0
    y_all, yhat_all = [], []
    for data in loader:
        data = data.to(device, non_blocking=True)
        y    = data.y.view(-1)
        yhat = model(data).view(-1)

        loss = F.mse_loss(yhat, y)
        loss_sum += loss.item() * data.num_graphs
        n += data.num_graphs
        y_all.append(y.detach().cpu())
        yhat_all.append(yhat.detach().cpu())

    y    = torch.cat(y_all, dim=0)
    yhat = torch.cat(yhat_all, dim=0)
    r2   = compute_R2(yhat, y).item()
    if return_preds:
        return loss_sum/max(n,1), r2, yhat, y
    return loss_sum/max(n,1), r2





def main():
    # ---- Base seed; each repeat will derive its own rep_seed ----
    base_seed = 2025
    set_seed(base_seed)

    # ---- Data & meta ----
    target = "om"
    data = "CAMELS" #"CAMELS-SAM"
    data_dir = f"/data/schen355/cosmodata/CosmoBench_{data}"
    grp_key = "BSQ" if "Quijote" in data_dir else "LH"

    # ---- Hyperparameters ----
    bs = 8
    hid_dim = 96
    n_layers = 3
    lr = 5e-4
    wd = 1e-5
    min_lr = 1e-5
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    Rclist = [(0.5, 1), (3.5, 7)]
    for Rcs in Rclist:
        Rc1, Rc2 = Rcs
        Rc1 = round(float(Rc1), 1)
        Rc2 = round(float(Rc2), 1)
        

        # ---- Output dir and files (only one config + one results.csv) ----
        run_name = f"EGNN_GT_{Rc1},{Rc2}"
        run_dir = os.path.join(f"runs/{data}/{target}/GT", run_name)
        os.makedirs(run_dir, exist_ok=True)

        config = {
            "data_dir": data_dir, "group": grp_key, "Rc1": Rc1, "Rc2": Rc2, "batch_size": bs,
            "hid_dim": hid_dim, "n_layers": n_layers, "lr": lr, "weight_decay": wd,
            "seed": base_seed, "device": device
        }
        with open(os.path.join(run_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

        results_csv = os.path.join(run_dir, "results.csv")
        need_header = (not os.path.exists(results_csv)) or (os.path.getsize(results_csv) == 0)
        if need_header:
            with open(results_csv, "w", newline="") as f:
                csv.writer(f).writerow([
                    "best_epoch",
                    "best_val_R2",
                    "test_loss",
                    "test_R2",
                    "bootstrap_R2_mean",
                    "bootstrap_R2_std",
                    "total_sec",
                    "seed"
                ])

        # ---- Build datasets once; reuse across repeats ----
        train_set = CamelsTupleDataset(root=data_dir, split="train", R1=str(Rc1), R2=str(Rc2), target=target)
        val_set   = CamelsTupleDataset(root=data_dir, split="val",   R1=str(Rc1), R2=str(Rc2), target=target)
        test_set  = CamelsTupleDataset(root=data_dir, split="test",  R1=str(Rc1), R2=str(Rc2), target=target)

        # ---- Run 10 repeats ----
        num_repeats = 10
        for rep in range(num_repeats):
            # Deterministic per-repeat seed
            rep_seed = base_seed + (int(Rc1 * 10) + int(Rc2)) * 1000 + rep
            set_seed(rep_seed)

            # Fresh loaders per repeat
            train_loader = DataLoader(train_set, batch_size=bs, shuffle=True,  num_workers=4)
            val_loader   = DataLoader(val_set,   batch_size=bs, shuffle=False, num_workers=4)
            test_loader  = DataLoader(test_set,  batch_size=bs, shuffle=False, num_workers=4)

            # Model & optimizer/scheduler
            node_feat_dim = train_set[0].x.size(1)  # assumes .x exists
            model = EGNN_GT(
                node_feature_dim=node_feat_dim,
                hid_dim=hid_dim, out_dim=1,
                n_layers=n_layers, Rc1=Rc1, Rc2=Rc2, use_pbc=True
            ).to(device)
            opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt, mode='min', factor=0.7, patience=5, min_lr=min_lr
            )

            best_r2, best_state, best_epoch = None, None, -1
            t0 = time.perf_counter()

            # ---- Train loop ----
            for epoch in tqdm(range(1, 301)):
                t_ep = time.perf_counter()
                tr_loss, tr_r2   = train_one_epoch(model, train_loader, device, opt)
                val_loss, val_r2 = eval_metrics(model,  val_loader,   device)
                sched.step(val_loss)

                current_lr = opt.param_groups[0]['lr']
                epoch_sec = time.perf_counter() - t_ep
                print(
                    f"[GT Rc1={Rc1} Rc2={Rc2} rep={rep} seed={rep_seed}] "
                    f"ep {epoch:03d} | tr_loss {tr_loss:.6f} | tr_R2 {tr_r2:.6f} | "
                    f"val_loss {val_loss:.6f} | val_R2 {val_r2:.6f} | lr {current_lr:.2e} | {epoch_sec:.1f}s"
                )

                # Track best by validation R2 (maximize)
                if (best_r2 is None) or (val_r2 > best_r2):
                    best_r2, best_epoch = val_r2, epoch
                    best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

                # Early stop when LR hits floor
                if current_lr <= min_lr + 1e-12:
                    print(f"[INFO] LR hit floor {current_lr:.2e}, early stop.")
                    break

            total_sec = time.perf_counter() - t0
            print(f"[select] best_epoch={best_epoch} | best_val_R2={best_r2:.6f} | total {total_sec/60:.1f} min")

            # ---- Load best and test ----
            if best_state is not None:
                model.load_state_dict(best_state, strict=True)

            test_loss, test_r2, yhat_test, y_test = eval_metrics(
                model, test_loader, device, return_preds=True
            )
            mean_boot, std_boot = bootstrap_r2(yhat_test, y_test, num_bootstrap=1000, seed=42)

            # ---- Append a single summary row for this repeat ----
            with open(results_csv, "a", newline="") as f:
                csv.writer(f).writerow([
                    best_epoch,
                    f"{float(best_r2):.6f}",
                    f"{float(test_loss):.6f}",
                    f"{float(test_r2):.6f}",
                    f"{float(mean_boot):.6f}",
                    f"{float(std_boot):.6f}",
                    f"{float(total_sec):.3f}",
                    rep_seed
                ])

            # ---- Cleanup for this repeat ----
            del model, opt, sched
            del train_loader, val_loader, test_loader
            del y_test, yhat_test
            try:
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
            except Exception:
                pass

        # Datasets cleaned after all repeats
        del train_set, val_set, test_set


if __name__ == "__main__":
    main()


