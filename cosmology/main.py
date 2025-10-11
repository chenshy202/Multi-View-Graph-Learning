# dataset_camels.py
import os, glob, re, torch
from torch_geometric.data import Dataset
from utils import compute_R2, bootstrap_r2, pbc_delta, rbf_encode
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.nn import global_mean_pool
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import os, csv, json, time
from datetime import datetime
from models import EGNN, EGNN_GT
from pathlib import Path
import pandas as pd


def set_seed(seed: int = 0):
    import random, numpy as np
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _natural_key(s):
    return [int(t) if t.isdigit() else t for t in re.findall(r'\d+|\D+', s)]

class CamelsPointCloud(Dataset):
    def __init__(self, root: str, split: str = "train", R_str: str = "3", target="om"):
        super().__init__(root)
        self.pt_dir = os.path.join(root, f"{target}/{R_str}")
        pattern = os.path.join(self.pt_dir, f"data_{split}_*.pt")
        self.file_paths = sorted(glob.glob(pattern), key=_natural_key)
        if not self.file_paths:
            raise FileNotFoundError(f"No files found: {pattern}")

    def len(self):
        return len(self.file_paths)

    def get(self, idx):
        return torch.load(self.file_paths[idx])  # Data(x=[N,1], pos=[N,3], edge_index, y=[1], L=scalar)


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
    # ---- Global seed (used only for top-level determinism; each run uses rep_seed) ----
    base_seed = 2025
    set_seed(base_seed)

    # ---- Data & group tag ----
    target = "om" #"sigma_8"
    data = "CAMELS" #"CAMELS-SAM"
    data_dir = f"/data/schen355/cosmodata/CosmoBench_{data}"
    grp_key = "BSQ" if "Quijote" in data_dir else "LH"

    # ---- Training hyperparameters ----
    bs = 8
    hid_dim = 96
    n_layers = 3
    lr = 5e-4
    wd = 1e-5
    min_lr = 1e-5
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ---- Radii to sweep & repeats per radius ----
    Rclist = [1]   
    num_repeats = 10

    # ---- Prepare a global CSV to collect all runs across all Rc ----
    global_dir = f"runs/{data}/{target}/thres"
    os.makedirs(global_dir, exist_ok=True)

    all_csv = os.path.join(global_dir, "all_results.csv")
    need_all_header = (not os.path.exists(all_csv)) or (os.path.getsize(all_csv) == 0)
    if need_all_header:
        with open(all_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Rc", "rep", "seed",           # run metadata
                "best_epoch",
                "best_val_R2",
                "test_loss",
                "test_R2",
                "bootstrap_R2_mean",
                "bootstrap_R2_std",
                "total_sec"
            ])

    # ---- Iterate radii ----
    for Rc in Rclist:
        run_name = f"EGNN_Rc{float(Rc):.1f}"
        run_dir = os.path.join(global_dir, run_name)
        os.makedirs(run_dir, exist_ok=True)

        # Record an Rc-level config once (kept minimal on purpose)
        config = {
            "data_dir": data_dir, "group": grp_key, "Rc": f"{float(Rc):.1f}", "batch_size": bs,
            "hid_dim": hid_dim, "n_layers": n_layers, "lr": lr, "weight_decay": wd,
            "seed": base_seed, "device": device, "target": target
        }
        with open(os.path.join(run_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

        # Per-Rc local CSV: only the final summary of each repeat (includes seed)
        results_csv = os.path.join(run_dir, "results.csv")
        need_header = (not os.path.exists(results_csv)) or (os.path.getsize(results_csv) == 0)
        if need_header:
            with open(results_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "best_epoch",
                    "best_val_R2",
                    "test_loss",
                    "test_R2",
                    "bootstrap_R2_mean",
                    "bootstrap_R2_std",
                    "total_sec",
                    "seed"
                ])

        # ---- Build datasets for this Rc once; reuse across repeats ----
        train_set = CamelsPointCloud(root=data_dir, split='train', R_str=f"{float(Rc):.1f}", target=target)
        val_set   = CamelsPointCloud(root=data_dir, split='val',   R_str=f"{float(Rc):.1f}", target=target)
        test_set  = CamelsPointCloud(root=data_dir, split='test',  R_str=f"{float(Rc):.1f}", target=target)

        # ---- Repeat runs with different seeds for this Rc ----
        for rep in range(num_repeats):
            # Derive a deterministic seed for this (Rc, rep)
            rep_seed = base_seed + int(Rc * 1000) + rep
            set_seed(rep_seed)

            # Fresh loaders per repeat (avoid stale worker state)
            train_loader = DataLoader(train_set, batch_size=bs, shuffle=True,  num_workers=4)
            val_loader   = DataLoader(val_set,   batch_size=bs, shuffle=False, num_workers=4)
            test_loader  = DataLoader(test_set,  batch_size=bs, shuffle=False, num_workers=4)

            # Infer node feature dimension from the first sample.
            # WARNING: This assumes `.x` exists; if not, replace by your own logic.
            node_feat_dim = train_set[0].x.size(1)

            # Model & optimizer/scheduler
            model = EGNN(
                node_feature_dim=node_feat_dim,
                hid_dim=hid_dim, out_dim=1, Rc=Rc, n_layers=n_layers, use_pbc=True
            ).to(device)
            opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt, mode='min', factor=0.7, patience=5, min_lr=min_lr
            )

            best_r2, best_state, best_epoch = None, None, -1
            t0 = time.perf_counter()

            # ---- Train loop ----
            for epoch in range(1, 301):
                t_ep = time.perf_counter()

                tr_loss, tr_r2   = train_one_epoch(model, train_loader, device, opt)
                val_loss, val_r2 = eval_metrics(model,  val_loader,   device)
                sched.step(val_loss)

                epoch_sec  = time.perf_counter() - t_ep
                current_lr = opt.param_groups[0]['lr']

                print(
                    f"[Rc={float(Rc):.1f} rep={rep} seed={rep_seed}] epoch {epoch:03d} | "
                    f"train_loss {tr_loss:.6f} | train_R2 {tr_r2:.6f} | "
                    f"val_loss {val_loss:.6f} | val_R2 {val_r2:.6f} | "
                    f"lr {current_lr:.2e} | {epoch_sec:.1f}s"
                )

                # Track the best checkpoint by validation R2 (maximize)
                if (best_r2 is None) or (val_r2 > best_r2):
                    best_r2, best_epoch = val_r2, epoch
                    # Keep on CPU to save GPU memory; load back before testing
                    best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

                # Early stop when LR hits the floor
                if current_lr <= min_lr + 1e-12:
                    print(f"[INFO] LR hit floor {current_lr:.2e}, early stop.")
                    break

            total_sec = time.perf_counter() - t0
            print(
                f"[select Rc={float(Rc):.1f} rep={rep} seed={rep_seed}] epoch={best_epoch} | "
                f"val_R2={best_r2:.6f} | total {total_sec/60:.1f} min"
            )

            # ---- Evaluate the best checkpoint on test ----
            if best_state is not None:
                model.load_state_dict(best_state, strict=True)

            test_loss, test_r2, yhat_test, y_test = eval_metrics(
                model, test_loader, device, return_preds=True
            )
            mean_boot, std_boot = bootstrap_r2(
                yhat_test, y_test, num_bootstrap=1000, seed=42
            )

            # ---- Append to the per-Rc local CSV ----
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

            # ---- Append to the global CSV across all Rc ----
            with open(all_csv, "a", newline="") as f:
                csv.writer(f).writerow([
                    f"{float(Rc):.1f}", int(rep), int(rep_seed),
                    best_epoch,
                    f"{float(best_r2):.6f}",
                    f"{float(test_loss):.6f}",
                    f"{float(test_r2):.6f}",
                    f"{float(mean_boot):.6f}",
                    f"{float(std_boot):.6f}",
                    f"{float(total_sec):.3f}"
                ])

            # ---- Cleanup per repeat ----
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

        # ---- Cleanup per Rc ----
        del train_set, val_set, test_set



if __name__ == "__main__":
    main()


