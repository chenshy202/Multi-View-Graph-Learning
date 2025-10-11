import os.path as osp
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T

from tqdm.auto import tqdm
from sklearn.model_selection import KFold, train_test_split

import argparse
import pickle
from torch_geometric.datasets import QM7b


from GNN import GINE_Model, Gt_GINE_Model
from utils import get_fs, get_binary_expansion, ExtractTarget, get_graph_tuple_QM7b
from time import time


def train_gnn(train_loader, model, optimizer, device):
    model.train()
    loss_all = 0
    criterion = nn.L1Loss()  

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        pred = model(data)
        loss = criterion(pred, data.y)
        loss.backward()
        loss_all += loss.item() * data.num_graphs
        optimizer.step()
    return loss_all / len(train_loader.dataset)

def test_gnn(loader, model, device, per_target=False):
    model.eval()
    criterion = nn.L1Loss(reduction='none')
    out_dim = model.out_dim

    error = 0.0
    if per_target:
        error_pt = torch.zeros(out_dim)

    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            pred = model(data)
        diff = criterion(pred, data.y)  # shape = [batch_size, out_dim]
        if per_target:
            error_pt += diff.sum(dim=0).cpu()
        error += diff.sum().item()

    if per_target:
        return error / len(loader.dataset), error_pt / len(loader.dataset)
    else:
        return error / len(loader.dataset), None


def nn_evaluation_gnn(dataset, node_feature_dim, hid_dim, out_dim, edge_dim, max_num_epochs=200, batch_size=128,
                      start_lr=0.01, min_lr=1e-6, factor=0.5, patience=50,
                      num_repetitions=1, verbose=True, per_target=False):
    torch.manual_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_all = []


    for rep in range(num_repetitions):
        test_error_list = []
        kf = KFold(n_splits=10, shuffle=True, random_state=rep)

        all_indices = list(range(len(dataset)))
        for train_index, test_index in kf.split(all_indices):
            train_index, val_index = train_test_split(train_index, test_size=0.1, random_state=rep)

            train_dataset = dataset[train_index]
            val_dataset = dataset[val_index]
            test_dataset = dataset[test_index]

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

            model = Gt_GINE_Model(node_feature_dim, hid_dim, out_dim, edge_dim).to(device)
            # model = GINE_Model(node_feature_dim, hid_dim, out_dim).to(device)
            
            optimizer = torch.optim.Adam(model.parameters(), lr=start_lr, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=factor, patience=patience, min_lr=min_lr
            )

            best_val_error = None
            best_test_error = None
            n = 0

            
            for epoch in tqdm(range(1, max_num_epochs + 1)):
                # st = time()
                current_lr = scheduler.optimizer.param_groups[0]['lr']
                loss = train_gnn(train_loader, model, optimizer, device)
                val_error, _ = test_gnn(val_loader, model, device, per_target)
                scheduler.step(val_error)
                # et = time()
                # with open("results_QM7b/GIN_weighted/time.txt", "a") as file:
                #     file.write(str(et-st))
                #     file.write("\n")
                if best_val_error is None or val_error < best_val_error:
                    best_val_error = val_error
                    best_test_error, _ = test_gnn(test_loader, model, device, per_target)
                    n = 0
                else:
                    n += 1

                if verbose and epoch % 10 == 0:
                    print(f"Epoch={epoch}, LR={current_lr:.6f}, Loss={loss:.4f}, Val MAE={val_error:.4f}")

                if current_lr <= min_lr:
                    break
            


            test_error_list.append(best_test_error)
            print(f"[Fold] Best test MAE = {best_test_error:.4f}")

        test_all.append(test_error_list)

    test_all = np.array(test_all)
    return test_all


def main(args):
    targets = np.arange(0, 14, 1)
    for target in targets:
        dataset = QM7b(args.path, pre_transform=T.Compose([get_fs, get_binary_expansion]),
               transform=T.Compose([ExtractTarget(target)]))
        dataset.name = "QM7b"
        dataset = get_graph_tuple_QM7b(dataset, args.thresh)
        test_all = nn_evaluation_gnn(
            dataset, node_feature_dim=args.node_feature_dim, hid_dim=args.hid_dim, edge_dim=args.edge_dim, out_dim=1,
            max_num_epochs=1000, batch_size=args.bs,
            start_lr=0.005, min_lr=0.00001, factor=0.8, patience=5,
            num_repetitions=1, verbose=True, per_target=False
        )

        file_name = osp.join(args.result_path, f"gnn_target_{target}.pkl")
        pickle.dump(test_all, open(file_name, "wb"))

        print(f"Target {target}: Cross-val MAEs:", test_all)
        print(f"Saved results to {file_name}")

        results_file = osp.join(args.result_path, "results.txt")
        with open(results_file, "a") as file:
            file.write(f"Target {target}:\n")
            file.write(f"Cross-val MAEs: {test_all}\n")
            file.write("=" * 50 + "\n")  

        test_all_np = test_all.flatten()
        mean_mae = np.mean(test_all_np)
        std_mae = np.std(test_all_np)
        stderr_mae = np.std(test_all_np, ddof=1) / np.sqrt(len(test_all_np))

        stats_file = osp.join(args.result_path, "stats.txt")
        with open(stats_file, "a") as file:
            file.write(f"Target {target}: {mean_mae:.4f}, {std_mae:.4f}, {stderr_mae:.4f}\n")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GNN for QM7b")
    parser.add_argument("--target", type=int, default=0, help="Which target index [0..13]")
    parser.add_argument("--bs", type=int, default=128, help="Batch size")
    parser.add_argument("--hid_dim", type=int, default=100, help="Hidden dimension in GNN")
    parser.add_argument("--path", type=str, default="Data_Molecules/QM7b", help="Dataset folder path")
    parser.add_argument("--result_path", type=str, default="./results_QM7b/Gt_GINE", help="Result folder path")
    parser.add_argument("--node_feature_dim", type=int, default=100, help="node feature dimension in GNN")
    parser.add_argument("--edge_dim", type=int, default=100, help="edge feature dimension in GNN")
    parser.add_argument("--thresh", type=float, default=2, help="thresh for partitioning graph")

    args = parser.parse_args()

    print(args)
    main(args)

    # file_name = osp.join(args.result_path, f"gnn_target_{args.target}.pkl")

    # with open(file_name, "rb") as f:
    #     test_all = pickle.load(f)

    # print("Loaded test_all shape:", np.array(test_all).shape)

    # mean_mae = np.mean(test_all)
    # std_mae = np.std(test_all)

    # print(f"Mean Test MAE: {mean_mae:.4f}")
    # print(f"Standard Deviation of MAE: {std_mae:.4f}")