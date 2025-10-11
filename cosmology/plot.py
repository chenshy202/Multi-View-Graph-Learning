import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl


mpl.rcParams.update({
    "font.size": 12,      
    "axes.labelsize": 26,  
    "axes.titlesize": 18,   
    "xtick.labelsize": 19,   
    "ytick.labelsize": 19,   
    "legend.fontsize": 26,   
    "figure.titlesize": 18,
})

data = "CAMELS" #"CAMELS-SAM"
targetl = ["om", "sigma_8"]
name_map = {'sigma_8': r'$\sigma_8$', 'om': r'$\Omega_{\rm m}$'}

for target in targetl:
    fig, axes = plt.subplots(1, 2, figsize=(11, 7), sharey=True)

    for choice_idx, choice in enumerate([1, 2]):
        ax = axes[choice_idx] 

        CSV_PATH1 = f"runs/{data}/{target}/thres/all_results.csv"  
        
        df = pd.read_csv(CSV_PATH1)
        df["Rc"] = pd.to_numeric(df["Rc"], errors="coerce")
        df["test_R2"] = pd.to_numeric(df["test_R2"], errors="coerce")
        df = df.dropna(subset=["Rc","test_R2"])

        agg = (
            df.groupby("Rc")["test_R2"]
              .agg(mean="mean", std="std", n="count")
              .reset_index()
        )
        agg["sem"] = agg["std"] / np.sqrt(agg["n"].clip(lower=1))
        agg = agg.sort_values("Rc")

        rc_pick1 = np.arange(0.5, 4, 0.5)
        if target == "sigma_8":
          if data == "CAMELS":
            rc_pick1 = np.arange(1.5, 5.1, 0.5)
          else:
            rc_pick1 = np.arange(1, 4.5, 0.5)
        pick_mask1 = agg['Rc'].round(1).isin([round(x,1) for x in rc_pick1])
        sel1 = agg.loc[pick_mask1].sort_values('Rc')

        rc_pick2 = 2 * rc_pick1
        pick_mask2 = agg['Rc'].round(1).isin([round(x,1) for x in rc_pick2])
        sel2 = agg.loc[pick_mask2].sort_values('Rc')

        gt_pick = [f"{round(c1,1)},{round(c2,1)}" for c1, c2 in zip(rc_pick1, rc_pick2)]
        gt_means, gt_stderrs = [], []
        for gt in gt_pick:
            CSV_PATH2 = f"runs/{data}/{target}/GT/EGNN_GT_{gt}/results.csv"
            df_gt = pd.read_csv(CSV_PATH2)
            vals = df_gt['test_R2'].astype(float).to_numpy()
            gt_means.append(vals.mean())
            gt_stderrs.append(vals.std(ddof=1) / np.sqrt(vals.size))

        if choice == 1:
            ax.set_xlabel(r"$\mathbf{c}_{\mathbf{1}}$")
            ax.errorbar(sel1['Rc'].values,
                        sel1['mean'].values,
                        yerr=sel1['sem'].values,
                        fmt='o', capsize=8, linewidth=6, color="#1f77b4",
                        linestyle="-.",
                        markersize=13,     
                        markeredgewidth=1.2,
                        label=r'EGNN-$c_1$')
            x_gt = sel1['Rc'].values
        else: # choice == 2
            ax.set_xlabel(r"$\mathbf{c}_{\mathbf{2}}$")
            ax.errorbar(sel2['Rc'].values,
                        sel2['mean'].values,
                        yerr=sel2['sem'].values,
                        fmt='s', capsize=8, linewidth=6, color="#2ca02c",
                        linestyle="-.",
                        markersize=13,       
                        markeredgewidth=1.2,
                        label=r'EGNN-$c_2$')
            x_gt = sel2['Rc'].values
          
        ax.errorbar(x_gt, gt_means, gt_stderrs, 
                    fmt="^-", capsize=8,
                    markersize=13,       
                    markeredgewidth=1.2,
                    linewidth=6, label=r"EGNN-$(c_1, c_2)$", color="tab:orange")
                    
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()

    task_name = name_map.get(target, str(target))
    # fig.suptitle(f"CAMELS: {task_name}", fontsize=22, y=1.0) 
    axes[0].set_ylabel(r"$\mathbf{R}^\mathbf{2}$")

    plt.tight_layout(rect=[0, 0, 1, 0.95]) 
    plt.savefig(f"figures/{data}/{target}_plots.png", dpi=300, bbox_inches="tight")
    plt.close() 

    print(f"figures/{data}/{target}_plots.png")