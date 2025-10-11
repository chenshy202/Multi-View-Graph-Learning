# Multi-View Graph Learning with Graph-Tuple

This repository contains the official PyTorch implementation for our paper, "Multi-View Graph Learning with Graph-Tuple". We introduce a multi-view graph-tuple framework that captures both fine-grained and contextual interactions. Instead of a single graph, our graph-tuple framework partitions the graph into disjoint subgraphs, capturing primary local interactions and weaker, long-range connections. We instantiate our framework on molecular property prediction and cosmological parameter inference. 

## Code Structure
- The `molecule/` directory contains all code and experiments related to the molecular property prediction.
- The `cosmology/` directory contains all code and experiments related to the cosmological parameter inference.
- Dependencies mainly follow [nhuang37/InvariantFeatures](https://github.com/nhuang37/InvariantFeatures).
   


## Datasets

- For molecular property prediction, we use the QM7b dataset provided by PyTorch Geometric, which can be loaded via `from torch_geometric.datasets import QM7b`.
- For molecular property prediction, we use the point-cloud suites from CosmoBench (CAMELS and CAMELS-SAM), downloaded from the official site: [cosmobench](https://cosmobench.streamlit.app/).

## Experiments
- For molecular property prediction: run `python main.py`.
- For cosmological parameter inference: 
  - Build radius graphs from the [CAMELS/CAMELS-SAM HDF5 files](https://users.flatironinstitute.org/~fvillaescusa/CosmoBench/) through `prepossesing.py` and saves them as PyG Data objects.
  - Run `python main.py` and `python main_gt.py`.
