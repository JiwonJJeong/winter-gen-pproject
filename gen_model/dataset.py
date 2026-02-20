# Backward-compat shim — real code lives in gen_model/data/dataset.py
from gen_model.data.dataset import MDGenDataset, ConditionalMDGenDataset

__all__ = ['MDGenDataset', 'ConditionalMDGenDataset']
