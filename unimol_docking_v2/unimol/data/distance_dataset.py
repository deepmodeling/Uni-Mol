# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from scipy.spatial import distance_matrix
from functools import lru_cache
from unicore.data import BaseWrapperDataset


class DistanceDataset(BaseWrapperDataset):
    def __init__(self, dataset):
        super().__init__(dataset)
        self.dataset = dataset

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        pos = self.dataset[idx].view(-1, 3).numpy()
        dist = distance_matrix(pos, pos).astype(np.float32)
        return torch.from_numpy(dist)


class EdgeTypeDataset(BaseWrapperDataset):
    def __init__(self, dataset: torch.utils.data.Dataset, num_types: int):
        self.dataset = dataset
        self.num_types = num_types

    @lru_cache(maxsize=16)
    def __getitem__(self, index: int):
        node_input = self.dataset[index].clone()
        offset = node_input.view(-1, 1) * self.num_types + node_input.view(1, -1)
        return offset


class CrossDistanceDataset(BaseWrapperDataset):
    def __init__(self, mol_dataset, pocket_dataset):
        super().__init__(mol_dataset)
        self.mol_dataset = mol_dataset
        self.pocket_dataset = pocket_dataset

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        mol_pos = self.mol_dataset[idx].view(-1, 3).numpy()
        pocket_pos = self.pocket_dataset[idx].view(-1, 3).numpy()
        dist = distance_matrix(mol_pos, pocket_pos).astype(np.float32)
        return torch.from_numpy(dist)
