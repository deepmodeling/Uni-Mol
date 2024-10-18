# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from functools import lru_cache
from unicore.data import BaseWrapperDataset, data_utils
from numba import njit


@njit
def floyd_warshall(M):
    (nrows, ncols) = M.shape
    assert nrows == ncols
    n = nrows
    # set unreachable nodes distance to 510
    for i in range(n):
        for j in range(n):
            if M[i, j] == 0:
                M[i, j] = 510

    for i in range(n):
        M[i, i] = 0

    # floyed algo
    for k in range(n):
        for i in range(n):
            for j in range(n):
                cost_ikkj = M[i, k] + M[k, j]
                if M[i, j] > cost_ikkj:
                    M[i, j] = cost_ikkj

    for i in range(n):
        for j in range(n):
            if M[i, j] >= 510:
                M[i, j] = 510
    return M


class ShortestPathDataset(BaseWrapperDataset):
    def __init__(self, dataset, has_bos=True, has_eos=True):
        super().__init__(dataset)
        self.dataset = dataset
        self.has_bos = has_bos
        self.has_eos = has_eos

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        data = self.dataset[idx]
        num_atoms = data["atoms"].shape[0]
        offset = 0
        if self.has_bos:
            num_atoms += 1
            offset = 1
        if self.has_eos:
            num_atoms += 1
        adj = np.full(
            (num_atoms, num_atoms),
            510,
            dtype=np.int,
        )
        edge_index = data["edge_index"]
        adj[edge_index[0, :] + offset, edge_index[1, :] + offset] = 1
        shortest_path_result = floyd_warshall(adj)
        shortest_path_result[shortest_path_result > 510] = 510
        spatial_pos = torch.from_numpy((shortest_path_result)).long()
        return spatial_pos


class DegreeDataset(BaseWrapperDataset):
    def __init__(self, dataset, has_bos=True, has_eos=True):
        super().__init__(dataset)
        self.dataset = dataset
        self.has_bos = has_bos
        self.has_eos = has_eos

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        data = self.dataset[idx]
        num_atoms = data["atoms"].shape[0]
        offset = 0
        if self.has_bos:
            num_atoms += 1
            offset = 1
        if self.has_eos:
            num_atoms += 1
        adj = np.full(
            (num_atoms, num_atoms),
            0,
            dtype=np.int,
        )
        edge_index = data["edge_index"]
        adj[edge_index[0, :] + offset, edge_index[1, :] + offset] = 1
        # +1 for padding
        degree = np.sum(adj, axis=1) + 1
        return torch.from_numpy(degree).long()


def collate_1d_features(
    values,
    pad_idx,
    pad_to_multiple=1,
):
    """Convert a list of 1d tensors into a padded 2d tensor."""
    v = values[0]
    size = max(v.size(0) for v in values)
    if pad_to_multiple != 1 and size % pad_to_multiple != 0:
        size = int(((size - 0.1) // pad_to_multiple + 1) * pad_to_multiple)
    res = values[0].new(len(values), size, v.shape[-1]).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(
            v,
            res[i][: len(v), :],
        )
    return res


class AtomFeatDataset(BaseWrapperDataset):
    def __init__(
        self, dataset, num_features=8, num_vals=16, has_bos=True, has_eos=True,
        remove_hydrogen=False,
        remove_polar_hydrogen=False,
    ):
        super().__init__(dataset)
        self.dataset = dataset
        self.num_features = num_features
        self.num_vals = num_vals
        self.has_bos = has_bos
        self.has_eos = has_eos
        self.remove_hydrogen = remove_hydrogen
        self.remove_polar_hydrogen = remove_polar_hydrogen

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        data = self.dataset[idx]
        if self.remove_hydrogen:
            num_atoms = len(data["atoms_token"])
        else:
            num_atoms = len(data["atoms_h_token"])

        # if the atoms with h is different, then check atoms without h.
        def check(orig_atoms, dst_atoms):
            mask_hydrogen = orig_atoms != "H"
            orig_atoms = orig_atoms[mask_hydrogen]
            assert (orig_atoms == dst_atoms).all()

        if (data['atoms'].shape != data['atoms_h_token'].shape) or (not (data['atoms'] == data['atoms_h_token']).all()):
            print ("exception data...", data['smi'], data['atoms'], data['atoms_h_token'], data["atoms_token"])
            check(data['atoms'], data["atoms_token"])

        offset = 0
        if self.has_bos:
            num_atoms += 1
            offset = 1
        if self.has_eos:
            num_atoms += 1
        # 1 for pad token feat
        # 2 for drop feat
        feat = np.full(
            (num_atoms, self.num_features),
            1,
            dtype=np.int,
        )
        if not data['drop_feat']:
            node_attr = data["node_attr"]
            # skip first dimension
            feat[offset : offset + node_attr.shape[0], :] = node_attr[:, 1:] + 3
        else:
            node_attr = data["node_attr"]
            feat[offset : offset + node_attr.shape[0], :] = 2

        # dim, idx encoding...
        for i in range(self.num_features):
            feat[:, i] += i * self.num_vals
        return torch.from_numpy(feat).long()

    def collater(self, samples):
        return collate_1d_features(samples, 0, pad_to_multiple=8)


def collate_2d_features(
    values,
    pad_idx,
    pad_to_multiple=1,
):
    """Convert a list of 1d tensors into a padded 2d tensor."""
    v = values[0]
    size = max(v.size(0) for v in values)
    if pad_to_multiple != 1 and size % pad_to_multiple != 0:
        size = int(((size - 0.1) // pad_to_multiple + 1) * pad_to_multiple)
    res = values[0].new(len(values), size, size, v.shape[-1]).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(
            v,
            res[i][: len(v), : len(v), :],
        )
    return res


class BondDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset,
        num_features=4,
        num_vals=8,
        has_bos=True,
        has_eos=True,
        remove_hydrogen=False,
        remove_polar_hydrogen=False,
    ):
        super().__init__(dataset)
        self.dataset = dataset
        self.num_features = num_features
        self.num_vals = num_vals
        self.has_bos = has_bos
        self.has_eos = has_eos
        self.remove_hydrogen = remove_hydrogen
        self.remove_polar_hydrogen = remove_polar_hydrogen

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        data = self.dataset[idx]
        if self.remove_hydrogen:
            num_atoms = len(data["atoms_token"])
        else:
            num_atoms = len(data["atoms_h_token"])
        offset = 0
        if self.has_bos:
            num_atoms += 1
            offset = 1
        if self.has_eos:
            num_atoms += 1
        edge_feat = np.full(
            (num_atoms, num_atoms, self.num_features),
            1,
            dtype=np.int,
        )
        if not data['drop_feat']:
            edge_index = data["edge_index"]
            edge_attr = data["edge_attr"]
            # no connected
            edge_feat[:, :, 0] = 3
            # self connected
            for i in range(num_atoms):
                edge_feat[i, i, 0] = 4
            # bond connected
            edge_feat[edge_index[0, :] + offset, edge_index[1, :] + offset, 0] = 5
            # other bond features, 1+3 = 4
            edge_feat[edge_index[0, :] + offset, edge_index[1, :] + offset, 1:] = (
                edge_attr + 3
            )
        else:
            edge_index = data["edge_index"]
            edge_attr = data["edge_attr"]
            edge_feat[edge_index[0, :] + offset, edge_index[1, :] + offset, :] = 2

        for i in range(self.num_features):
            # add offset
            edge_feat[:, :, i] += self.num_vals * i
        return torch.from_numpy(edge_feat).long()

    def collater(self, samples):
        return collate_2d_features(samples, 0, pad_to_multiple=8)


def convert_to_single_emb(x, sizes):
    assert x.shape[-1] == len(sizes)
    offset = 1
    for i in range(len(sizes)):
        assert (x[..., i] < sizes[i]).all()
        x[..., i] = x[..., i] + offset
        offset += sizes[i]
    return x


class PairTypeDataset(BaseWrapperDataset):
    def __init__(self, dataset: torch.utils.data.Dataset, num_types: int):
        self.dataset = dataset
        self.num_types = num_types

    @lru_cache(maxsize=16)
    def __getitem__(self, index: int):
        node_input = self.dataset[index].clone()
        N = len(node_input)
        pair_type = torch.cat(
            [
                node_input.view(-1, 1, 1).expand(-1, N, -1),
                node_input.view(1, -1, 1).expand(N, -1, -1),
            ],
            dim=-1,
        )
        pair_type = convert_to_single_emb(pair_type, [128, 128])
        return pair_type

    def collater(self, samples):
        return collate_2d_features(samples, 0, pad_to_multiple=8)
