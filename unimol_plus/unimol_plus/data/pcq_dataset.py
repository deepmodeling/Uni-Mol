# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from unicore.data import BaseWrapperDataset
from . import data_utils
from numba import njit
from functools import lru_cache
from scipy.spatial.transform import Rotation


def get_graph_features(item):
    atom_feat_sizes = [128] + [16 for _ in range(8)]
    edge_feat_sizes = [16, 16, 16]
    edge_attr, edge_index, x = (
        item["edge_attr"],
        item["edge_index"],
        item["node_attr"],
    )
    N = x.shape[0]
    atom_feat = data_utils.convert_to_single_emb(x, atom_feat_sizes)

    # node adj matrix [N, N] bool
    adj = np.zeros([N, N], dtype=np.int32)
    adj[edge_index[0, :], edge_index[1, :]] = 1
    degree = adj.sum(axis=-1)

    # edge feature here
    if len(edge_attr.shape) == 1:
        edge_attr = edge_attr[:, None]
    edge_feat = np.zeros([N, N, edge_attr.shape[-1]], dtype=np.int32)
    edge_feat[edge_index[0, :], edge_index[1, :]] = (
        data_utils.convert_to_single_emb(edge_attr, edge_feat_sizes) + 1
    )
    shortest_path_result = floyd_warshall(adj)

    spatial_pos = torch.from_numpy((shortest_path_result)).long()  # plus 1 for padding

    # combine
    feat = {}
    feat["atom_feat"] = torch.from_numpy(atom_feat).long()
    feat["atom_mask"] = torch.ones(N).long()
    feat["edge_feat"] = torch.from_numpy(edge_feat).long() + 1
    feat["shortest_path"] = spatial_pos + 1
    feat["degree"] = torch.from_numpy(degree).long().view(-1) + 1
    # pair-type
    atoms = feat["atom_feat"][..., 0]
    pair_type = torch.cat(
        [
            atoms.view(-1, 1, 1).expand(-1, N, -1),
            atoms.view(1, -1, 1).expand(N, -1, -1),
        ],
        dim=-1,
    )
    feat["pair_type"] = data_utils.convert_to_single_emb(pair_type, [128, 128])
    feat["attn_bias"] = torch.zeros((N + 1, N + 1), dtype=torch.float32)
    return feat


def kabsch_rotation(P, Q):
    C = P.transpose(-1, -2) @ Q
    V, _, W = np.linalg.svd(C)
    d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0
    if d:
        V[:, -1] = -V[:, -1]
    U = V @ W
    return U


def get_optimal_transform(src_atoms, tgt_atoms):
    src_center = src_atoms.mean(-2)[None, :]
    tgt_center = tgt_atoms.mean(-2)[None, :]
    r = kabsch_rotation(src_atoms - src_center, tgt_atoms - tgt_center)
    x = tgt_center - src_center @ r
    return r, x


class PCQDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset,
        src_pos_dataset,
        tgt_pos_dataset,
        is_train,
        label_prob,
        mid_prob,
        mid_lower,
        mid_upper,
        noise,
        seed,
    ):
        super().__init__(dataset)
        self.dataset = dataset
        self.src_pos_dataset = src_pos_dataset
        self.tgt_pos_dataset = tgt_pos_dataset
        self.seed = seed
        self.is_train = is_train
        self.label_prob = label_prob
        self.mid_prob = mid_prob
        self.mid_lower = mid_lower
        self.mid_upper = mid_upper
        self.noise = noise

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    def __getitem__(self, idx: int):
        return self.__getitem_cached__(self.epoch, idx)

    @lru_cache(maxsize=16)
    def __getitem_cached__(self, epoch: int, idx: int):
        with data_utils.numpy_seed(self.seed, epoch, idx):
            data = self.dataset[idx]
            feat = get_graph_features(data)
            pos = self.src_pos_dataset[idx]
            pos_target = (
                self.tgt_pos_dataset[idx] if self.tgt_pos_dataset is not None else pos
            )
            if self.is_train:
                random_rotate = Rotation.random().as_matrix()
                pos_target = pos_target @ random_rotate
            pos_target = torch.from_numpy(pos_target).float()

            use_label = False
            use_mid = False
            if self.is_train:
                p = np.random.rand()
                if p < self.label_prob:
                    use_label = True
                elif p < self.label_prob + self.mid_prob:
                    use_mid = True
            if use_label:
                feat["pos"] = (
                    pos_target
                    + self.noise
                    * torch.from_numpy(np.random.randn(*pos_target.shape)).float()
                )
            elif use_mid:
                q = np.random.uniform(self.mid_lower, self.mid_upper)
                pos = torch.from_numpy(pos).float()
                R, T = get_optimal_transform(pos, pos_target)
                pos = pos @ R + T
                feat["pos"] = q * pos + (1 - q) * (
                    pos_target
                    + self.noise
                    * torch.from_numpy(np.random.randn(*pos_target.shape)).float()
                )
            else:
                feat["pos"] = torch.from_numpy(pos).float()

            def zero_center(pos):
                return pos - pos.mean(0, keepdim=True)

            feat["pos"] = zero_center(feat["pos"])
            feat["pos_target"] = zero_center(pos_target)

            R, T = get_optimal_transform(feat["pos"], feat["pos_target"])
            feat["pos"] = feat["pos"] @ R + T

            feat["target"] = data["target"] if data["target"] is not None else 0.0
            feat["id"] = data["id"]
            return feat

    def collater(self, items):
        target = np.stack([x["target"] for x in items])
        id = np.stack([int(x["id"]) for x in items]).astype(np.int64)
        pad_fns = {
            "atom_feat": data_utils.pad_1d_feat,
            "atom_mask": data_utils.pad_1d,
            "edge_feat": data_utils.pad_2d_feat,
            "shortest_path": data_utils.pad_2d,
            "degree": data_utils.pad_1d,
            "pos": data_utils.pad_1d_feat,
            "pos_target": data_utils.pad_1d_feat,
            "pair_type": data_utils.pad_2d_feat,
            "attn_bias": data_utils.pad_attn_bias,
        }
        max_node_num = max([item["atom_mask"].shape[0] for item in items])
        max_node_num = (max_node_num + 1 + 3) // 4 * 4 - 1
        batched_data = {}
        for key in items[0].keys():
            samples = [item[key] for item in items]
            if key in pad_fns:
                batched_data[key] = pad_fns[key](samples, max_node_num)
        batched_data["target"] = torch.from_numpy(target).float()
        batched_data["id"] = torch.from_numpy(id).long()
        return batched_data


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
