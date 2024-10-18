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
from unimol2.data.molecule_dataset import get_graph

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


def get_graph_features(edge_attr, edge_index, node_attr, drop_feat):
    # atom_feat_sizes = [128] + [16 for _ in range(8)]
    atom_feat_sizes = [16 for _ in range(8)]
    edge_feat_sizes = [16, 16, 16]
    edge_attr, edge_index, x = edge_attr, edge_index, node_attr
    N = x.shape[0]

    # atom feature here
    atom_feat = convert_to_single_emb(x[:, 1:], atom_feat_sizes)

    # node adj matrix [N, N] bool
    adj = np.zeros([N, N], dtype=np.int32)
    adj[edge_index[0, :], edge_index[1, :]] = 1
    degree = adj.sum(axis=-1)

    # edge feature here
    if len(edge_attr.shape) == 1:
        edge_attr = edge_attr[:, None]
    edge_feat = np.zeros([N, N, edge_attr.shape[-1]], dtype=np.int32)
    edge_feat[edge_index[0, :], edge_index[1, :]] = (
        convert_to_single_emb(edge_attr, edge_feat_sizes) + 1
    )
    shortest_path_result = floyd_warshall(adj)
    # max distance is 509
    if drop_feat:
        atom_feat[...] = 1
        edge_feat[...] = 1
        degree[...] = 1
        shortest_path_result[...] = 511
    else:
        atom_feat = atom_feat + 2
        edge_feat = edge_feat + 2
        degree = degree + 2
        shortest_path_result = shortest_path_result + 1

    # combine, plus 1 for padding
    feat = {}
    feat["atom_feat"] = torch.from_numpy(atom_feat).long()
    feat["atom_mask"] = torch.ones(N).long()
    feat["edge_feat"] = torch.from_numpy(edge_feat).long()
    feat["shortest_path"] = torch.from_numpy((shortest_path_result)).long()
    feat["degree"] = torch.from_numpy(degree).long().view(-1)
    # pair-type
    atoms = feat["atom_feat"][..., 0]
    pair_type = torch.cat(
        [
            atoms.view(-1, 1, 1).expand(-1, N, -1),
            atoms.view(1, -1, 1).expand(N, -1, -1),
        ],
        dim=-1,
    )
    feat["pair_type"] = convert_to_single_emb(pair_type, [128, 128])
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


class Unimol2FeatureDataset(BaseWrapperDataset):
    def __init__(
        self,
        smi_dataset: torch.utils.data.Dataset,
        token_dataset: torch.utils.data.Dataset,
        src_pos_dataset: torch.utils.data.Dataset,
        src_2d_pos_dataset: torch.utils.data.Dataset,
        pad_idx: int,
        mask_idx: int,
        mask_token_prob: float = 0.15,

        mask_pos_prob: float = 1.0,
        noise: float = 1.0,
        noise_type: str = "uniform",
        drop_feat_prob: float = 1.0,
        use_2d_pos: float = 0.5,
        seed: int = 1,
    ):
        super().__init__(smi_dataset)
        self.smi_dataset = smi_dataset
        self.token_dataset = token_dataset
        self.src_pos_dataset = src_pos_dataset
        self.src_2d_pos_dataset = src_2d_pos_dataset
        self.use_2d_pos = use_2d_pos

        self.pad_idx = pad_idx
        self.mask_idx = mask_idx
        self.mask_token_prob = mask_token_prob

        self.noise = noise
        self.noise_type = noise_type
        self.mask_pos_prob = mask_pos_prob
        self.drop_feat_prob = drop_feat_prob

        if self.noise_type == "trunc_normal":
            self.noise_f = lambda num_mask: np.clip(
                np.random.randn(num_mask, 3) * self.noise,
                a_min=-self.noise * 2.0,
                a_max=self.noise * 2.0,
            )
        elif self.noise_type == "normal":
            self.noise_f = lambda num_mask: np.random.randn(num_mask, 3) * self.noise
        elif self.noise_type == "uniform":
            self.noise_f = lambda num_mask: np.random.uniform(
                low=-self.noise, high=self.noise, size=(num_mask, 3)
            )
        else:
            self.noise_f = lambda num_mask: 0.0

        self.seed = seed
        self.set_epoch(None)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch


    def __getitem__(self, idx: int):
        return self.__getitem_cached__(self.epoch, idx)

    def get_masked_token(self, src_token, mask_token_prob):
        sz = len(src_token)
        # don't allow empty sequence
        assert sz > 0

        num_mask_token = int(
            # add a random number for probabilistic rounding
            mask_token_prob * sz
            + np.random.rand()
        )
        mask_idc = np.random.choice(sz, num_mask_token, replace=False)
        mask_token = np.full(sz, False)
        mask_token[mask_idc] = True
        target_token = np.full(len(mask_token), self.pad_idx)
        target_token[mask_token] = src_token[mask_token]

        new_item = np.copy(src_token)
        new_item[mask_token] = self.mask_idx

        return new_item, target_token

    def get_noised_coord(self, coord, mask_cord_prob):
        sz = coord.shape[0]
        # decide elements to mask
        num_mask = int(
            # add a random number for probabilistic rounding
            mask_cord_prob * sz
            + np.random.rand()
        )
        mask_idc = np.random.choice(sz, num_mask, replace=False)
        mask = np.full(sz, False)
        mask[mask_idc] = True

        new_coord = np.copy(coord)
        new_coord[mask, :] += self.noise_f(num_mask)
        return new_coord, mask

    def align_dataset(self, src_pos, tgt_pos):
        R, T = get_optimal_transform(src_pos, tgt_pos)
        aligned_pos = src_pos @ R + T
        return aligned_pos

    def get_molecule_feat(self, smiles, drop_feat_prob, epoch, idx):
        data = {}
        mol = Chem.MolFromSmiles(smiles)

        # remove atom
        mol = AllChem.AddHs(mol, addCoords=True)
        atoms_h = np.array([atom.GetSymbol() for atom in mol.GetAtoms()])
        atoms = np.array([atom.GetSymbol() for atom in mol.GetAtoms() if atom.GetSymbol() != 'H'])

        # change AllChem.RemoveHs to AllChem.RemoveAllHs
        mol = AllChem.RemoveAllHs(mol)
        x, edge_index, edge_attr = get_graph(mol)

        data['node_attr']  = x
        data['edge_index'] = edge_index
        data['edge_attr']  = edge_attr
        data['atoms_h_token'] = atoms_h
        data['atoms_token'] = atoms

        with data_utils.numpy_seed(self.seed, epoch, idx):
            # please note that the context manager
            data['drop_feat'] = np.random.rand() < drop_feat_prob
        return data

    @lru_cache(maxsize=16)
    def __getitem_cached__(self, epoch: int, idx: int):
        ret = {}
        with data_utils.numpy_seed(self.seed, epoch, idx):
            src_token = self.token_dataset[idx]
            src_token_mask, target_token = self.get_masked_token(src_token, self.mask_token_prob)
            ret['src_token'] = torch.from_numpy(src_token_mask).long()
            ret['target_token'] = torch.from_numpy(target_token).long()

            molecule_feat = self.get_molecule_feat(self.smi_dataset[idx], self.drop_feat_prob, epoch, idx)

            feat = get_graph_features(
                molecule_feat['edge_attr'],
                molecule_feat['edge_index'],
                molecule_feat['node_attr'],
                molecule_feat['drop_feat']
            )

            if not molecule_feat['drop_feat']:
                if np.random.rand() < self.use_2d_pos:
                    src_pos = self.src_2d_pos_dataset[idx]
                    tgt_pos = src_pos
                else:
                    src_pos = self.src_pos_dataset[idx]
                    tgt_pos = src_pos
            else:
                src_pos = self.src_pos_dataset[idx]
                tgt_pos = src_pos

            masked_pos, mask_coord_index = self.get_noised_coord(src_pos, self.mask_pos_prob)
            masked_pos = self.align_dataset(masked_pos, tgt_pos)

            # from noised pos to origin pos
            ret['src_pos'] = torch.from_numpy(masked_pos).float()
            ret['target_pos'] = torch.from_numpy(tgt_pos).float()
            ret['src_mask_cord'] = torch.from_numpy(mask_coord_index).bool()

            ret.update(feat)
            return ret

    def collater(self, items):
        pad_fns = {
            "src_token": pad_1d,
            "target_token": pad_1d,
            "src_pos": pad_1d_feat,
            "target_pos": pad_1d_feat,
            "src_mask_cord": pad_1d,

            "atom_feat": pad_1d_feat,
            "atom_mask": pad_1d,
            "edge_feat": pad_2d_feat,
            "shortest_path": pad_2d,
            "degree": pad_1d,
            "pair_type": pad_2d_feat,
            "attn_bias": pad_attn_bias,
        }
        max_node_num = max([item["atom_mask"].shape[0] for item in items])
        max_node_num = (max_node_num + 1 + 3) // 4 * 4 - 1
        batched_data = {}
        for key in items[0].keys():
            samples = [item[key] for item in items]
            if key in pad_fns:
                batched_data[key] = pad_fns[key](samples, max_node_num)
        return batched_data


@njit
def floyd_warshall(M):
    (nrows, ncols) = M.shape
    assert nrows == ncols
    n = nrows
    # set unreachable nodes distance to 509
    for i in range(n):
        for j in range(n):
            if M[i, j] == 0:
                M[i, j] = 509

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
            if M[i, j] >= 509:
                M[i, j] = 509
    return M


def convert_to_single_emb(x, sizes):
    assert x.shape[-1] == len(sizes)
    offset = 1
    for i in range(len(sizes)):
        assert (x[..., i] < sizes[i]).all()
        x[..., i] = x[..., i] + offset
        offset += sizes[i]
    return x


def pad_1d(samples, pad_len, pad_value=0):
    batch_size = len(samples)
    tensor = torch.full([batch_size, pad_len], pad_value, dtype=samples[0].dtype)
    for i in range(batch_size):
        tensor[i, : samples[i].shape[0]] = samples[i]
    return tensor


def pad_1d_feat(samples, pad_len, pad_value=0):
    batch_size = len(samples)
    assert len(samples[0].shape) == 2
    feat_size = samples[0].shape[-1]
    tensor = torch.full(
        [batch_size, pad_len, feat_size], pad_value, dtype=samples[0].dtype
    )
    for i in range(batch_size):
        tensor[i, : samples[i].shape[0]] = samples[i]
    return tensor


def pad_2d(samples, pad_len, pad_value=0):
    batch_size = len(samples)
    tensor = torch.full(
        [batch_size, pad_len, pad_len], pad_value, dtype=samples[0].dtype
    )
    for i in range(batch_size):
        tensor[i, : samples[i].shape[0], : samples[i].shape[1]] = samples[i]
    return tensor


def pad_2d_feat(samples, pad_len, pad_value=0):
    batch_size = len(samples)
    assert len(samples[0].shape) == 3
    feat_size = samples[0].shape[-1]
    tensor = torch.full(
        [batch_size, pad_len, pad_len, feat_size], pad_value, dtype=samples[0].dtype
    )
    for i in range(batch_size):
        tensor[i, : samples[i].shape[0], : samples[i].shape[1]] = samples[i]
    return tensor


def pad_attn_bias(samples, pad_len):
    batch_size = len(samples)
    pad_len = pad_len + 1
    tensor = torch.full(
        [batch_size, pad_len, pad_len], float("-inf"), dtype=samples[0].dtype
    )
    for i in range(batch_size):
        tensor[i, : samples[i].shape[0], : samples[i].shape[1]] = samples[i]
        tensor[i, samples[i].shape[0] :, : samples[i].shape[1]] = 0
    return tensor


class Unimol2FinetuneFeatureDataset(BaseWrapperDataset):
    def __init__(
        self,
        smi_dataset: torch.utils.data.Dataset,
        token_dataset: torch.utils.data.Dataset,
        src_pos_dataset: torch.utils.data.Dataset,
        molecule_dataset: torch.utils.data.Dataset,
        seed: int = 1,
    ):
        super().__init__(smi_dataset)
        self.smi_dataset = smi_dataset
        self.token_dataset = token_dataset
        self.src_pos_dataset = src_pos_dataset
        self.molecule_dataset = molecule_dataset

        self.seed = seed
        self.set_epoch(None)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch
        self.smi_dataset.set_epoch(epoch)
        self.token_dataset.set_epoch(epoch)
        self.src_pos_dataset.set_epoch(epoch)
        self.molecule_dataset.set_epoch(epoch)

    def __getitem__(self, idx: int):
        return self.__getitem_cached__(self.epoch, idx)

    @lru_cache(maxsize=16)
    def __getitem_cached__(self, epoch: int, idx: int):
        ret = {}
        with data_utils.numpy_seed(self.seed, epoch, idx):
            src_token = self.token_dataset[idx]
            ret['src_token'] = torch.from_numpy(src_token).long()

            src_pos = self.src_pos_dataset[idx]
            # from noised pos to origin pos
            ret['src_pos'] = torch.from_numpy(src_pos).float()
            molecule_feat = self.molecule_dataset[idx]

            feat = get_graph_features(
                molecule_feat['edge_attr'],
                molecule_feat['edge_index'],
                molecule_feat['node_attr'],
                molecule_feat['drop_feat']
            )

            ret.update(feat)
            return ret

    def collater(self, items):
        pad_fns = {
            "src_token": pad_1d,
            "src_pos": pad_1d_feat,

            "atom_feat": pad_1d_feat,
            "atom_mask": pad_1d,
            "edge_feat": pad_2d_feat,
            "shortest_path": pad_2d,
            "degree": pad_1d,
            "pair_type": pad_2d_feat,
            "attn_bias": pad_attn_bias,
        }
        max_node_num = max([item["atom_mask"].shape[0] for item in items])
        max_node_num = (max_node_num + 1 + 3) // 4 * 4 - 1
        batched_data = {}
        for key in items[0].keys():
            samples = [item[key] for item in items]
            if key in pad_fns:
                batched_data[key] = pad_fns[key](samples, max_node_num)
        return batched_data
