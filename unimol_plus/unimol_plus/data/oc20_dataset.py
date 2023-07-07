# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import lmdb
import os
import pickle
import torch
import numpy as np
from functools import lru_cache
from unicore.data import BaseWrapperDataset
from . import data_utils
import torch.nn as nn
from copy import deepcopy


class Is2reDataset(BaseWrapperDataset):
    def __init__(self, dataset, args, is_train=False):
        super().__init__(dataset)
        self.dataset = dataset
        self.is_train = is_train
        self.args = args
        self.cell_offsets = torch.tensor(
            [
                [-1, -1, 0],
                [-1, 0, 0],
                [-1, 1, 0],
                [0, -1, 0],
                [0, 1, 0],
                [1, -1, 0],
                [1, 0, 0],
                [1, 1, 0],
            ],
        ).float()
        self.n_cells = self.cell_offsets.size(0)
        self.cutoff = self.args.cutoff
        self.set_epoch(None)

    def __len__(self):
        return len(self.dataset)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    def __getitem__(self, idx: int):
        return self.__getitem_cached__(self.epoch, idx)

    def _generate_noise_pos_input(self, pos, pos_relaxed, tags):
        use_label = False
        use_mid = False
        if self.is_train:
            p = np.random.rand()
            if p < self.args.label_prob:
                use_label = True
            elif p < self.args.label_prob + self.args.mid_prob:
                use_mid = True
        if use_label:
            pos_t = (
                pos_relaxed
                + self.args.noise_scale
                * torch.from_numpy(np.random.randn(*pos_relaxed.shape)).float()
            )
        elif use_mid:
            q = np.random.uniform(self.args.mid_lower, self.args.mid_upper)
            pos_t = q * pos + (1 - q) * (
                pos_relaxed
                + self.args.noise_scale
                * torch.from_numpy(np.random.randn(*pos_relaxed.shape)).float()
            )
        else:
            pos_t = pos
        update_index = tags > 0  # tag==0 don't move in 3D space
        pos[update_index] = pos_t[update_index]
        return pos

    def _crop_by_distance(self, pos, tags, min_dist, expand_used_mask):
        incell_used_mask = tags >= 1
        keep_num = incell_used_mask.long().sum()
        sample_size = self.args.atom_crop_size - keep_num
        # only sample in the cutoff range
        dist_pre = (pos[tags > 1].unsqueeze(1) - pos[tags == 0].unsqueeze(0)).norm(
            dim=-1
        )
        min_dist_pre = dist_pre.min(dim=0)[0]
        pre_cnt = min_dist_pre.size(0)
        min_dist[~expand_used_mask] = float("inf")
        min_dist = torch.cat(
            [
                min_dist_pre,
                min_dist,
            ],
            dim=0,
        )
        used_mask = torch.zeros_like(min_dist).bool()
        if sample_size > 0:
            sample_prob = torch.nn.functional.softmax(min_dist * -1, dim=-1)
            sampled_index = torch.multinomial(
                sample_prob, sample_size, replacement=False
            )
            used_mask[sampled_index] = True
            incell_used_mask[tags == 0] = used_mask[:pre_cnt]
            return incell_used_mask, used_mask[pre_cnt:]
        else:
            return incell_used_mask, used_mask[pre_cnt:]

    @lru_cache(maxsize=16)
    def __getitem_cached__(self, epoch: int, idx: int):
        with data_utils.numpy_seed(self.args.seed, epoch, idx):
            data = deepcopy(self.dataset[idx])
            cell = torch.from_numpy(data["cell"]).view(3, 3).float()
            pos = torch.from_numpy(data["pos"]).view(-1, 3).float()
            if "pos_relaxed" in data:
                pos_relaxed = torch.from_numpy(data["pos_relaxed"]).view(-1, 3).float()
            else:
                pos_relaxed = pos
            atoms = torch.from_numpy(data["atomic_numbers"]).view(-1).long()
            tags = torch.from_numpy(data["tags"]).long()

            # TODO: add noise after expanding
            pos = (
                self._generate_noise_pos_input(pos, pos_relaxed, tags)
                if self.is_train
                else pos
            )

            expand_offsets = torch.matmul(self.cell_offsets, cell).view(
                self.n_cells, 1, 3
            )

            def get_expand_pos(pos):
                expand_pos = pos.unsqueeze(0).expand(self.n_cells, -1, -1)
                return (expand_pos + expand_offsets).view(-1, 3)

            expand_pos = get_expand_pos(pos)
            expand_pos_relaxed = get_expand_pos(pos_relaxed)
            expand_atoms = atoms.repeat(self.n_cells)
            expand_tags = tags.repeat(self.n_cells)

            # only use atoms with tag > 1 for the radius cutoff
            src_pos = pos[tags > 1]
            dist = (src_pos.unsqueeze(1) - expand_pos.unsqueeze(0)).norm(dim=-1)
            min_dist = dist.min(dim=0)[0]
            expand_used_mask = min_dist < self.cutoff
            num_atom = pos.shape[0]
            expand_num_atom = expand_used_mask.long().sum().item()
            if self.is_train and num_atom + expand_num_atom > self.args.atom_crop_size:
                incell_used_mask, expand_used_mask = self._crop_by_distance(
                    pos, tags, min_dist, expand_used_mask
                )
                pos = pos[incell_used_mask]
                pos_relaxed = pos_relaxed[incell_used_mask]
                atoms = atoms[incell_used_mask]
                tags = tags[incell_used_mask]

            expand_pos = expand_pos[expand_used_mask]
            expand_pos_relaxed = expand_pos_relaxed[expand_used_mask]
            expand_atoms = expand_atoms[expand_used_mask]
            expand_tags = expand_tags[expand_used_mask]

            data["tags"] = torch.cat([tags, expand_tags])
            data["main_cell"] = torch.cat(
                [torch.ones_like(tags).long(), torch.zeros_like(expand_tags).long()]
            )
            data["atomic_numbers"] = torch.cat([atoms, expand_atoms.long()])
            data["pos"] = torch.cat([pos, expand_pos], dim=0)
            data["pos_target"] = torch.cat(
                [
                    pos_relaxed,
                    expand_pos_relaxed,
                ],
                dim=0,
            )
            num_atom = data["atomic_numbers"].shape[0]
            pair_type = torch.cat(
                [
                    data["atomic_numbers"].view(-1, 1, 1).expand(-1, num_atom, -1),
                    data["atomic_numbers"].view(1, -1, 1).expand(num_atom, -1, -1),
                ],
                dim=-1,
            )
            data["pair_type"] = data_utils.convert_to_single_emb(pair_type, [128, 128])
            data["atom_mask"] = torch.ones_like(data["atomic_numbers"]).float()
            data["attn_bias"] = torch.zeros(
                (num_atom + 1, num_atom + 1), dtype=torch.float32
            )
            data["id"] = int(data["sid"])
            if "y_relaxed" not in data:
                data["y_relaxed"] = 0.0
            return data

    def collater(self, items):
        target = np.stack([x["y_relaxed"] for x in items])
        id = np.stack([int(x["id"]) for x in items]).astype(np.int64)
        pad_fns = {
            "atomic_numbers": data_utils.pad_1d,
            "atom_mask": data_utils.pad_1d,
            "tags": data_utils.pad_1d,
            "main_cell": data_utils.pad_1d,
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
