# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from functools import lru_cache

import numpy as np
import torch
from unicore.data import Dictionary
from unicore.data import BaseWrapperDataset
from . import data_utils


class AttnBiasDataset(BaseWrapperDataset):
    def __init__(self, dataset,
            has_bos=True, has_eos=True,
            remove_hydrogen=False,
            remove_polar_hydrogen=False,
    ):
        super().__init__(dataset)
        self.dataset = dataset
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
        attn_bias = torch.zeros((num_atoms, num_atoms), dtype=torch.float32)
        return attn_bias


class PadBiasDataset2D(BaseWrapperDataset):
    def __init__(self, dataset, pad_idx,left_pad=False):
        super().__init__(dataset)
        self.pad_idx = pad_idx
        self.left_pad = left_pad

    def collater(self, samples):
        def collate_tokens_2d(
            values,
            pad_idx,
            pad_to_length=None,
            pad_to_multiple=1,
        ):
            """Convert a list of 1d tensors into a padded 2d tensor."""
            size = max(v.size(0) for v in values)
            size = size if pad_to_length is None else max(size, pad_to_length)
            if pad_to_multiple != 1 and size % pad_to_multiple != 0:
                size = int(((size - 0.1) // pad_to_multiple + 1) * pad_to_multiple)
            res = values[0].new(len(values), size, size).fill_(pad_idx)

            for i in range(len(values)):
                res[i, : values[i].shape[0], : values[i].shape[1]] = values[i]
                res[i, values[i].shape[0] :, : values[i].shape[1]] = 0
            return res
        return collate_tokens_2d(samples, self.pad_idx, pad_to_multiple=8)


class NoisePointsDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        coord_dataset: torch.utils.data.Dataset,
        vocab: Dictionary,
        pad_idx: int,
        mask_idx: int,
        noise_type: str,
        noise: float = 1.0,
        seed: int = 1,
        mask_prob: float = 0.15,
        mask_token_prob: float = 0.15,
        leave_unmasked_prob: float = 0.1,
        random_token_prob: float = 0.1,
        has_bos=True,
        has_eos=True,
    ):
        assert 0.0 < mask_prob <= 1.0
        assert 0.0 < mask_token_prob < 1.0

        assert 0.0 <= random_token_prob <= 1.0
        assert 0.0 <= leave_unmasked_prob <= 1.0
        assert random_token_prob + leave_unmasked_prob <= 1.0

        self.dataset = dataset
        self.coord_dataset = coord_dataset
        self.vocab = vocab
        self.pad_idx = pad_idx
        self.mask_idx = mask_idx
        self.noise_type = noise_type
        self.noise = noise
        self.seed = seed
        self.mask_prob = mask_prob
        self.mask_token_prob = mask_token_prob
        self.leave_unmasked_prob = leave_unmasked_prob
        self.random_token_prob = random_token_prob

        self.has_bos = has_bos
        self.has_eos = has_eos

        if random_token_prob > 0.0:
            weights = np.ones(len(self.vocab))
            weights[vocab.special_index()] = 0
            self.weights = weights / weights.sum()

        self.epoch = None
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

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.coord_dataset.set_epoch(epoch)
        self.dataset.set_epoch(epoch)
        self.epoch = epoch

    def __getitem__(self, index: int):
        return self.__getitem_cached__(self.epoch, index)

    @lru_cache(maxsize=16)
    def __getitem_cached__(self, epoch: int, index: int):
        ret = {}
        with data_utils.numpy_seed(self.seed, epoch, index):
            item = self.dataset[index]
            coord = self.coord_dataset[index]
            sz = len(item)
            # don't allow empty sequence
            assert sz > 0

            num_mask_token = int(
                # add a random number for probabilistic rounding
                self.mask_token_prob * sz
                + np.random.rand()
            )
            mask_idc = np.random.choice(sz, num_mask_token, replace=False)
            mask_token = np.full(sz, False)
            mask_token[mask_idc] = True
            ret["targets"] = np.full(len(mask_token), self.pad_idx)
            ret["targets"][mask_token] = item[mask_token]
            ret["targets"] = torch.from_numpy(ret["targets"]).long()
            # decide unmasking and random replacement
            rand_or_unmask_prob = self.random_token_prob + self.leave_unmasked_prob
            if rand_or_unmask_prob > 0.0:
                rand_or_unmask = mask_token & (np.random.rand(sz) < rand_or_unmask_prob)
                if self.random_token_prob == 0.0:
                    unmask = rand_or_unmask
                    rand_mask = None
                elif self.leave_unmasked_prob == 0.0:
                    unmask = None
                    rand_mask = rand_or_unmask
                else:
                    unmask_prob = self.leave_unmasked_prob / rand_or_unmask_prob
                    decision = np.random.rand(sz) < unmask_prob
                    unmask = rand_or_unmask & decision
                    rand_mask = rand_or_unmask & (~decision)
            else:
                unmask = rand_mask = None

            if unmask is not None:
                mask_token = mask_token ^ unmask

            new_item = np.copy(item)
            new_item[mask_token] = self.mask_idx

            if rand_mask is not None:
                num_rand = rand_mask.sum()
                if num_rand > 0:
                    new_item[rand_mask] = np.random.choice(
                        len(self.vocab),
                        num_rand,
                        p=self.weights,
                    )
            # ret["atoms"] = item
            ret["atoms"] = torch.from_numpy(new_item).long()

            # if num_rand > 0:
            #     print ("debug... unmask, rand_mask, num_rand ", unmask, rand_mask, num_rand)

            # mask cord
            # decide elements to mask
            num_mask = int(
                # add a random number for probabilistic rounding
                self.mask_prob * sz
                + np.random.rand()
            )

            mask_idc = np.random.choice(sz, num_mask, replace=False)
            mask = np.full(sz, False)
            mask[mask_idc] = True
            ret['mask_cord'] = torch.from_numpy(mask)

            new_coord = np.copy(coord)
            new_coord[mask, :] += self.noise_f(num_mask)

            ret["coordinates"] = torch.from_numpy(new_coord).float()

            # pading two dim data here.
            if self.has_bos:
                sz += 1
            if self.has_eos:
                sz += 1
            ret["attn_bias"] = torch.zeros((sz, sz), dtype=torch.float32)
            return ret