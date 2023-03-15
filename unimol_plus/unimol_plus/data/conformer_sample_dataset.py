# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from functools import lru_cache
from unicore.data import BaseWrapperDataset, data_utils
from copy import deepcopy
from tqdm import tqdm


class ConformationSampleDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset,
        seed,
        coordinates,
        target_coordinates,
    ):
        self.dataset = dataset
        self.seed = seed
        self.coordinates = coordinates
        self.target_coordinates = target_coordinates
        self.set_epoch(None)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    @lru_cache(maxsize=16)
    def __cached_item__(self, index: int, epoch: int):
        data = deepcopy(self.dataset[index])
        size = len(data[self.coordinates])
        with data_utils.numpy_seed(self.seed, epoch, index):
            sample_idx = np.random.randint(size)
        coordinates = data[self.coordinates][sample_idx]
        if isinstance(data[self.target_coordinates], list):
            target_coordinates = data[self.target_coordinates][-1]
        else:
            target_coordinates = data[self.target_coordinates]
        del data[self.coordinates]
        del data[self.target_coordinates]
        data["coordinates"] = coordinates
        data["target_coordinates"] = target_coordinates
        return data

    def __getitem__(self, index: int):
        return self.__cached_item__(index, self.epoch)


class ConformationExpandDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset,
        seed,
        coordinates,
        target_coordinates,
    ):
        self.dataset = dataset
        self.seed = seed
        self.coordinates = coordinates
        self.target_coordinates = target_coordinates
        self._init_idx()
        self.set_epoch(None)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    def _init_idx(self):
        self.idx2key = []
        for i in tqdm(range(len(self.dataset))):
            size = len(self.dataset[i][self.coordinates])
            self.idx2key.extend([(i, j) for j in range(size)])
        self.cnt = len(self.idx2key)

    def __len__(self):
        return self.cnt

    @lru_cache(maxsize=16)
    def __cached_item__(self, index: int, epoch: int):
        key_idx, conf_idx = self.idx2key[index]
        data = self.dataset[key_idx]
        coordinates = data[self.coordinates][conf_idx]
        if isinstance(data[self.target_coordinates], list):
            target_coordinates = data[self.target_coordinates][-1]
        else:
            target_coordinates = data[self.target_coordinates]

        ret_data = deepcopy(data)
        del ret_data[self.coordinates]
        del ret_data[self.target_coordinates]
        ret_data["coordinates"] = coordinates
        ret_data["target_coordinates"] = target_coordinates
        return ret_data

    def __getitem__(self, index: int):
        return self.__cached_item__(index, self.epoch)
