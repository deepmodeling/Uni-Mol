# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from functools import lru_cache
from unicore.data import BaseWrapperDataset


class KeyDataset(BaseWrapperDataset):
    def __init__(self, dataset, key):
        self.dataset = dataset
        self.key = key
        self.epoch = None

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    def __len__(self):
        return len(self.dataset)

    @lru_cache(maxsize=16)
    def __cached_item__(self, idx: int, epoch: int):
        return self.dataset[idx][self.key]

    def __getitem__(self, idx):
        return self.__cached_item__(idx, self.epoch)
