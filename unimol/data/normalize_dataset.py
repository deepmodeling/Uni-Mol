# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from functools import lru_cache
from unicore.data import BaseWrapperDataset


class NormalizeDataset(BaseWrapperDataset):
    def __init__(self, dataset, coordinates, normalize_coord=True):
        self.dataset = dataset
        self.coordinates = coordinates
        self.normalize_coord = normalize_coord  # normalize the coordinates.
        self.set_epoch(None)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    @lru_cache(maxsize=16)
    def __cached_item__(self, index: int, epoch: int):
        dd = self.dataset[index].copy()
        coordinates = dd[self.coordinates]
        # normalize
        if self.normalize_coord:
            coordinates = coordinates - coordinates.mean(axis=0)
            dd[self.coordinates] = coordinates.astype(np.float32)
        return dd

    def __getitem__(self, index: int):
        return self.__cached_item__(index, self.epoch)


class NormalizeDockingPoseDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset,
        coordinates,
        pocket_coordinates,
        center_coordinates="center_coordinates",
    ):
        self.dataset = dataset
        self.coordinates = coordinates
        self.pocket_coordinates = pocket_coordinates
        self.center_coordinates = center_coordinates
        self.set_epoch(None)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    @lru_cache(maxsize=16)
    def __cached_item__(self, index: int, epoch: int):
        dd = self.dataset[index].copy()
        coordinates = dd[self.coordinates]
        pocket_coordinates = dd[self.pocket_coordinates]
        # normalize coordinates and pocket coordinates ,align with pocket center coordinates
        center_coordinates = pocket_coordinates.mean(axis=0)
        coordinates = coordinates - center_coordinates
        pocket_coordinates = pocket_coordinates - center_coordinates
        dd[self.coordinates] = coordinates.astype(np.float32)
        dd[self.pocket_coordinates] = pocket_coordinates.astype(np.float32)
        dd[self.center_coordinates] = center_coordinates.astype(np.float32)
        return dd

    def __getitem__(self, index: int):
        return self.__cached_item__(index, self.epoch)
