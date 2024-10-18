# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from functools import lru_cache
from unicore.data import BaseWrapperDataset
from . import data_utils


class ConformerSampleDataset(BaseWrapperDataset):
    def __init__(self, dataset, seed, atoms, coordinates, coordinates_2d):
        self.dataset = dataset
        self.seed = seed
        self.atoms = atoms
        self.coordinates = coordinates
        self.coordinates_2d = coordinates_2d
        self.set_epoch(None)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    @lru_cache(maxsize=16)
    def __cached_item__(self, index: int, epoch: int):
        atoms = np.array(self.dataset[index][self.atoms])
        assert len(atoms) > 0
        coordinates_list = self.dataset[index][self.coordinates]
        if not isinstance(coordinates_list, list):
            coordinates_list = [coordinates_list]

        size = len(coordinates_list)
        with data_utils.numpy_seed(self.seed, epoch, index):
            sample_idx = np.random.randint(size)
        coordinates = coordinates_list[sample_idx]

        return {
            "atoms": atoms,
            "coordinates": coordinates.astype(np.float32),
            "coordinates_2d": self.dataset[index][self.coordinates_2d]
        }

    def __getitem__(self, index: int):
        return self.__cached_item__(index, self.epoch)