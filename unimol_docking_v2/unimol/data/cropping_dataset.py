# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from functools import lru_cache
import logging
from unicore.data import BaseWrapperDataset
from . import data_utils

logger = logging.getLogger(__name__)


class CroppingPocketDataset(BaseWrapperDataset):
    def __init__(self, dataset, seed, atoms, coordinates, holo_coordinates, max_atoms=256):
        self.dataset = dataset
        self.seed = seed
        self.atoms = atoms
        self.coordinates = coordinates
        self.holo_coordinates = holo_coordinates
        self.max_atoms = (
            max_atoms  # max number of atoms in a molecule, None indicates no limit.
        )
        self.set_epoch(None)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    @lru_cache(maxsize=16)
    def __cached_item__(self, index: int, epoch: int):
        dd = self.dataset[index].copy()
        atoms = dd[self.atoms]
        coordinates = dd[self.coordinates]
        # residue = dd["residue"]
        holo_coordinates = dd[self.holo_coordinates]

        # crop atoms according to their distance to the center of pockets
        if self.max_atoms and len(atoms) > self.max_atoms:
            with data_utils.numpy_seed(self.seed, epoch, index):
                distance = np.linalg.norm(
                    coordinates - coordinates.mean(axis=0), axis=1
                )

                def softmax(x):
                    x -= np.max(x)
                    x = np.exp(x) / np.sum(np.exp(x))
                    return x

                distance += 1  # prevent inf
                weight = softmax(np.reciprocal(distance))
                index = np.random.choice(
                    len(atoms), self.max_atoms, replace=False, p=weight
                )
                atoms = atoms[index]
                coordinates = coordinates[index]
                # residue = residue[index]
                holo_coordinates = holo_coordinates[index]

        dd[self.atoms] = atoms
        dd[self.coordinates] = coordinates.astype(np.float32)
        # dd["residue"] = residue
        dd[self.holo_coordinates] = holo_coordinates.astype(np.float32)
        return dd

    def __getitem__(self, index: int):
        return self.__cached_item__(index, self.epoch)
