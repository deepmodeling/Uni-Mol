# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from functools import lru_cache
import logging
from unicore.data import BaseWrapperDataset
from . import data_utils

logger = logging.getLogger(__name__)


class CroppingDataset(BaseWrapperDataset):
    def __init__(self, dataset, seed, atoms, coordinates, max_atoms=256):
        self.dataset = dataset
        self.seed = seed
        self.atoms = atoms
        self.coordinates = coordinates
        self.max_atoms = max_atoms
        self.set_epoch(None)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    @lru_cache(maxsize=16)
    def __cached_item__(self, index: int, epoch: int):
        dd = self.dataset[index].copy()
        atoms = dd[self.atoms]
        coordinates = dd[self.coordinates]
        if self.max_atoms and len(atoms) > self.max_atoms:
            with data_utils.numpy_seed(self.seed, epoch, index):
                index = np.random.choice(len(atoms), self.max_atoms, replace=False)
                atoms = np.array(atoms)[index]
                coordinates = coordinates[index]
        dd[self.atoms] = atoms
        dd[self.coordinates] = coordinates.astype(np.float32)
        return dd

    def __getitem__(self, index: int):
        return self.__cached_item__(index, self.epoch)


class CroppingPocketDataset(BaseWrapperDataset):
    def __init__(self, dataset, seed, atoms, coordinates, max_atoms=256):
        self.dataset = dataset
        self.seed = seed
        self.atoms = atoms
        self.coordinates = coordinates
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
        residue = dd["residue"]

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
                residue = residue[index]

        dd[self.atoms] = atoms
        dd[self.coordinates] = coordinates.astype(np.float32)
        dd["residue"] = residue
        return dd

    def __getitem__(self, index: int):
        return self.__cached_item__(index, self.epoch)


class CroppingResiduePocketDataset(BaseWrapperDataset):
    def __init__(self, dataset, seed, atoms, residues, coordinates, max_atoms=256):
        self.dataset = dataset
        self.seed = seed
        self.atoms = atoms
        self.residues = residues
        self.coordinates = coordinates
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
        residues = dd[self.residues]
        coordinates = dd[self.coordinates]

        residues_distance_map = {}

        # crop atoms according to their distance to the center of pockets
        if self.max_atoms and len(atoms) > self.max_atoms:
            with data_utils.numpy_seed(self.seed, epoch, index):
                distance = np.linalg.norm(
                    coordinates - coordinates.mean(axis=0), axis=1
                )
                residues_ids, residues_distance = [], []
                for res in residues:
                    if res not in residues_ids:
                        residues_ids.append(res)
                        residues_distance.append(distance[residues == res].mean())
                residues_ids = np.array(residues_ids)
                residues_distance = np.array(residues_distance)

                def softmax(x):
                    x -= np.max(x)
                    x = np.exp(x) / np.sum(np.exp(x))
                    return x

                residues_distance += 1  # prevent inf and smoothing out the distance
                weight = softmax(np.reciprocal(residues_distance))
                max_residues = self.max_atoms // (len(atoms) // (len(residues_ids) + 1))
                if max_residues < 1:
                    max_residues += 1
                max_residues = min(max_residues, len(residues_ids))
                residue_index = np.random.choice(
                    len(residues_ids), max_residues, replace=False, p=weight
                )
                index = [
                    i
                    for i in range(len(atoms))
                    if residues[i] in residues_ids[residue_index]
                ]
                atoms = atoms[index]
                coordinates = coordinates[index]
                residues = residues[index]

        dd[self.atoms] = atoms
        dd[self.coordinates] = coordinates.astype(np.float32)
        dd[self.residues] = residues
        return dd

    def __getitem__(self, index: int):
        return self.__cached_item__(index, self.epoch)


class CroppingPocketDockingPoseDataset(BaseWrapperDataset):
    def __init__(
        self, dataset, seed, atoms, coordinates, holo_coordinates, max_atoms=256
    ):
        self.dataset = dataset
        self.seed = seed
        self.atoms = atoms
        self.coordinates = coordinates
        self.holo_coordinates = holo_coordinates
        self.max_atoms = max_atoms

        self.set_epoch(None)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    @lru_cache(maxsize=16)
    def __cached_item__(self, index: int, epoch: int):
        dd = self.dataset[index].copy()
        atoms = dd[self.atoms]
        coordinates = dd[self.coordinates]
        holo_coordinates = dd[self.holo_coordinates]

        # crop atoms according to their distance to the center of pockets
        if self.max_atoms and len(atoms) > self.max_atoms:
            with data_utils.numpy_seed(self.seed, epoch):
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
                holo_coordinates = holo_coordinates[index]

        dd[self.atoms] = atoms
        dd[self.coordinates] = coordinates.astype(np.float32)
        dd[self.holo_coordinates] = holo_coordinates.astype(np.float32)
        return dd

    def __getitem__(self, index: int):
        return self.__cached_item__(index, self.epoch)
