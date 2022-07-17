# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from functools import lru_cache
from unicore.data import BaseWrapperDataset, data_utils


class ConformerSampleDataset(BaseWrapperDataset):
    def __init__(self, dataset, seed, atoms, coordinates):
        self.dataset = dataset
        self.seed = seed
        self.atoms = atoms
        self.coordinates = coordinates
        self.set_epoch(None)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    @lru_cache(maxsize=16)
    def __cached_item__(self, index: int, epoch: int):
        atoms = np.array(self.dataset[index][self.atoms])
        assert len(atoms) > 0
        size = len(self.dataset[index][self.coordinates])
        with data_utils.numpy_seed(self.seed, epoch, index):
            sample_idx = np.random.randint(size)
        coordinates = self.dataset[index][self.coordinates][sample_idx]
        return {'atoms': atoms, 'coordinates': coordinates.astype(np.float32)}

    def __getitem__(self, index: int):
        return self.__cached_item__(index, self.epoch)


class ConformerSamplePocketDataset(BaseWrapperDataset):
    def __init__(self, dataset, seed, atoms, coordinates, dict_name):
        self.dataset = dataset
        self.seed = seed
        self.atoms = atoms
        self.dict_name = dict_name
        self.coordinates = coordinates
        self.set_epoch(None)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    @lru_cache(maxsize=16)
    def __cached_item__(self, index: int, epoch: int):
        if self.dict_name == 'dict_coarse.txt':
            atoms = np.array([a[0] for a in self.dataset[index][self.atoms]])
        elif self.dict_name == 'dict_fine.txt':
            atoms = np.array([a[0] if len(a) == 1 or a[0] == 'H' else a[:2] for a in self.dataset[index][self.atoms]])
        assert len(atoms) > 0
        size = len(self.dataset[index][self.coordinates])
        with data_utils.numpy_seed(self.seed, epoch, index):
            sample_idx = np.random.randint(size)
        coordinates = self.dataset[index][self.coordinates][sample_idx]
        residue = np.array(self.dataset[index]['residue'])
        score = np.float(self.dataset[index]['meta_info']['fpocket']['Score'])
        return {'atoms': atoms, 'coordinates': coordinates.astype(np.float32), 'residue': residue, 'score': score}

    def __getitem__(self, index: int):
        return self.__cached_item__(index, self.epoch)


class ConformerSamplePocketFinetuneDataset(BaseWrapperDataset):
    def __init__(self, dataset, seed, atoms, residues, coordinates):
        self.dataset = dataset
        self.seed = seed
        self.atoms = atoms
        self.residues = residues
        self.coordinates = coordinates
        self.set_epoch(None)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    @lru_cache(maxsize=16)
    def __cached_item__(self, index: int, epoch: int):
        atoms = np.array([a[0] for a in self.dataset[index][self.atoms]])  # only 'C H O N S'
        assert len(atoms) > 0
        if isinstance(self.dataset[index][self.coordinates], list):
            size = len(self.dataset[index][self.coordinates])
            with data_utils.numpy_seed(self.seed, epoch, index):
                sample_idx = np.random.randint(size)
            coordinates = self.dataset[index][self.coordinates][sample_idx]
        else:
            coordinates = self.dataset[index][self.coordinates]

        if self.residues in self.dataset[index]:
            residues = np.array(self.dataset[index][self.residues])
        else:
            residues = None
        assert len(atoms) == len(coordinates)
        return {self.atoms: atoms,
                self.coordinates: coordinates.astype(np.float32),
                self.residues: residues}

    def __getitem__(self, index: int):
        return self.__cached_item__(index, self.epoch)
