# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from functools import lru_cache
from unicore.data import BaseWrapperDataset
from . import data_utils


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
        return {"atoms": atoms, "coordinates": coordinates.astype(np.float32)}

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
        if self.dict_name == "dict_coarse.txt":
            atoms = np.array([a[0] for a in self.dataset[index][self.atoms]])
        elif self.dict_name == "dict_fine.txt":
            atoms = np.array(
                [
                    a[0] if len(a) == 1 or a[0] == "H" else a[:2]
                    for a in self.dataset[index][self.atoms]
                ]
            )
        assert len(atoms) > 0
        size = len(self.dataset[index][self.coordinates])
        with data_utils.numpy_seed(self.seed, epoch, index):
            sample_idx = np.random.randint(size)
        coordinates = self.dataset[index][self.coordinates][sample_idx]
        residue = np.array(self.dataset[index]["residue"])
        score = np.float(self.dataset[index]["meta_info"]["fpocket"]["Score"])
        return {
            "atoms": atoms,
            "coordinates": coordinates.astype(np.float32),
            "residue": residue,
            "score": score,
        }

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
        atoms = np.array(
            [a[0] for a in self.dataset[index][self.atoms]]
        )  # only 'C H O N S'
        assert len(atoms) > 0
        # This judgment is reserved for possible future expansion.
        # The number of pocket conformations is 1, and the 'sample' does not work.
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
        return {
            self.atoms: atoms,
            self.coordinates: coordinates.astype(np.float32),
            self.residues: residues,
        }

    def __getitem__(self, index: int):
        return self.__cached_item__(index, self.epoch)


class ConformerSampleConfGDataset(BaseWrapperDataset):
    def __init__(self, dataset, seed, atoms, coordinates, tgt_coordinates):
        self.dataset = dataset
        self.seed = seed
        self.atoms = atoms
        self.coordinates = coordinates
        self.tgt_coordinates = tgt_coordinates
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
        tgt_coordinates = self.dataset[index][self.tgt_coordinates]
        return {
            self.atoms: atoms,
            self.coordinates: coordinates.astype(np.float32),
            self.tgt_coordinates: tgt_coordinates.astype(np.float32),
        }

    def __getitem__(self, index: int):
        return self.__cached_item__(index, self.epoch)


class ConformerSampleConfGV2Dataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset,
        seed,
        atoms,
        coordinates,
        tgt_coordinates,
        beta=1.0,
        smooth=0.1,
        topN=10,
    ):
        self.dataset = dataset
        self.seed = seed
        self.atoms = atoms
        self.coordinates = coordinates
        self.tgt_coordinates = tgt_coordinates
        self.beta = beta
        self.smooth = smooth
        self.topN = topN
        self.set_epoch(None)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    @lru_cache(maxsize=16)
    def __cached_item__(self, index: int, epoch: int):
        atoms = np.array(self.dataset[index][self.atoms])
        assert len(atoms) > 0
        meta_df = self.dataset[index]["meta"]
        tgt_conf_ids = meta_df["gid"].unique()
        # randomly choose one conf
        with data_utils.numpy_seed(self.seed, epoch, index):
            conf_id = np.random.choice(tgt_conf_ids)
        conf_df = meta_df[meta_df["gid"] == conf_id]
        conf_df = conf_df.sort_values("score").reset_index(drop=False)[
            : self.topN
        ]  # only use top 5 confs for sampling...
        # importance sampling with rmsd inverse score

        def normalize(x, beta=1.0, smooth=0.1):
            x = 1.0 / (x**beta + smooth)
            return x / x.sum()

        rmsd_score = conf_df["score"].values
        weight = normalize(
            rmsd_score, beta=self.beta, smooth=self.smooth
        )  # for smoothing purpose
        with data_utils.numpy_seed(self.seed, epoch, index):
            idx = np.random.choice(len(conf_df), 1, replace=False, p=weight)
        # idx = [np.argmax(weight)]
        coordinates = conf_df.iloc[idx]["rdkit_coords"].values[0]
        tgt_coordinates = conf_df.iloc[idx]["tgt_coords"].values[0]
        return {
            self.atoms: atoms,
            self.coordinates: coordinates.astype(np.float32),
            self.tgt_coordinates: tgt_coordinates.astype(np.float32),
        }

    def __getitem__(self, index: int):
        return self.__cached_item__(index, self.epoch)


class ConformerSampleDockingPoseDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset,
        seed,
        atoms,
        coordinates,
        pocket_atoms,
        pocket_coordinates,
        holo_coordinates,
        holo_pocket_coordinates,
        is_train=True,
    ):
        self.dataset = dataset
        self.seed = seed
        self.atoms = atoms
        self.coordinates = coordinates
        self.pocket_atoms = pocket_atoms
        self.pocket_coordinates = pocket_coordinates
        self.holo_coordinates = holo_coordinates
        self.holo_pocket_coordinates = holo_pocket_coordinates
        self.is_train = is_train
        self.set_epoch(None)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    @lru_cache(maxsize=16)
    def __cached_item__(self, index: int, epoch: int):
        atoms = np.array(self.dataset[index][self.atoms])
        size = len(self.dataset[index][self.coordinates])
        with data_utils.numpy_seed(self.seed, epoch, index):
            sample_idx = np.random.randint(size)
        coordinates = self.dataset[index][self.coordinates][sample_idx]
        pocket_atoms = np.array(
            [item[0] for item in self.dataset[index][self.pocket_atoms]]
        )
        pocket_coordinates = self.dataset[index][self.pocket_coordinates][0]
        if self.is_train:
            holo_coordinates = self.dataset[index][self.holo_coordinates][0]
            holo_pocket_coordinates = self.dataset[index][self.holo_pocket_coordinates][
                0
            ]
        else:
            holo_coordinates = coordinates
            holo_pocket_coordinates = pocket_coordinates

        smi = self.dataset[index]["smi"]
        pocket = self.dataset[index]["pocket"]

        return {
            "atoms": atoms,
            "coordinates": coordinates.astype(np.float32),
            "pocket_atoms": pocket_atoms,
            "pocket_coordinates": pocket_coordinates.astype(np.float32),
            "holo_coordinates": holo_coordinates.astype(np.float32),
            "holo_pocket_coordinates": holo_pocket_coordinates.astype(np.float32),
            "smi": smi,
            "pocket": pocket,
        }

    def __getitem__(self, index: int):
        return self.__cached_item__(index, self.epoch)
