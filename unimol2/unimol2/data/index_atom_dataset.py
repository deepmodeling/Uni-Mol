# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch

from functools import lru_cache
from unicore.data import BaseWrapperDataset

from rdkit import Chem
from rdkit.Chem import AllChem


class IndexAtomDataset(BaseWrapperDataset):
    def __init__(
        self,
        smi_dataset: torch.utils.data.Dataset,
        token_dataset: torch.utils.data.Dataset,
    ):
        super().__init__(smi_dataset)
        self.smi_dataset = smi_dataset
        self.token_dataset = token_dataset
        self.set_epoch(None)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    @lru_cache(maxsize=16)
    def __getitem__(self, index: int):
        atoms = self.token_dataset[index]

        atom_index = [
            AllChem.GetPeriodicTable().GetAtomicNumber(item) for item in atoms
        ]

        return np.array(atom_index)