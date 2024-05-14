# Copyright (c) DP Techonology, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from functools import lru_cache
import logging
from unicore.data import BaseWrapperDataset
from scipy.spatial.transform import Rotation

logger = logging.getLogger(__name__)


class ReAlignLigandDataset(BaseWrapperDataset):
    def __init__(self, dataset, coordinates, pocket_coordinates):
        self.dataset = dataset
        self.coordinates = coordinates
        self.pocket_coordinates = pocket_coordinates
        self.set_epoch(None)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    @lru_cache(maxsize=16)
    def __cached_item__(self, index: int, epoch: int):
        dd = self.dataset[index].copy()
        coordinates = dd[self.coordinates]
        pocket_coordinates = dd[self.pocket_coordinates]
        normal_coordinates, normal_pocket_coordinates = realigncoordinates(coordinates, pocket_coordinates)

        dd[self.coordinates] = normal_coordinates.astype(np.float32)
        dd[self.pocket_coordinates] = normal_pocket_coordinates.astype(np.float32)
        return dd


    def __getitem__(self, index: int):
        return self.__cached_item__(index, self.epoch)

def calc_inertia_tensor(new_coord, mass=None):
    """  This function calculates the Elements of inertia tensor for the
    center-moved coordinates.
    """
    if mass is None:
        mass = 1.0
    I_xx = (mass * np.sum(np.square(new_coord[:,1:3:1]),axis=1)).sum()
    I_yy = (mass * np.sum(np.square(new_coord[:,0:3:2]),axis=1)).sum()
    I_zz = (mass * np.sum(np.square(new_coord[:,0:2:1]),axis=1)).sum()
    I_xy = (-1 * mass * np.prod(new_coord[:,0:2:1],axis=1)).sum()
    I_yz = (-1 * mass * np.prod(new_coord[:,1:3:1],axis=1)).sum()
    I_xz = (-1 * mass * np.prod(new_coord[:,0:3:2],axis=1)).sum()
    I = np.array([[I_xx, I_xy, I_xz],
		  [I_xy, I_yy, I_yz],
		  [I_xz, I_yz, I_zz]])
    return I

def realigncoordinates(coordinates, pocket_coordinates):
    coordinates = coordinates - coordinates.mean(axis=0)
    pocket_coordinates = pocket_coordinates - pocket_coordinates.mean(axis=0)

    D = calc_inertia_tensor(coordinates)
    I, E = np.linalg.eigh(D)

    D_poc = calc_inertia_tensor(pocket_coordinates)
    I_poc, E_poc = np.linalg.eigh(D_poc)
    
    _R, _score = Rotation.align_vectors(E[:,:].T, E_poc[:,:].T)
    new_coordinates = np.dot(coordinates, _R.as_matrix())

    return new_coordinates, pocket_coordinates


    
