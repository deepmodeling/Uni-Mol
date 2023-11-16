# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function

import logging
import copy
import os
import pandas as pd
import numpy as np
import csv
from typing import List, Optional
from sklearn.model_selection import (
    GroupKFold, 
    KFold, 
    StratifiedKFold,
)


class Splitter(object):
    def __init__(self, split_method='5fold_random', seed=42):
        self.n_splits, self.method = int(split_method.split('fold')[0]), split_method.split('_')[-1]    # Nfold_xxxx
        self.seed = seed
        self.splitter = self._init_split()

    def _init_split(self):
        if self.method == 'random':
            splitter = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
        elif self.method == 'scaffold' or self.method == 'group':
            splitter = GroupKFold(n_splits=self.n_splits)
        elif self.method == 'stratified':
            splitter = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
        else:
            raise ValueError('Unknown splitter method: {}fold - {}'.format(self.n_splits, self.method))

        return splitter

    def split(self, data, target=None, group=None):
        try:
            return self.splitter.split(data, target, group)
        except:
            raise ValueError('Unknown splitter method: {}fold - {}'.format(self.n_splits, self.method))
