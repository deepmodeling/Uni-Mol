# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function

import numpy as np
from sklearn.model_selection import (
    GroupKFold, 
    KFold, 
    StratifiedKFold,
)
from ..utils import logger


class Splitter(object):
    """
    The Splitter class is responsible for splitting a dataset into train and test sets 
    based on the specified method.
    """
    def __init__(self, method='random', kfold=5, seed=42, **params):
        """
        Initializes the Splitter with a specified split method and random seed.

        :param split_method: (str) The method for splitting the dataset, in the format 'Nfold_method'. 
                             Defaults to '5fold_random'.
        :param seed: (int) Random seed for reproducibility in random splitting. Defaults to 42.
        """
        self.method = method
        self.n_splits = kfold
        self.seed = seed
        self.splitter = self._init_split()

    def _init_split(self):
        """
        Initializes the actual splitter object based on the specified method.

        :return: The initialized splitter object.
        :raises ValueError: If an unknown splitting method is specified.
        """
        if self.n_splits == 1:
            return None
        if self.method == 'random':
            splitter = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
        elif self.method == 'scaffold' or self.method == 'group':
            splitter = GroupKFold(n_splits=self.n_splits)
        elif self.method == 'stratified':
            splitter = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
        elif self.method == 'select':
            splitter = GroupKFold(n_splits=self.n_splits)
        else:
            raise ValueError('Unknown splitter method: {}fold - {}'.format(self.n_splits, self.method))

        return splitter

    def split(self, smiles, target=None, group=None, scaffolds=None, **params):
        """
        Splits the dataset into train and test sets based on the initialized method.

        :param data: The dataset to be split.
        :param target: (optional) Target labels for stratified splitting. Defaults to None.
        :param group: (optional) Group labels for group-based splitting. Defaults to None.

        :return: An iterator yielding train and test set indices for each fold.
        :raises ValueError: If the splitter method does not support the provided parameters.
        """
        if self.n_splits == 1:
            logger.warning('Only one fold is used for training, no splitting is performed.')
            return [(np.arange(len(smiles)), ())]
        if smiles is None and 'atoms' in params:
            smiles = params['atoms']
            logger.warning('Atoms are used as SMILES for splitting.')
        if self.method in ['random']:
            self.skf = self.splitter.split(smiles)
        elif self.method in ['scaffold']:
            self.skf = self.splitter.split(smiles, target, scaffolds)
        elif self.method in ['group']:
            self.skf = self.splitter.split(smiles, target, group)
        elif self.method in ['stratified']:
            self.skf = self.splitter.split(smiles, group)
        elif self.method in ['select']:
            unique_groups = np.unique(group)
            if len(unique_groups) == self.n_splits:
                split_folds = []
                for unique_group in unique_groups:
                    train_idx = np.where(group != unique_group)[0]
                    test_idx = np.where(group == unique_group)[0]
                    split_folds.append((train_idx, test_idx))
                self.split_folds = split_folds
                return self.split_folds
            else:
                logger.error('The number of unique groups is not equal to the number of splits.')
                exit(1)
        else:
            logger.error('Unknown splitter method: {}'.format(self.method))
            exit(1)
        self.split_folds = list(self.skf)
        return self.split_folds
