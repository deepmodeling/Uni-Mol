# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function

from sklearn.model_selection import (
    GroupKFold, 
    KFold, 
    StratifiedKFold,
)

class Splitter(object):
    """
    The Splitter class is responsible for splitting a dataset into train and test sets 
    based on the specified method.
    """
    def __init__(self, split_method='5fold_random', seed=42):
        """
        Initializes the Splitter with a specified split method and random seed.

        :param split_method: (str) The method for splitting the dataset, in the format 'Nfold_method'. 
                             Defaults to '5fold_random'.
        :param seed: (int) Random seed for reproducibility in random splitting. Defaults to 42.
        """
        self.n_splits, self.method = int(split_method.split('fold')[0]), split_method.split('_')[-1]    # Nfold_xxxx
        self.seed = seed
        self.splitter = self._init_split()

    def _init_split(self):
        """
        Initializes the actual splitter object based on the specified method.

        :return: The initialized splitter object.
        :raises ValueError: If an unknown splitting method is specified.
        """
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
        """
        Splits the dataset into train and test sets based on the initialized method.

        :param data: The dataset to be split.
        :param target: (optional) Target labels for stratified splitting. Defaults to None.
        :param group: (optional) Group labels for group-based splitting. Defaults to None.

        :return: An iterator yielding train and test set indices for each fold.
        :raises ValueError: If the splitter method does not support the provided parameters.
        """
        try:
            return self.splitter.split(data, target, group)
        except:
            raise ValueError('Unknown splitter method: {}fold - {}'.format(self.n_splits, self.method))
