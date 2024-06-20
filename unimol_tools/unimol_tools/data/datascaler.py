# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function

import os
import numpy as np
import joblib
from sklearn.preprocessing import (
    StandardScaler, 
    MinMaxScaler, 
    MaxAbsScaler, 
    RobustScaler, 
    Normalizer, 
    QuantileTransformer, 
    PowerTransformer, 
    FunctionTransformer,
)
from scipy.stats import skew, kurtosis
from ..utils import logger


SCALER_MODE = {
    'minmax': MinMaxScaler,
    'standard': StandardScaler,
    'robust': RobustScaler,
    'maxabs': MaxAbsScaler,
    'quantile': QuantileTransformer,
    'power_trans': PowerTransformer,
    'normalizer': Normalizer,
    'log1p': FunctionTransformer,
}

class TargetScaler(object):
    '''
    A class to scale the target.
    '''
    def __init__(self, ss_method, task, load_dir=None):
        """
        Initializes the TargetScaler object for scaling target values.

        :param ss_method: (str) The scaling method to be used. 
        :param task: (str) The type of machine learning task (e.g., 'classification', 'regression').
        :param load_dir: (str, optional) Directory from which to load an existing scaler.
        """
        self.ss_method = ss_method
        self.task = task
        if load_dir and os.path.exists(os.path.join(load_dir, 'target_scaler.ss')):
            self.scaler = joblib.load(os.path.join(load_dir, 'target_scaler.ss'))
        else:
            self.scaler = None
    
    def transform(self, target):
        """
        Transforms the target values using the appropriate scaling method.

        :param target: (array-like) The target values to be transformed.

        :return: (array-like) The transformed target values.
        """
        if self.task in ['classification', 'multiclass', 'multilabel_classification']:
            return target
        elif self.ss_method == 'none':
            return target 
        elif self.task == 'regression':
            return self.scaler.transform(target)
        elif self.task == 'multilabel_regression':
            assert isinstance(self.scaler, list) and len(self.scaler) == target.shape[1]
            target = np.ma.masked_invalid(target)  # mask NaN value
            new_target = np.zeros_like(target)
            for i in range(target.shape[1]):
                new_target[:, i] = self.scaler[i].transform(target[:, i:i+1]).reshape(-1,)
            return new_target
        else:
            return target
        
    def fit(self, target, dump_dir):
        """
        Fits the scaler to the target values and optionally saves the scaler to disk.

        :param target: (array-like) The target values to fit the scaler.
        :param dump_dir: (str) Directory where the fitted scaler will be saved.
        """
        if self.task in ['classification', 'multiclass', 'multilabel_classification']:
            return 
        elif self.ss_method == 'none':
            return 
        elif self.ss_method == 'auto':
            if self.task == 'regression':
                if self.is_skewed(target):
                    self.scaler = SCALER_MODE['robust']()
                    logger.info('Auto select robust transformer.')
                else:
                    self.scaler = SCALER_MODE['standard']()
                self.scaler.fit(target)
            elif self.task == 'multilabel_regression':
                self.scaler = []
                target = np.ma.masked_invalid(target)  # mask NaN value
                for i in range(target.shape[1]):
                    if self.is_skewed(target[:, i]):
                        self.scaler.append(SCALER_MODE['robust']())
                        logger.info('Auto select robust transformer.')
                    else:
                        self.scaler.append(SCALER_MODE['standard']())
                    self.scaler[-1].fit(target[:, i:i+1])
        else:
            if self.task == 'regression':
                self.scaler = self.scaler_choose(self.ss_method, target)
                self.scaler.fit(target)
            elif self.task == 'multilabel_regression':
                self.scaler = []
                for i in range(target.shape[1]):
                    self.scaler.append(self.scaler_choose(self.ss_method, target[:, i:i+1]))
                    self.scaler[-1].fit(target[:, i:i+1])
        try:
            os.remove(os.path.join(dump_dir, 'target_scaler.ss'))
        except:
            pass
        os.makedirs(dump_dir, exist_ok=True)
        joblib.dump(self.scaler, os.path.join(dump_dir, 'target_scaler.ss'))
    
    def scaler_choose(self, method, target):
        """
        Selects the appropriate scaler based on the scaling method and fit it to the target.

        :param method: (str) The scaling method to be used.

            currently support:

                - 'minmax': MinMaxScaler,

                - 'standard': StandardScaler,

                - 'robust': RobustScaler,

                - 'maxabs': MaxAbsScaler,

                - 'quantile': QuantileTransformer,

                - 'power_trans': PowerTransformer,

                - 'normalizer': Normalizer,

                - 'log1p': FunctionTransformer,
            
        :param target: (array-like) The target values to fit the scaler.
        :return: The fitted scaler object.
        """
        if method=='power_trans':
            scaler = SCALER_MODE[method](method='box-cox') if min(target) > 0 else SCALER_MODE[method](method='yeo-johnson')
        elif method=='log1p':
            scaler = SCALER_MODE[method](np.log1p)              
        else:
            scaler = SCALER_MODE[method]()
        return scaler

    def inverse_transform(self, target):
        """
        Inverse transforms the scaled target values back to their original scale.

        :param target: (array-like) The target values to be inverse transformed.

        :return: (array-like) The target values in their original scale.
        """
        if self.task in ['classification', 'multiclass', 'multilabel_classification']:
            return target
        if self.ss_method == 'none' or self.scaler is None:
            return target
        elif self.task == 'regression':
            return self.scaler.inverse_transform(target)
        elif self.task == 'multilabel_regression':
            assert isinstance(self.scaler, list) and len(self.scaler) == target.shape[1]
            new_target = np.zeros_like(target)
            for i in range(target.shape[1]):
                new_target[:, i] = self.scaler[i].inverse_transform(target[:, i:i+1]).reshape(-1,)
            return new_target
        else:
            raise ValueError('Unknown scaler method: {}'.format(self.ss_method))
            
    def is_skewed(self, target):
        """
        Determines whether the target values are skewed based on skewness and kurtosis metrics.

        :param target: (array-like) The target values to be checked for skewness.

        :return: (bool) True if the target is skewed, False otherwise.
        """
        if self.task in ['classification', 'multiclass', 'multilabel_classification']:
            return False
        else:
            return abs(skew(target)) > 5.0 or abs(kurtosis(target)) > 20.0
