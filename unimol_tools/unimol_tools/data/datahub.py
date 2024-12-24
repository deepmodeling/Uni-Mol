# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function
import numpy as np
from .datareader import MolDataReader
from .datascaler import TargetScaler
from .conformer import ConformerGen, UniMolV2Feature
from .split import Splitter
from ..utils import logger


class DataHub(object):
    """
    The DataHub class is responsible for storing and preprocessing data for machine learning tasks.
    It initializes with configuration options to handle different types of tasks such as regression, 
    classification, and others. It also supports data scaling and handling molecular data.
    """
    def __init__(self, data=None, is_train=True, save_path=None, **params):
        """
        Initializes the DataHub instance with data and configuration for the ML task.

        :param data: Initial dataset to be processed.
        :param is_train: (bool) Indicates if the DataHub is being used for training.
        :param save_path: (str) Path to save any necessary files, like scalers.
        :param params: Additional parameters for data preprocessing and model configuration.
        """
        self.data = data
        self.is_train = is_train
        self.save_path = save_path
        self.task = params.get('task', None)
        self.target_cols = params.get('target_cols', None)
        self.multiclass_cnt = params.get('multiclass_cnt', None)
        self.ss_method = params.get('target_normalize', 'none')
        self._init_data(**params)
        self._init_split(**params)
    
    def _init_data(self, **params):
        """
        Initializes and preprocesses the data based on the task and parameters provided.

        This method handles reading raw data, scaling targets, and transforming data for use with 
        molecular inputs. It tailors the preprocessing steps based on the task type, such as regression 
        or classification.

        :param params: Additional parameters for data processing.
        :raises ValueError: If the task type is unknown.
        """
        self.data = MolDataReader().read_data(self.data, self.is_train, **params)
        self.data['target_scaler'] = TargetScaler(self.ss_method, self.task, self.save_path)
        if self.task == 'regression': 
            target = np.array(self.data['raw_target']).reshape(-1,1).astype(np.float32)
            if self.is_train:
                self.data['target_scaler'].fit(target, self.save_path)
                self.data['target'] = self.data['target_scaler'].transform(target)
            else:
                self.data['target'] = target
        elif self.task == 'classification':
            target = np.array(self.data['raw_target']).reshape(-1,1).astype(np.int32)
            self.data['target'] = target
        elif self.task =='multiclass':
            target = np.array(self.data['raw_target']).reshape(-1,1).astype(np.int32)
            self.data['target'] = target
            if not self.is_train:
                self.data['multiclass_cnt'] = self.multiclass_cnt 
        elif self.task == 'multilabel_regression':
            target = np.array(self.data['raw_target']).reshape(-1,self.data['num_classes']).astype(np.float32)
            if self.is_train:
                self.data['target_scaler'].fit(target, self.save_path)
                self.data['target'] = self.data['target_scaler'].transform(target)                
            else:
                self.data['target'] = target
        elif self.task == 'multilabel_classification':
            target = np.array(self.data['raw_target']).reshape(-1,self.data['num_classes']).astype(np.int32)
            self.data['target'] = target
        elif self.task == 'repr':
            self.data['target'] = self.data['raw_target']
        else:
            raise ValueError('Unknown task: {}'.format(self.task))

        if params.get('model_name', None) == 'unimolv1':
            if 'atoms' in self.data and 'coordinates' in self.data:
                no_h_list = ConformerGen(**params).transform_raw(self.data['atoms'], self.data['coordinates'])
            else:
                smiles_list = self.data["smiles"]                  
                no_h_list = ConformerGen(**params).transform(smiles_list)
        elif params.get('model_name', None) == 'unimolv2':
            if 'atoms' in self.data and 'coordinates' in self.data:
                no_h_list = UniMolV2Feature(**params).transform_raw(self.data['atoms'], self.data['coordinates'])
            else:
                smiles_list = self.data["smiles"]                  
                no_h_list = UniMolV2Feature(**params).transform(smiles_list)

        self.data['unimol_input'] = no_h_list

    def _init_split(self, **params):

        self.split_method = params.get('split_method','5fold_random')
        kfold, method = int(self.split_method.split('fold')[0]), self.split_method.split('_')[-1]    # Nfold_xxxx
        self.kfold = params.get('kfold', kfold)
        self.method = params.get('split', method)
        self.split_seed = params.get('split_seed', 42)
        self.data['kfold'] = self.kfold
        if not self.is_train:
            return
        self.splitter = Splitter(self.method, self.kfold, seed=self.split_seed)
        split_nfolds = self.splitter.split(**self.data)
        if self.kfold == 1:
            logger.info(f"Kfold is 1, all data is used for training.")
        else:
            logger.info(f"Split method: {self.method}, fold: {self.kfold}")
        nfolds = np.zeros(len(split_nfolds[0][0])+len(split_nfolds[0][1]), dtype=int)
        for enu, (tr_idx, te_idx) in enumerate(split_nfolds):
            nfolds[te_idx] = enu
        self.data['split_nfolds'] = split_nfolds
        return split_nfolds