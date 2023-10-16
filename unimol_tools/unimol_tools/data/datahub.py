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
from collections import defaultdict
from .datareader import MolDataReader
from .datascaler import TargetScaler
from .conformer import ConformerGen
from ..utils import logger

class DataHub(object):
    def __init__(self, data=None, is_train=True, save_path=None, **params):
        self.data = data
        self.is_train = is_train
        self.save_path = save_path
        self.task = params.get('task', None) 
        self.target_cols = params.get('target_cols', None)
        self.multiclass_cnt = params.get('multiclass_cnt', None)
        self.ss_method = params.get('target_normalize', 'none')
        self._init_data(**params)
    
    def _init_data(self, **params):
        self.data = MolDataReader().read_data(self.data, self.is_train, **params)
        self.data['target_scaler'] = TargetScaler(self.ss_method, self.task, self.save_path)
        if self.task == 'regression': 
            target = np.array(self.data['raw_target']).reshape(-1,1).astype(np.float32)
            if self.is_train:
                self.data['target_scaler'].fit(target, self.save_path)
            self.data['target'] = self.data['target_scaler'].transform(target)
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
        elif self.task == 'multilabel_classification':
            target = np.array(self.data['raw_target']).reshape(-1,self.data['num_classes']).astype(np.int32)
            self.data['target'] = target
        elif self.task == 'repr':
            self.data['target'] = self.data['raw_target']
        else:
            raise ValueError('Unknown task: {}'.format(self.task))
        
        if 'atoms' in self.data and 'coordinates' in self.data:
            no_h_list = ConformerGen(**params).transform_raw(self.data['atoms'], self.data['coordinates'])
        else:
            smiles_list = self.data["smiles"]                  
            no_h_list = ConformerGen(**params).transform(smiles_list)

        self.data['unimol_input'] = no_h_list
