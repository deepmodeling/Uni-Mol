# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function

import os
import pandas as pd
import numpy as np
from rdkit import Chem
from ..utils import logger
import pathlib
from rdkit.Chem.Scaffolds import MurckoScaffold

class MolDataReader(object):
    '''A class to read Mol Data.'''
    def read_data(self, data=None, is_train=True, **params):
        # TO DO 
        # 1. add anomaly detection & outlier removal.
        # 2. add support for other file format.
        # 3. add support for multi tasks.

        """
        Reads and preprocesses molecular data from various input formats for model training or prediction.
        Parsing target columns
        1. if target_cols is not None, use target_cols as target columns.
        2. if target_cols is None, use all columns with prefix 'target_col_prefix' as target columns.
        3. use given target_cols as target columns placeholder with value -1.0 for predict
        
        :param data: The input molecular data. Can be a file path (str), a dictionary, or a list of SMILES strings.
        :param is_train: (bool) A flag indicating if the operation is for training. Determines data processing steps.
        :param params: A dictionary of additional parameters for data processing.

        :return: A dictionary containing processed data and related information for model consumption.
        :raises ValueError: If the input data type is not supported or if any SMILES string is invalid (when strict).
        """
        task = params.get('task', None)
        target_cols = params.get('target_cols', None)
        smiles_col = params.get('smiles_col', 'SMILES')
        target_col_prefix = params.get('target_col_prefix', 'TARGET')
        anomaly_clean = params.get('anomaly_clean', False)
        smi_strict = params.get('smi_strict', False)
        split_group_col = params.get('split_group_col', 'scaffold')

        if isinstance(data, str):
            # load from file
            self.data_path = data
            data = pd.read_csv(self.data_path)
        elif isinstance(data, dict):
            # load from dict
            if 'target' in data:
                label = np.array(data['target'])
                if len(label.shape)==1 or label.shape[1] == 1:
                    data[target_col_prefix] = label.reshape(-1)
                else:
                    for i in range(label.shape[1]):
                        data[target_col_prefix + str(i)] = label[:,i]

            _ = data.pop('target', None)
            data = pd.DataFrame(data).rename(columns={smiles_col: 'SMILES'})
        
        elif isinstance(data, list) or isinstance(data, np.ndarray):
            # load from smiles list
            data = pd.DataFrame(data, columns=['SMILES'])
        else:
            raise ValueError('Unknown data type: {}'.format(type(data)))
        
        #### parsing target columns
        #### 1. if target_cols is not None, use target_cols as target columns.
        #### 2. if target_cols is None, use all columns with prefix 'target_col_prefix' as target columns.
        #### 3. use given target_cols as target columns placeholder with value -1.0 for predict
        if task == 'repr':
            # placeholder for repr task
            targets = None
            target_cols = None
            num_classes = None
            multiclass_cnt = None
        else:
            if target_cols is None:
                target_cols = [item for item in data.columns if item.startswith(target_col_prefix)]
            else: 
                for col in target_cols:
                    if col not in data.columns:
                        data[target_cols] = -1.0
                        break

            if is_train and anomaly_clean:
                data = self.anomaly_clean(data, task, target_cols)
            
            if is_train and task == 'multiclass':
                multiclass_cnt = int(data[target_cols].max() + 1)

            targets = data[target_cols].values.tolist()
            num_classes = len(target_cols)
        
        dd = {
            'raw_data': data,
            'raw_target': targets,
            'num_classes': num_classes,
            'target_cols': target_cols,
            'multiclass_cnt': multiclass_cnt if task == 'multiclass' and is_train else None
        }
        if smiles_col in data.columns:
            mask = data[smiles_col].apply(lambda smi: self.check_smiles(smi, is_train, smi_strict))
            data = data[mask]  
            dd['smiles'] = data[smiles_col].tolist()
            dd['scaffolds'] = data[smiles_col].map(self.smi2scaffold).tolist()
        else:
            dd['smiles'] = None
            dd['scaffolds'] = None

        if split_group_col in data.columns:
            dd['group'] = data[split_group_col].tolist()
        elif split_group_col == 'scaffold':
            dd['group'] = dd['scaffolds']
        else:
            dd['group'] = None

        if 'atoms' in data.columns and 'coordinates' in data.columns:
            dd['atoms'] = data['atoms'].tolist()
            dd['coordinates'] = data['coordinates'].tolist()

        return dd

    def check_smiles(self,smi, is_train, smi_strict):
        """
        Validates a SMILES string and decides whether it should be included based on training mode and strictness.

        :param smi: (str) The SMILES string to check.
        :param is_train: (bool) Indicates if this check is happening during training.
        :param smi_strict: (bool) If true, invalid SMILES strings raise an error, otherwise they're logged and skipped.

        :return: (bool) True if the SMILES string is valid, False otherwise.
        :raises ValueError: If the SMILES string is invalid and strict mode is on.
        """
        if Chem.MolFromSmiles(smi) is None:
            if is_train and not smi_strict:
                logger.info(f'Illegal SMILES clean: {smi}')
                return False
            else:
                raise ValueError(f'SMILES rule is illegal: {smi}')
        return True    
    
    def smi2scaffold(self,smi):
        """
        Converts a SMILES string to its corresponding scaffold.

        :param smi: (str) The SMILES string to convert.

        :return: (str) The scaffold of the SMILES string, or the original SMILES if conversion fails.
        """
        try:
            return MurckoScaffold.MurckoScaffoldSmiles(smiles=smi, includeChirality=True)
        except:
            return smi
    
    def anomaly_clean(self, data, task, target_cols):
        """
        Performs anomaly cleaning on the data based on the specified task.

        :param data: (DataFrame) The dataset to be cleaned.
        :param task: (str) The type of task which determines the cleaning strategy.
        :param target_cols: (list) The list of target columns to consider for cleaning.

        :return: (DataFrame) The cleaned dataset.
        :raises ValueError: If the provided task is not recognized.
        """
        if task in ['classification', 'multiclass', 'multilabel_classification', 'multilabel_regression']:
            return data
        if task == 'regression':
            return self.anomaly_clean_regression(data, target_cols)
        else:
            raise ValueError('Unknown task: {}'.format(task))
    
    def anomaly_clean_regression(self, data, target_cols):
        """
        Performs anomaly cleaning specifically for regression tasks using a 3-sigma threshold.

        :param data: (DataFrame) The dataset to be cleaned.
        :param target_cols: (list) The list of target columns to consider for cleaning.

        :return: (DataFrame) The cleaned dataset after applying the 3-sigma rule.
        """
        sz = data.shape[0]
        target_col = target_cols[0]
        _mean, _std = data[target_col].mean(), data[target_col].std()
        data = data[(data[target_col] > _mean - 3 * _std) & (data[target_col] < _mean + 3 * _std)]
        logger.info('Anomaly clean with 3 sigma threshold: {} -> {}'.format(sz, data.shape[0]))
        return data
