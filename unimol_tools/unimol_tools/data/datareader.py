# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function

import logging
import copy
import os
import pandas as pd
import re
from pymatgen.core import Structure
from .conformer import inner_coords, coords2unimol_mof
from unicore.data import Dictionary
import numpy as np
import csv
from typing import List, Optional
from rdkit import Chem
from ..utils import logger
from ..config import MODEL_CONFIG
import pathlib
from rdkit.Chem.Scaffolds import MurckoScaffold
WEIGHT_DIR = os.path.join(pathlib.Path(__file__).resolve().parents[1], 'weights')

class MolDataReader(object):

    def read_data(self, data=None, is_train=True, **params):
        # TO DO 
        # 1. add anomaly detection & outlier removal.
        # 2. add support for other file format.
        # 3. add support for multi tasks.
        task = params.get('task', None)
        target_cols = params.get('target_cols', None)
        smiles_col = params.get('smiles_col', 'SMILES')
        target_col_prefix = params.get('target_col_prefix', 'TARGET')
        anomaly_clean = params.get('anomaly_clean', False)
        smi_strict = params.get('smi_strict', False)

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
            data = pd.DataFrame(data).rename(columns={smiles_col: 'SMILES'})
            if 'target' in data:
                data = data.drop(columns=['target'])
        elif isinstance(data, list):
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
            dd['smiles'] = np.arange(data.shape[0]).tolist()
            dd['scaffolds'] = np.arange(data.shape[0]).tolist()

        if 'atoms' in data.columns and 'coordinates' in data.columns:
            dd['atoms'] = data['atoms'].tolist()
            dd['coordinates'] = data['coordinates'].tolist()

        return dd

    def check_smiles(self,smi, is_train, smi_strict):
        if Chem.MolFromSmiles(smi) is None:
            if is_train and not smi_strict:
                logger.info(f'Illegal SMILES clean: {smi}')
                return False
            else:
                raise ValueError(f'SMILES rule is illegal: {smi}')
        return True    
    
    def smi2scaffold(self,smi):
        try:
            return MurckoScaffold.MurckoScaffoldSmiles(smiles=smi, includeChirality=True)
        except:
            return smi
    
    def anomaly_clean(self, data, task, target_cols):
        if task in ['classification', 'multiclass', 'multilabel_classification', 'multilabel_regression']:
            return data
        if task == 'regression':
            return self.anomaly_clean_regression(data, target_cols)
        else:
            raise ValueError('Unknown task: {}'.format(task))
    
    def anomaly_clean_regression(self, data, target_cols):
        sz = data.shape[0]
        target_col = target_cols[0]
        _mean, _std = data[target_col].mean(), data[target_col].std()
        data = data[(data[target_col] > _mean - 3 * _std) & (data[target_col] < _mean + 3 * _std)]
        logger.info('Anomaly clean with 3 sigma threshold: {} -> {}'.format(sz, data.shape[0]))
        return data
    

class MOFReader(object):
    def __init__(self):
        self.gas_list = ['CH4','CO2','Ar','Kr','Xe','O2','He','N2','H2']
        self.GAS2ID = {
            "UNK":0,
            "CH4":1, 
            "CO2":2, 
            "Ar":3, 
            "Kr":4, 
            "Xe":5, 
            "O2":6, 
            "He":7, 
            "N2":8, 
            "H2":9,
        }
        self.GAS2ATTR = {
            "CH4":[0.295589,0.165132,0.251511019,-0.61518,0.026952,0.25887781],
            "CO2":[1.475242,1.475921,1.620478155,0.086439,1.976795,1.69928074],
            "Ar":[-0.11632,0.294448,0.1914686,-0.01667,-0.07999,-0.1631478],
            "Kr":[0.48802,0.602454,0.215485568,1.084671,0.415991,0.39885917],
            "Xe":[1.324657,0.751519,0.233498293,2.276323,1.12122,1.18462811],
            "O2":[-0.08095,0.37909,0.335570404,-0.61626,-0.5363,-0.1130181],
            "He":[-1.66617,-1.88746,-2.15618995,-0.9173,-1.36413,-1.6042445],
            "N2":[-0.37636,-0.3968,0.41962979,-0.31495,-0.40022,-0.3355659],
            "H2":[-1.34371,-1.3843,-1.11145188,-0.96708,-1.16031,-1.3256695],
        }
        self.dict_name = MODEL_CONFIG['dict']['mof']
        self.dictionary = Dictionary.load(os.path.join(WEIGHT_DIR, self.dict_name))
        self.dictionary.add_symbol("[MASK]", is_special=True)
        self.max_atoms = 512

    def cif_parser(self, cif_path, primitive=False):
        """
        Parser for single cif file
        """
        s = Structure.from_file(cif_path, primitive=primitive)
        id = cif_path.split('/')[-1][:-4]
        lattice = s.lattice
        abc = lattice.abc # lattice vectors
        angles = lattice.angles # lattice angles
        volume = lattice.volume # lattice volume
        lattice_matrix = lattice.matrix # lattice 3x3 matrix

        df = s.as_dataframe()
        atoms = df['Species'].astype(str).map(lambda x: re.sub("\d+", "", x)).tolist()
        coordinates = df[['x', 'y', 'z']].values.astype(np.float32)
        abc_coordinates = df[['a', 'b', 'c']].values.astype(np.float32)
        assert len(atoms) == coordinates.shape[0]
        assert len(atoms) == abc_coordinates.shape[0]

        return {'ID':id, 
                'atoms':atoms, 
                'coordinates':coordinates, 
                'abc':abc, 
                'angles':angles, 
                'volume':volume, 
                'lattice_matrix':lattice_matrix, 
                'abc_coordinates':abc_coordinates,
                }

    def gas_parser(self, gas='CH4'):
        assert gas in self.gas_list, "{} is not in list, current we support: {}".format(gas, '-'.join(self.gas_list))
        gas_id = self.GAS2ID.get(gas, 0)
        gas_attr = self.GAS2ATTR.get(gas, np.zeros(6))

        return {'gas_id': gas_id, 'gas_attr': gas_attr}

    def read_with_gas(self, cif_path, gas):
        dd = self.cif_parser(cif_path)
        atoms, coordinates = inner_coords(dd['atoms'], dd['coordinates'])
        dd = coords2unimol_mof(atoms, coordinates, self.dictionary, max_atoms=self.max_atoms)
        dd.update(self.gas_parser(gas))

        return dd
