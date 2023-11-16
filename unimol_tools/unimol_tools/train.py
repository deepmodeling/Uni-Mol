# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function

import logging
import copy
import os
import argparse
import json
import numpy as np
import pandas as pd
import joblib
from .data import DataHub
from .models import NNModel
from .tasks import Trainer
from .utils import YamlHandler
from .utils import logger

class MolTrain(object):
    def __init__(self, 
                task='classification',
                data_type='molecule',
                epochs=10,
                learning_rate=1e-4,
                batch_size=16,
                early_stopping=5,
                metrics= "none",
                split='random',                   # random, scaffold, group, stratified
                split_group_col='scaffold',       # only active with group split
                kfold=5,
                save_path='./exp',
                remove_hs=False,
                smiles_col='SMILES',
                target_col_prefix='TARGET',
                target_anomaly_check="filter",
                smiles_check="filter",
                target_normalize="auto",
                max_norm=5.0,
                use_cuda=True,
                use_amp=True,
                **params,
                ):
        config_path = os.path.join(os.path.dirname(__file__), 'config/default.yaml')
        self.yamlhandler = YamlHandler(config_path)
        config = self.yamlhandler.read_yaml()
        config.task = task
        config.data_type = data_type
        config.epochs = epochs
        config.learning_rate = learning_rate
        config.batch_size = batch_size
        config.patience = early_stopping
        config.metrics = metrics
        config.split = split
        config.split_group_col = split_group_col
        config.kfold = kfold
        config.remove_hs = remove_hs
        config.smiles_col = smiles_col
        config.target_col_prefix = target_col_prefix
        config.anomaly_clean = target_anomaly_check in ['filter']
        config.smi_strict = smiles_check in ['filter']
        config.target_normalize = target_normalize
        config.max_norm = max_norm
        config.use_cuda = use_cuda
        config.use_amp = use_amp
        self.save_path = save_path
        self.config = config


    def fit(self, data):
        self.datahub = DataHub(data = data, is_train=True, save_path=self.save_path, **self.config)
        self.data = self.datahub.data
        self.update_and_save_config()
        self.trainer = Trainer(save_path=self.save_path, **self.config)
        self.model = NNModel(self.data, self.trainer, **self.config)
        self.model.run()
        scalar = self.data['target_scaler']
        y_pred = self.model.cv['pred']
        y_true = np.array(self.data['target'])
        metrics = self.trainer.metrics
        if scalar is not None:
            y_pred = scalar.inverse_transform(y_pred)
            y_true = scalar.inverse_transform(y_true)

        if self.config["task"] in ['classification', 'multilabel_classification']:
            threshold = metrics.calculate_classification_threshold(y_true, y_pred)
            joblib.dump(threshold, os.path.join(self.save_path, 'threshold.dat'))
        
        self.cv_pred = y_pred
        return

    def update_and_save_config(self):
        self.config['num_classes'] = self.data['num_classes']
        self.config['target_cols'] = ','.join(self.data['target_cols'])
        if self.config['task'] == 'multiclass':
            self.config['multiclass_cnt'] = self.data['multiclass_cnt']

        self.config['split_method'] = f"{self.config['kfold']}fold_{self.config['split']}"
        if self.save_path is not None:
            if not os.path.exists(self.save_path):
                logger.info('Create output directory: {}'.format(self.save_path))
                os.makedirs(self.save_path)
            else:
                logger.info('Output directory already exists: {}'.format(self.save_path))
                logger.info('Warning: Overwrite output directory: {}'.format(self.save_path))
            out_path = os.path.join(self.save_path, 'config.yaml')
            self.yamlhandler.write_yaml(data = self.config, out_file_path = out_path)
        return
