# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function

import logging
import copy
import os
import pandas as pd
import numpy as np
import argparse
import joblib

from .data import DataHub
from .models import NNModel
from .tasks import Trainer
from .utils import YamlHandler
from .utils import logger


class MolPredict(object):
    def __init__(self, load_model=None):
        if not load_model:
            raise ValueError("load_model is empty")
        self.load_model = load_model
        config_path = os.path.join(load_model, 'config.yaml')
        self.config = YamlHandler(config_path).read_yaml()
        self.config.target_cols = self.config.target_cols.split(',')
        self.task = self.config.task
        self.target_cols = self.config.target_cols

    def predict(self, data, save_path=None, metrics='none'):
        self.save_path = save_path
        if not metrics or metrics != 'none':
            self.config.metrics = metrics
        ## load test data
        self.datahub = DataHub(data = data, is_train = False, save_path=self.load_model, **self.config)
        self.trainer = Trainer(save_path=self.load_model, **self.config)
        self.model = NNModel(self.datahub.data, self.trainer, **self.config)
        self.model.evaluate(self.trainer, self.load_model)

        y_pred = self.model.cv['test_pred']
        scalar = self.datahub.data['target_scaler']
        if scalar is not None:
            y_pred = scalar.inverse_transform(y_pred)

        df = self.datahub.data['raw_data'].copy()
        predict_cols = ['predict_' + col for col in self.target_cols]
        if self.task == 'multiclass' and self.config.multiclass_cnt is not None:
            prob_cols = ['prob_' + str(i) for i in range(self.config.multiclass_cnt)]
            df[prob_cols] = y_pred
            df[predict_cols] = np.argmax(y_pred, axis=1).reshape(-1, 1)
        elif self.task in ['classification', 'multilabel_classification']:
            threshold = joblib.load(open(os.path.join(self.load_model, 'threshold.dat'), "rb"))
            prob_cols = ['prob_' + col for col in self.target_cols]
            df[prob_cols] = y_pred
            df[predict_cols] = (y_pred > threshold).astype(int)
        else:
            prob_cols = predict_cols
            df[predict_cols] = y_pred
        if self.save_path:
            os.makedirs(self.save_path, exist_ok=True)
        if not (df[self.target_cols] == -1.0).all().all():
            metrics = self.trainer.metrics.cal_metric(df[self.target_cols].values, df[prob_cols].values)
            logger.info("final predict metrics score: \n{}".format(metrics))
            if self.save_path:
                joblib.dump(metrics, os.path.join(self.save_path, 'test_metric.result'))
        else:
            df.drop(self.target_cols, axis=1, inplace=True)
        if self.save_path:
            prefix = data.split('/')[-1].split('.')[0] if isinstance(data, str) else 'test'
            self.save_predict(df, self.save_path, prefix)
            logger.info("pipeline finish!")

        return y_pred
    
    def save_predict(self, data, dir, prefix):
        run_id = 0
        if not os.path.exists(dir):
            os.makedirs(dir)
        else:
            folders = [x for x in os.listdir(dir)]
            while prefix + f'.predict.{run_id}' + '.csv' in folders:
                run_id += 1
        name = prefix + f'.predict.{run_id}' + '.csv'
        path = os.path.join(dir, name)
        data.to_csv(path)
        logger.info("save predict result to {}".format(path))