# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function

import numpy as np
import joblib
import os

from .data import DataHub
from .models import NNModel
from .tasks import Trainer
from .utils import YamlHandler
from .utils import logger


class MolPredict(object):
    """A :class:`MolPredict` class is responsible for interface of predicting process of molecular data."""
    def __init__(self, load_model=None):
        """ 
        Initialize a :class:`MolPredict` class.

        :param load_model: str, default=None, path of model to load.
        """
        if not load_model:
            raise ValueError("load_model is empty")
        self.load_model = load_model
        config_path = os.path.join(load_model, 'config.yaml')
        self.config = YamlHandler(config_path).read_yaml()
        self.config.target_cols = self.config.target_cols.split(',')
        self.task = self.config.task
        self.target_cols = self.config.target_cols

    def predict(self, data, save_path=None, metrics='none'):
        """ 
        Predict molecular data.

        :param data: str or pandas.DataFrame or dict of atoms and coordinates, input data for prediction. \
            - str: path of csv file.
            - pandas.DataFrame: dataframe of data.
            - dict: dict of atoms and coordinates, e.g. {'atoms': ['C', 'C', 'C'], 'coordinates': [[0, 0, 0], [0, 0, 1], [0, 0, 2]]}
        :param save_path: str, default=None, path to save predict result.
        :param metrics: str, default='none', metrics to evaluate model performance.
        
            currently support: 

            - classification: auc, auprc, log_loss, acc, f1_score, mcc, precision, recall, cohen_kappa. 

            - regression: mse, pearsonr, spearmanr, mse, r2.

            - multiclass: log_loss, acc.

            - multilabel_classification: auc, auprc, log_loss, acc, mcc.

            - multilabel_regression: mae, mse, r2.

        :return y_pred: numpy.ndarray, predict result.
        """
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
        """
        Save predict result to csv file.

        :param data: pandas.DataFrame, predict result.
        :param dir: str, directory to save predict result.
        :param prefix: str, prefix of predict result file name.
        """
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
