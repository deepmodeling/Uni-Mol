# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function

import logging
import copy
import os
import torch
import torch.nn as nn
from torch.nn import functional as F
import joblib
from torch.utils.data import Dataset
import numpy as np
from ..utils import logger
from .unimol import UniMolModel
from .loss import GHMC_Loss, FocalLossWithLogits, myCrossEntropyLoss


NNMODEL_REGISTER = {
    'unimolv1': UniMolModel,
}

LOSS_RREGISTER = {
    'classification': myCrossEntropyLoss,
    'multiclass': myCrossEntropyLoss,
    'regression': nn.MSELoss(),
    'multilabel_classification': {
        'bce': nn.BCEWithLogitsLoss(),
        'ghm': GHMC_Loss(bins=10, alpha=0.5),
        'focal': FocalLossWithLogits,
    },
    'multilabel_regression': nn.MSELoss(),
}
ACTIVATION_FN = {
    # predict prob shape should be (N, K), especially for binary classification, K equals to 1.
    'classification': lambda x: F.softmax(x, dim=-1)[:, 1:],
    # softmax is used for multiclass classification
    'multiclass': lambda x: F.softmax(x, dim=-1),
    'regression': lambda x: x,
    # sigmoid is used for multilabel classification
    'multilabel_classification': lambda x: F.sigmoid(x),
    # no activation function is used for multilabel regression
    'multilabel_regression': lambda x: x,
}
OUTPUT_DIM = {
    'classification': 2,
    'regression': 1,
}


class NNModel(object):
    def __init__(self, data, trainer, **params):
        self.data = data
        self.num_classes = self.data['num_classes']
        self.target_scaler = self.data['target_scaler']
        self.features = data['unimol_input']
        self.model_name = params.get('model_name', 'unimolv1')
        self.data_type = params.get('data_type', 'molecule')
        self.loss_key = params.get('loss_key', None)
        self.trainer = trainer
        self.splitter = self.trainer.splitter
        self.model_params = params.copy()
        self.task = params['task']
        if self.task in OUTPUT_DIM:
            self.model_params['output_dim'] = OUTPUT_DIM[self.task]
        elif self.task == 'multiclass':
            self.model_params['output_dim'] = self.data['multiclass_cnt']
        else:
            self.model_params['output_dim'] = self.num_classes
        self.model_params['device'] = self.trainer.device
        self.cv = dict()
        self.metrics = self.trainer.metrics
        if self.task == 'multilabel_classification':
            if self.loss_key is None:
                self.loss_key = 'focal'
            self.loss_func = LOSS_RREGISTER[self.task][self.loss_key]
        else:
            self.loss_func = LOSS_RREGISTER[self.task]
        self.activation_fn = ACTIVATION_FN[self.task]
        self.save_path = self.trainer.save_path
        self.trainer.set_seed(self.trainer.seed)
        self.model = self._init_model(**self.model_params)

    def _init_model(self, model_name, **params):
        if model_name in NNMODEL_REGISTER:
            model = NNMODEL_REGISTER[model_name](**params)
        else:
            raise ValueError('Unknown model: {}'.format(self.model_name))
        return model

    def collect_data(self, X, y, idx):
        assert isinstance(y, np.ndarray), 'y must be numpy array'
        if isinstance(X, np.ndarray):
            return torch.from_numpy(X[idx]).float(), torch.from_numpy(y[idx])
        elif isinstance(X, list):
            return {k: v[idx] for k, v in X.items()}, torch.from_numpy(y[idx])
        else:
            raise ValueError('X must be numpy array or dict')

    def run(self):
        logger.info("start training Uni-Mol:{}".format(self.model_name))
        X = np.asarray(self.features)
        y = np.asarray(self.data['target'])
        scaffold = np.asarray(self.data['scaffolds'])
        if self.task == 'classification':
            y_pred = np.zeros_like(
                y.reshape(y.shape[0], self.num_classes)).astype(float)
        else:
            y_pred = np.zeros((y.shape[0], self.model_params['output_dim']))
        for fold, (tr_idx, te_idx) in enumerate(self.splitter.split(X, y, scaffold)):
            X_train, y_train = X[tr_idx], y[tr_idx]
            X_valid, y_valid = X[te_idx], y[te_idx]
            traindataset = NNDataset(X_train, y_train)
            validdataset = NNDataset(X_valid, y_valid)
            if fold > 0:
                # need to initalize model for next fold training
                self.model = self._init_model(**self.model_params)
            _y_pred = self.trainer.fit_predict(
                self.model, traindataset, validdataset, self.loss_func, self.activation_fn, self.save_path, fold, self.target_scaler)
            y_pred[te_idx] = _y_pred

            if 'multiclass_cnt' in self.data:
                label_cnt = self.data['multiclass_cnt']
            else:
                label_cnt = None

            logger.info("fold {0}, result {1}".format(
                fold,
                self.metrics.cal_metric(
                        self.data['target_scaler'].inverse_transform(y_valid),
                        self.data['target_scaler'].inverse_transform(_y_pred),
                        label_cnt=label_cnt
                        )
            )
            )

        self.cv['pred'] = y_pred
        self.cv['metric'] = self.metrics.cal_metric(self.data['target_scaler'].inverse_transform(
            y), self.data['target_scaler'].inverse_transform(self.cv['pred']))
        self.dump(self.cv['pred'], self.save_path, 'cv.data')
        self.dump(self.cv['metric'], self.save_path, 'metric.result')
        logger.info("Uni-Mol metrics score: \n{}".format(self.cv['metric']))
        logger.info("Uni-Mol & Metric result saved!")

    def dump(self, data, dir, name):
        path = os.path.join(dir, name)
        if not os.path.exists(dir):
            os.makedirs(dir)
        joblib.dump(data, path)

    def evaluate(self, trainer=None,  checkpoints_path=None):
        logger.info("start predict NNModel:{}".format(self.model_name))
        testdataset = NNDataset(self.features, np.asarray(self.data['target']))
        for fold in range(self.splitter.n_splits):
            model_path = os.path.join(checkpoints_path, f'model_{fold}.pth')
            self.model.load_state_dict(torch.load(
                model_path, map_location=self.trainer.device)['model_state_dict'])
            _y_pred, _, __ = trainer.predict(self.model, testdataset, self.loss_func, self.activation_fn,
                                             self.save_path, fold, self.target_scaler, epoch=1, load_model=True)
            if fold == 0:
                y_pred = np.zeros_like(_y_pred)
            y_pred += _y_pred
        y_pred /= self.splitter.n_splits
        self.cv['test_pred'] = y_pred

    def count_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


def NNDataset(data, label=None):
    return TorchDataset(data, label)


class TorchDataset(Dataset):
    def __init__(self, data, label=None):
        self.data = data
        self.label = label if label is not None else np.zeros((len(data), 1))

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]

    def __len__(self):
        return len(self.data)
