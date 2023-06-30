# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function
from ast import Load

import logging
import copy
import os
import pandas as pd
import numpy as np
import csv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader as TorchDataLoader
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from transformers.optimization import get_linear_schedule_with_warmup
from ..utils import Metrics
from ..utils import logger
from .split import Splitter
from tqdm import tqdm

import time
import sys


class Trainer(object):
    def __init__(self, save_path=None, **params):
        self.save_path = save_path
        self.task = params.get('task', None)

        if self.task != 'repr':
            self.metrics_str = params['metrics']
            self.metrics = Metrics(self.task, self.metrics_str)
        self._init_trainer(**params)

    def _init_trainer(self, **params):
        ### init common params ###
        self.split_method = params.get('split_method', '5fold_random')
        self.split_seed = params.get('split_seed', 42)
        self.seed = params.get('seed', 42)
        self.set_seed(self.seed)
        self.splitter = Splitter(self.split_method, self.split_seed)
        self.logger_level = int(params.get('logger_level', 1))
        ### init NN trainer params ###
        self.learning_rate = float(params.get('learning_rate', 1e-4))
        self.batch_size = params.get('batch_size', 32)
        self.max_epochs = params.get('epochs', 50)
        self.warmup_ratio = params.get('warmup_ratio', 0.1)
        self.patience = params.get('patience', 10)
        self.max_norm = params.get('max_norm', 1.0)
        self.cuda = params.get('cuda', False)
        self.amp = params.get('amp', False)
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() and self.cuda else "cpu")
        self.scaler = torch.cuda.amp.GradScaler(
        ) if self.device.type == 'cuda' and self.amp == True else None

    def decorate_batch(self, batch, feature_name=None):
        return self.decorate_torch_batch(batch)

    def decorate_graph_batch(self, batch):
        net_input, net_target = {'net_input': batch.to(
            self.device)}, batch.y.to(self.device)
        if self.task in ['classification', 'multiclass', 'multilabel_classification']:
            net_target = net_target.long()
        else:
            net_target = net_target.float()
        return net_input, net_target

    def decorate_torch_batch(self, batch):
        """function used to decorate batch data
        """
        net_input, net_target = batch
        if isinstance(net_input, dict):
            net_input, net_target = {
                k: v.to(self.device) for k, v in net_input.items()}, net_target.to(self.device)
        else:
            net_input, net_target = {'net_input': net_input.to(
                self.device)}, net_target.to(self.device)
        if self.task == 'repr':
            net_target = None
        elif self.task in ['classification', 'multiclass', 'multilabel_classification']:
            net_target = net_target.long()
        else:
            net_target = net_target.float()
        return net_input, net_target

    def fit_predict(self, model, train_dataset, valid_dataset, loss_func, activation_fn, dump_dir, fold, target_scaler, feature_name=None):
        model = model.to(self.device)
        train_dataloader = NNDataLoader(
            feature_name=feature_name,
            dataset=train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=model.batch_collate_fn,
            drop_last=True,
        )
        # remove last batch, bs=1 can not work on batchnorm1d
        min_val_loss = float("inf")
        max_score = float("-inf")
        wait = 0
        ### init optimizer ###
        num_training_steps = len(train_dataloader) * self.max_epochs
        num_warmup_steps = int(num_training_steps * self.warmup_ratio)
        optimizer = Adam(model.parameters(), lr=self.learning_rate, eps=1e-6)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

        for epoch in range(self.max_epochs):
            model = model.train()
            # Progress Bar
            start_time = time.time()
            batch_bar = tqdm(total=len(train_dataloader), dynamic_ncols=True,
                             leave=False, position=0, desc='Train', ncols=5)
            trn_loss = []
            for i, batch in enumerate(train_dataloader):
                net_input, net_target = self.decorate_batch(
                    batch, feature_name)
                optimizer.zero_grad()  # Zero gradients
                if self.scaler and self.device.type == 'cuda':
                    with torch.cuda.amp.autocast():
                        outputs = model(**net_input)
                        loss = loss_func(outputs, net_target)
                else:
                    with torch.set_grad_enabled(True):
                        outputs = model(**net_input)
                        loss = loss_func(outputs, net_target)
                trn_loss.append(float(loss.data))
                # tqdm lets you add some details so you can monitor training as you train.
                batch_bar.set_postfix(
                    Epoch="Epoch {}/{}".format(epoch+1, self.max_epochs),
                    loss="{:.04f}".format(float(sum(trn_loss) / (i + 1))),
                    lr="{:.04f}".format(float(optimizer.param_groups[0]['lr'])))
                if self.scaler and self.device.type == 'cuda':
                    # This is a replacement for loss.backward()
                    self.scaler.scale(loss).backward()
                    # unscale the gradients of optimizer's assigned params in-place
                    self.scaler.unscale_(optimizer)
                    # Clip the norm of the gradients to max_norm.
                    clip_grad_norm_(model.parameters(), self.max_norm)
                    # This is a replacement for optimizer.step()
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    clip_grad_norm_(model.parameters(), self.max_norm)
                    optimizer.step()
                scheduler.step()
                batch_bar.update()

            batch_bar.close()
            total_trn_loss = np.mean(trn_loss)

            y_preds, val_loss, metric_score = self.predict(
                model, valid_dataset, loss_func, activation_fn, dump_dir, fold, target_scaler, epoch, load_model=False, feature_name=feature_name)
            end_time = time.time()
            total_val_loss = np.mean(val_loss)
            _score = list(metric_score.values())[0]
            _metric = list(metric_score.keys())[0]
            message = 'Epoch [{}/{}] train_loss: {:.4f}, val_loss: {:.4f}, val_{}: {:.4f}, lr: {:.6f}, ' \
                '{:.1f}s'.format(epoch+1, self.max_epochs,
                                 total_trn_loss, total_val_loss,
                                 _metric, _score,
                                 optimizer.param_groups[0]['lr'],
                                 (end_time - start_time))
            logger.info(message)
            is_early_stop, min_val_loss, wait, max_score = self._early_stop_choice(
                wait, total_val_loss, min_val_loss, metric_score, max_score, model, dump_dir, fold, self.patience, epoch)
            if is_early_stop:
                break

        y_preds, _, _ = self.predict(model, valid_dataset, loss_func, activation_fn,
                                     dump_dir, fold, target_scaler, epoch, load_model=True, feature_name=feature_name)
        return y_preds

    def _early_stop_choice(self, wait, loss, min_loss, metric_score, max_score, model, dump_dir, fold, patience, epoch):
        ### hpyerparameter need to tune if you want to use early stop, currently find use loss is suitable in benchmark test. ###
        if not isinstance(self.metrics_str, str) or self.metrics_str in ['loss', 'none', '']:
            # loss 作为早停 直接用trainer里面的早停函数
            is_early_stop, min_val_loss, wait = self._judge_early_stop_loss(
                wait, loss, min_loss, model, dump_dir, fold, patience, epoch)
        else:
            # 到metric进行判断
            is_early_stop, min_val_loss, wait, max_score = self.metrics._early_stop_choice(
                wait, min_loss, metric_score, max_score, model, dump_dir, fold, patience, epoch)
        return is_early_stop, min_val_loss, wait, max_score

    def _judge_early_stop_loss(self, wait, loss, min_loss, model, dump_dir, fold, patience, epoch):
        is_early_stop = False
        if loss <= min_loss:
            min_loss = loss
            wait = 0
            info = {'model_state_dict': model.state_dict()}
            os.makedirs(dump_dir, exist_ok=True)
            torch.save(info, os.path.join(dump_dir, f'model_{fold}.pth'))
        elif loss >= min_loss:
            wait += 1
            if wait == self.patience:
                logger.warning(f'Early stopping at epoch: {epoch+1}')
                is_early_stop = True
        return is_early_stop, min_loss, wait

    def predict(self, model, dataset, loss_func, activation_fn, dump_dir, fold, target_scaler=None, epoch=1, load_model=False, feature_name=None):
        model = model.to(self.device)
        if load_model == True:
            load_model_path = os.path.join(dump_dir, f'model_{fold}.pth')
            model_dict = torch.load(load_model_path, map_location=self.device)[
                "model_state_dict"]
            model.load_state_dict(model_dict)
            logger.info("load model success!")
        dataloader = NNDataLoader(
            feature_name=feature_name,
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=model.batch_collate_fn,
        )
        model = model.eval()
        batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True,
                         position=0, leave=False, desc='val', ncols=5)
        val_loss = []
        y_preds = []
        y_truths = []
        for i, batch in enumerate(dataloader):
            net_input, net_target = self.decorate_batch(batch, feature_name)
            # Get model outputs
            with torch.no_grad():
                outputs = model(**net_input)
                if not load_model:
                    loss = loss_func(outputs, net_target)
                    val_loss.append(float(loss.data))
            y_preds.append(activation_fn(outputs).cpu().numpy())
            y_truths.append(net_target.detach().cpu().numpy())
            if not load_model:
                batch_bar.set_postfix(
                    Epoch="Epoch {}/{}".format(epoch+1, self.max_epochs),
                    loss="{:.04f}".format(float(np.sum(val_loss) / (i + 1))))

            batch_bar.update()
        y_preds = np.concatenate(y_preds)
        y_truths = np.concatenate(y_truths)

        try:
            label_cnt = model.output_dim
        except:
            label_cnt = None

        if target_scaler is not None:
            inverse_y_preds = target_scaler.inverse_transform(y_preds)
            inverse_y_truths = target_scaler.inverse_transform(y_truths)
            metric_score = self.metrics.cal_metric(
                inverse_y_truths, inverse_y_preds, label_cnt=label_cnt) if not load_model else None
        else:
            metric_score = self.metrics.cal_metric(
                y_truths, y_preds, label_cnt=label_cnt) if not load_model else None
        batch_bar.close()
        return y_preds, val_loss, metric_score

    def inference(self, model, dataset, feature_name=None, return_repr=True):
        model = model.to(self.device)
        dataloader = NNDataLoader(
            feature_name=feature_name,
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=model.batch_collate_fn,
        )
        model = model.eval()
        batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True,
                         position=0, leave=False, desc='val', ncols=5)
        repr_dict = {"cls_repr": [], "atomic_reprs": []}
        for i, batch in enumerate(dataloader):
            net_input, _ = self.decorate_batch(batch, feature_name)
            with torch.no_grad():
                outputs = model(return_repr=return_repr, **net_input)
                assert isinstance(outputs, dict)
                for key, value in outputs.items():
                    if isinstance(value, list):
                        value_list = [item.cpu().numpy() for item in value]
                        repr_dict[key].extend(value_list)
                    else:
                        repr_dict[key].extend([value.cpu().numpy()])
        repr_dict["cls_repr"] = np.concatenate(repr_dict["cls_repr"]).tolist()
        return repr_dict

    def set_seed(self, seed):
        """function used to set a random seed
        Arguments:
            seed {int} -- seed number, will set to torch and numpy
        """
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)


def NNDataLoader(feature_name=None, dataset=None, batch_size=None, shuffle=False, collate_fn=None, drop_last=False):

    dataloader = TorchDataLoader(dataset=dataset,
                                 batch_size=batch_size,
                                 shuffle=shuffle,
                                 collate_fn=collate_fn,
                                 drop_last=drop_last)
    return dataloader