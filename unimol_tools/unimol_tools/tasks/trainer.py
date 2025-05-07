# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function

import os
import time
from functools import partial

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

# from transformers.optimization import get_linear_schedule_with_warmup
from ..utils import Metrics, logger


class Trainer(object):
    """A :class:`Trainer` class is responsible for initializing the model, and managing its training, validation, and testing phases."""

    def __init__(self, save_path=None, **params):
        """
        :param save_path: Path for saving the training outputs. Defaults to None.
        :param params: Additional parameters for training.
        """
        self.save_path = save_path
        self.task = params.get('task', None)

        if self.task != 'repr':
            self.metrics_str = params['metrics']
            self.metrics = Metrics(self.task, self.metrics_str)
        self._init_trainer(**params)

    def _init_trainer(self, **params):
        """
        Initializing the trainer class to train model.

        :param params: Containing training arguments.
        """
        ### init common params ###
        self.split_method = params.get('split_method', '5fold_random')
        self.split_seed = params.get('split_seed', 42)
        self.seed = params.get('seed', 42)
        self.set_seed(self.seed)
        self.logger_level = int(params.get('logger_level', 1))
        ### init NN trainer params ###
        self.learning_rate = float(params.get('learning_rate', 1e-4))
        self.batch_size = params.get('batch_size', 32)
        self.max_epochs = params.get('epochs', 50)
        self.warmup_ratio = params.get('warmup_ratio', 0.1)
        self.patience = params.get('patience', 10)
        self.max_norm = params.get('max_norm', 1.0)
        self._init_dist(params)

    def _init_dist(self, params):
        self.cuda = params.get('use_cuda', True)
        self.amp = params.get('use_amp', True)
        self.ddp = params.get('use_ddp', False)
        self.gpu = params.get('use_gpu', "all")

        if torch.cuda.is_available() and self.cuda:
            if self.amp:
                self.scaler = torch.cuda.amp.GradScaler()
            else:
                self.scaler = None
            self.device = torch.device("cuda")
            world_size = torch.cuda.device_count()
            logger.info(f"Number of GPUs available: {world_size}")
            if self.gpu is not None:
                if self.gpu == "all":
                    gpu = ",".join(str(i) for i in range(world_size))
                else:
                    gpu = self.gpu
            else:
                gpu = "0"
            gpu_count = len(str(gpu).split(","))

            if world_size > 1 and self.ddp and gpu_count > 1:
                gpu = str(gpu).replace(" ", "")
                os.environ['MASTER_ADDR'] = os.getenv('MASTER_ADDR', 'localhost')
                os.environ['MASTER_PORT'] = os.getenv('MASTER_PORT', '19198')
                os.environ['CUDA_VISIBLE_DEVICES'] = gpu
                os.environ['WORLD_SIZE'] = str(world_size)
                logger.info(f"Using DistributedDataParallel for multi-GPU. GPUs: {gpu}")
            else:
                self.device = torch.device("cuda:0")
                self.ddp = False
                logger.info("Using single GPU.")
        else:
            self.scaler = None
            self.device = torch.device("cpu")
            self.ddp = False
            logger.info("Using CPU.")
        return

    def init_ddp(self, local_rank):
        torch.cuda.set_device(local_rank)
        os.environ['RANK'] = str(local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')
        self.device = torch.device("cuda", local_rank)

    def decorate_batch(self, batch, feature_name=None):
        """
        Prepares a batch of data for processing by the model. This method is a wrapper that
        delegates to a specific batch decoration method based on the data type.

        :param batch: The batch of data to be processed.
        :param feature_name: (str, optional) Name of the feature used in batch decoration. Defaults to None.

        :return: The decorated batch ready for processing by the model.
        """
        return self.decorate_torch_batch(batch)

    def decorate_graph_batch(self, batch):
        """
        Prepares a graph-based batch of data for processing by the model. Specifically handles
        graph-based data structures.

        :param batch: The batch of graph-based data to be processed.

        :return: A tuple of (net_input, net_target) for model processing.
        """
        net_input, net_target = {'net_input': batch.to(self.device)}, batch.y.to(
            self.device
        )
        if self.task in ['classification', 'multiclass', 'multilabel_classification']:
            net_target = net_target.long()
        else:
            net_target = net_target.float()
        return net_input, net_target

    def decorate_torch_batch(self, batch):
        """
        Prepares a standard PyTorch batch of data for processing by the model. Handles tensor-based data structures.

        :param batch: The batch of tensor-based data to be processed.

        :return: A tuple of (net_input, net_target) for model processing.
        """
        net_input, net_target = batch
        if isinstance(net_input, dict):
            net_input, net_target = {
                k: v.to(self.device) for k, v in net_input.items()
            }, net_target.to(self.device)
        else:
            net_input, net_target = {
                'net_input': net_input.to(self.device)
            }, net_target.to(self.device)
        if self.task == 'repr':
            net_target = None
        elif self.task in ['classification', 'multiclass', 'multilabel_classification']:
            net_target = net_target.long()
        else:
            net_target = net_target.float()
        return net_input, net_target

    def fit_predict(
        self,
        model,
        train_dataset,
        valid_dataset,
        loss_func,
        activation_fn,
        dump_dir,
        fold,
        target_scaler,
        feature_name=None,
    ):
        """
        Trains the model on the given dataset.

        :param local_rank: (int) The local rank of the current process.
        :param args: Additional arguments for training.
        """
        if torch.cuda.device_count() and self.ddp:
            with mp.Manager() as manager:
                shared_queue = manager.Queue()
                mp.spawn(
                    self.fit_predict_with_ddp,
                    args=(
                        shared_queue,
                        model,
                        train_dataset,
                        valid_dataset,
                        loss_func,
                        activation_fn,
                        dump_dir,
                        fold,
                        target_scaler,
                        feature_name,
                    ),
                    nprocs=torch.cuda.device_count(),
                )

                try:
                    y_preds = shared_queue.get(timeout=1)
                    # print(f"Main function returned: {y_preds}")
                except:
                    print("No return value received from main function.")
            return y_preds
        else:
            return self.fit_predict_wo_ddp(
                model,
                train_dataset,
                valid_dataset,
                loss_func,
                activation_fn,
                dump_dir,
                fold,
                target_scaler,
                feature_name,
            )

    def fit_predict_wo_ddp(
        self,
        model,
        train_dataset,
        valid_dataset,
        loss_func,
        activation_fn,
        dump_dir,
        fold,
        target_scaler,
        feature_name=None,
    ):
        """
        Trains the model on the given training dataset and evaluates it on the validation dataset.

        :param model: The model to be trained and evaluated.
        :param train_dataset: Dataset used for training the model.
        :param valid_dataset: Dataset used for validating the model.
        :param loss_func: The loss function used during training.
        :param activation_fn: The activation function applied to the model's output.
        :param dump_dir: Directory where the best model state is saved.
        :param fold: The fold number in a cross-validation setting.
        :param target_scaler: Scaler used for scaling the target variable.
        :param feature_name: (optional) Name of the feature used in data loading. Defaults to None.

        :return: Predictions made by the model on the validation dataset.
        """
        model = model.to(self.device)
        train_dataloader = NNDataLoader(
            feature_name=feature_name,
            dataset=train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=model.batch_collate_fn,
            distributed=False,
            drop_last=True,
        )
        optimizer, scheduler = self._initialize_optimizer_scheduler(
            model, train_dataloader
        )
        early_stopper = EarlyStopper(
            self.patience, dump_dir, fold, self.metrics, self.metrics_str
        )

        for epoch in range(self.max_epochs):
            total_trn_loss = self._train_one_epoch(
                model,
                train_dataloader,
                optimizer,
                scheduler,
                loss_func,
                feature_name,
                epoch,
            )

            y_preds, val_loss, metric_score = self.predict(
                model,
                valid_dataset,
                loss_func,
                activation_fn,
                dump_dir,
                fold,
                target_scaler,
                epoch,
                load_model=False,
                feature_name=feature_name,
            )

            self._log_epoch_results(
                epoch, total_trn_loss, np.mean(val_loss), metric_score, optimizer
            )

            if early_stopper.early_stop_choice(
                model, epoch, np.mean(val_loss), metric_score
            ):
                break

        y_preds, _, _ = self.predict(
            model,
            valid_dataset,
            loss_func,
            activation_fn,
            dump_dir,
            fold,
            target_scaler,
            epoch,
            load_model=True,
            feature_name=feature_name,
        )
        return y_preds

    def fit_predict_with_ddp(
        self,
        local_rank,
        shared_queue,
        model,
        train_dataset,
        valid_dataset,
        loss_func,
        activation_fn,
        dump_dir,
        fold,
        target_scaler,
        feature_name=None,
    ):
        """
        Trains the model on the given training dataset and evaluates it on the validation dataset.

        :param model: The model to be trained and evaluated.
        :param train_dataset: Dataset used for training the model.
        :param valid_dataset: Dataset used for validating the model.
        :param loss_func: The loss function used during training.
        :param activation_fn: The activation function applied to the model's output.
        :param dump_dir: Directory where the best model state is saved.
        :param fold: The fold number in a cross-validation setting.
        :param target_scaler: Scaler used for scaling the target variable.
        :param feature_name: (optional) Name of the feature used in data loading. Defaults to None.

        :return: Predictions made by the model on the validation dataset.
        """
        self.init_ddp(local_rank)
        model = model.to(local_rank)
        model = DistributedDataParallel(
            model, device_ids=[local_rank], find_unused_parameters=True
        )
        train_dataloader = NNDataLoader(
            feature_name=feature_name,
            dataset=train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=model.module.batch_collate_fn,
            distributed=True,
            drop_last=True,
        )
        optimizer, scheduler = self._initialize_optimizer_scheduler(
            model, train_dataloader
        )
        early_stopper = EarlyStopper(
            self.patience, dump_dir, fold, self.metrics, self.metrics_str
        )
        for epoch in range(self.max_epochs):
            total_trn_loss = self._train_one_epoch(
                model,
                train_dataloader,
                optimizer,
                scheduler,
                loss_func,
                feature_name,
                epoch,
            )

            y_preds, val_loss, metric_score = self.predict(
                model,
                valid_dataset,
                loss_func,
                activation_fn,
                dump_dir,
                fold,
                target_scaler,
                epoch,
                load_model=False,
                feature_name=feature_name,
            )

            total_trn_loss = self.reduce_array(total_trn_loss)
            total_val_loss = self.reduce_array(np.mean(val_loss))

            if local_rank == 0:
                # self._log_epoch_results(
                #     epoch, total_trn_loss, total_val_loss, metric_score, optimizer
                # ) # TODO: this will generate redundant log files.
                is_early_stop = early_stopper.early_stop_choice(
                    model, epoch, total_val_loss, metric_score
                )
                if is_early_stop:
                    stop_flag = torch.tensor(1, device=self.device)
                else:
                    stop_flag = torch.tensor(0, device=self.device)
            else:
                stop_flag = torch.tensor(0, device=self.device)

            dist.broadcast(stop_flag, src=0)
            if stop_flag.item() == 1:
                break

            dist.barrier()

        y_preds, _, _ = self.predict(
            model,
            valid_dataset,
            loss_func,
            activation_fn,
            dump_dir,
            fold,
            target_scaler,
            epoch,
            load_model=False,
            feature_name=feature_name,
        )
        y_preds = self.gather_predictions(y_preds, len_valid_dataset=len(valid_dataset))
        dist.destroy_process_group()
        if local_rank == 0:
            shared_queue.put(y_preds)
        return y_preds

    def _initialize_optimizer_scheduler(self, model, train_dataloader):
        num_training_steps = len(train_dataloader) * self.max_epochs
        num_warmup_steps = int(num_training_steps * self.warmup_ratio)
        optimizer = Adam(model.parameters(), lr=self.learning_rate, eps=1e-6)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps, num_training_steps
        )
        return optimizer, scheduler

    def _train_one_epoch(
        self,
        model,
        train_dataloader,
        optimizer,
        scheduler,
        loss_func,
        feature_name,
        epoch,
    ):
        model.train()
        trn_loss = []
        batch_bar = tqdm(
            total=len(train_dataloader),
            dynamic_ncols=True,
            leave=False,
            position=0 if not self.ddp else dist.get_rank(),
            desc='Train' if not self.ddp else f'Train Rank:{dist.get_rank()}',
            ncols=5,
        )
        for i, batch in enumerate(train_dataloader):
            net_input, net_target = self.decorate_batch(batch, feature_name)
            optimizer.zero_grad()
            loss = self._compute_loss(model, net_input, net_target, loss_func)
            trn_loss.append(float(loss.data))
            self._backward_and_step(optimizer, loss, model)
            scheduler.step()
            batch_bar.set_postfix(
                Epoch=f"Epoch {epoch+1}/{self.max_epochs}",
                loss=f"{float(sum(trn_loss) / (i + 1)):.04f}",
                lr=f"{float(optimizer.param_groups[0]['lr']):.04f}",
            )
            batch_bar.update()
        batch_bar.close()
        return np.mean(trn_loss)

    def _compute_loss(self, model, net_input, net_target, loss_func):
        if self.scaler and self.device.type == 'cuda':
            with torch.cuda.amp.autocast():
                outputs = model(**net_input)
                loss = loss_func(outputs, net_target)
        else:
            with torch.set_grad_enabled(True):
                outputs = model(**net_input)
                loss = loss_func(outputs, net_target)
        return loss

    def _backward_and_step(self, optimizer, loss, model):
        if self.scaler and self.device.type == 'cuda':
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), self.max_norm)
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            loss.backward()
            clip_grad_norm_(model.parameters(), self.max_norm)
            optimizer.step()

    def _log_epoch_results(
        self, epoch, total_trn_loss, total_val_loss, metric_score, optimizer
    ):
        _score = list(metric_score.values())[0]
        _metric = list(metric_score.keys())[0]
        message = f'Epoch [{epoch+1}/{self.max_epochs}] train_loss: {total_trn_loss:.4f}, val_loss: {total_val_loss:.4f}, val_{_metric}: {_score:.4f}, lr: {optimizer.param_groups[0]["lr"]:.6f}'
        logger.info(message)
        return False

    def reduce_array(self, array):
        tensor = torch.tensor(array, device=self.device)
        rt = tensor.clone()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        rt /= dist.get_world_size()
        return rt.item()

    def gather_predictions(self, y_preds, len_valid_dataset):
        y_preds_tensor = torch.tensor(y_preds, device=self.device)
        gathered_y_preds = [
            torch.zeros_like(y_preds_tensor) for _ in range(dist.get_world_size())
        ]
        dist.all_gather(gathered_y_preds, y_preds_tensor)
        gathered_y_preds = torch.stack(gathered_y_preds, dim=1).view(-1, y_preds_tensor.size(1))

        if len(gathered_y_preds) != len_valid_dataset:
            gathered_y_preds = gathered_y_preds[
                :len_valid_dataset
            ]  # remove padding when using DDP
        return gathered_y_preds.cpu().numpy()

    def predict(
        self,
        model,
        dataset,
        loss_func,
        activation_fn,
        dump_dir,
        fold,
        target_scaler=None,
        epoch=1,
        load_model=False,
        feature_name=None,
    ):
        """
        Executes the prediction on a given dataset using the specified model.

        :param model: The model to be used for predictions.
        :param dataset: The dataset to perform predictions on.
        :param loss_func: The loss function used during training.
        :param activation_fn: The activation function applied to the model's output.
        :param dump_dir: Directory where the model state is saved.
        :param fold: The fold number in cross-validation.
        :param target_scaler: (optional) Scaler to inverse transform the model's output. Defaults to None.
        :param epoch: (int) The current epoch number. Defaults to 1.
        :param load_model: (bool) Whether to load the model from a saved state. Defaults to False.
        :param feature_name: (str, optional) Name of the feature for data processing. Defaults to None.

        :return: A tuple (y_preds, val_loss, metric_score), where y_preds are the predicted outputs,
                 val_loss is the validation loss, and metric_score is the calculated metric score.
        """
        model = self._prepare_model_for_prediction(model, dump_dir, fold, load_model)
        if isinstance(model, DistributedDataParallel):
            batch_collate_fn = model.module.batch_collate_fn
        else:
            batch_collate_fn = model.batch_collate_fn
        dataloader = NNDataLoader(
            feature_name=feature_name,
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=batch_collate_fn,
            distributed=self.ddp,
            valid_mode=True,
        )
        y_preds, val_loss, y_truths = self._perform_prediction(
            model, dataloader, loss_func, activation_fn, load_model, epoch, feature_name
        )

        metric_score = self._calculate_metrics(
            y_preds, y_truths, target_scaler, model, load_model
        )
        return y_preds, val_loss, metric_score

    def _prepare_model_for_prediction(self, model, dump_dir, fold, load_model):
        model = model.to(self.device)
        if load_model:
            load_model_path = os.path.join(dump_dir, f'model_{fold}.pth')
            model.load_pretrained_weights(load_model_path, strict=True)
            logger.info("load model success!")
        return model

    def _perform_prediction(
        self,
        model,
        dataloader,
        loss_func,
        activation_fn,
        load_model,
        epoch,
        feature_name,
    ):
        model = model.eval()
        batch_bar = tqdm(
            total=len(dataloader),
            dynamic_ncols=True,
            position=0,
            leave=False,
            desc='val',
            ncols=5,
        )
        val_loss = []
        y_preds = []
        y_truths = []
        for i, batch in enumerate(dataloader):
            net_input, net_target = self.decorate_batch(batch, feature_name)
            with torch.no_grad():
                outputs = model(**net_input)
                if not load_model:
                    loss = loss_func(outputs, net_target)
                    val_loss.append(float(loss.data))
            y_preds.append(activation_fn(outputs).cpu().numpy())
            y_truths.append(net_target.detach().cpu().numpy())
            if not load_model:
                batch_bar.set_postfix(
                    Epoch="Epoch {}/{}".format(epoch + 1, self.max_epochs),
                    loss="{:.04f}".format(float(np.sum(val_loss) / (i + 1))),
                )
            batch_bar.update()
        batch_bar.close()
        y_preds = np.concatenate(y_preds)
        y_truths = np.concatenate(y_truths)
        return y_preds, val_loss, y_truths

    def _calculate_metrics(self, y_preds, y_truths, target_scaler, model, load_model):
        try:
            label_cnt = model.output_dim
        except:
            label_cnt = None
        if target_scaler is not None:
            inverse_y_preds = target_scaler.inverse_transform(y_preds)
            inverse_y_truths = target_scaler.inverse_transform(y_truths)
            metric_score = (
                self.metrics.cal_metric(
                    inverse_y_truths, inverse_y_preds, label_cnt=label_cnt
                )
                if not load_model
                else None
            )
        else:
            metric_score = (
                self.metrics.cal_metric(y_truths, y_preds, label_cnt=label_cnt)
                if not load_model
                else None
            )
        return metric_score

    def inference(
        self,
        model,
        dataset,
        return_repr=False,
        return_atomic_reprs=False,
        feature_name=None,
    ):
        """
        Runs inference on the given dataset using the provided model. This method can return
        various representations based on the model's output.

        :param model: The neural network model to be used for inference.
        :param dataset: The dataset on which inference is to be performed.
        :param return_repr: (bool, optional) If True, returns class-level representations. Defaults to False.
        :param return_atomic_reprs: (bool, optional) If True, returns atomic-level representations. Defaults to False.
        :param feature_name: (str, optional) Name of the feature used for data loading. Defaults to None.

        :return: A dictionary containing different types of representations based on the model's output and the
                 specified parameters. This can include class-level representations, atomic coordinates,
                 atomic representations, and atomic symbols.
        """
        if torch.cuda.device_count() and self.ddp:
            with mp.Manager() as manager:
                shared_queue = manager.Queue()
                mp.spawn(
                    self.inference_with_ddp,
                    args=(
                        shared_queue,
                        model,
                        dataset,
                        return_repr,
                        return_atomic_reprs,
                        feature_name,
                    ),
                    nprocs=torch.cuda.device_count(),
                )
                try:
                    repr_dict = shared_queue.get(timeout=1)
                except:
                    print("No return value received from main function.")
                return repr_dict
        else:
            return self.inference_without_ddp(
                model, dataset, return_repr, return_atomic_reprs, feature_name
            )

    def inference_with_ddp(
        self,
        local_rank,
        shared_queue,
        model,
        dataset,
        return_repr=False,
        return_atomic_reprs=False,
        feature_name=None,
    ):
        """
        Runs inference on the given dataset using the provided model with DistributedDataParallel (DDP).

        :param local_rank: The local rank of the current process.
        :param shared_queue: A shared queue to store the inference results.
        :param model: The neural network model to be used for inference.
        :param dataset: The dataset on which inference is to be performed.
        :param return_repr: (bool, optional) If True, returns class-level representations. Defaults to False.
        :param return_atomic_reprs: (bool, optional) If True, returns atomic-level representations. Defaults to False.
        :param feature_name: (str, optional) Name of the feature used for data loading. Defaults to None.

        :return: A dictionary containing different types of representations based on the model's output and the
                 specified parameters. This can include class-level representations, atomic coordinates,
                 atomic representations, and atomic symbols.
        """
        self.init_ddp(local_rank)
        model = model.to(local_rank)
        model = DistributedDataParallel(
            model, device_ids=[local_rank], find_unused_parameters=True
        )
        dataloader = NNDataLoader(
            feature_name=feature_name,
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=model.module.batch_collate_fn,
            distributed=True,
        )
        model = model.eval()
        repr_dict = {
            "cls_repr": [],
            "atomic_coords": [],
            "atomic_reprs": [],
            "atomic_symbol": [],
        }
        for batch in tqdm(dataloader):
            net_input, _ = self.decorate_batch(batch, feature_name)
            with torch.no_grad():
                outputs = model(
                    **net_input,
                    return_repr=return_repr,
                    return_atomic_reprs=return_atomic_reprs,
                )

                assert isinstance(outputs, dict)
                repr_dict["cls_repr"].extend(
                    item.cpu().numpy() for item in outputs["cls_repr"]
                )
                if return_atomic_reprs:
                    repr_dict["atomic_symbol"].extend(outputs["atomic_symbol"])
                    repr_dict['atomic_coords'].extend(
                        item.cpu().numpy() for item in outputs['atomic_coords']
                    )
                    repr_dict['atomic_reprs'].extend(
                        item.cpu().numpy() for item in outputs['atomic_reprs']
                    )
        gathered_list = [{} for _ in range(dist.get_world_size())]
        dist.gather_object(repr_dict, gathered_list if local_rank == 0 else None, dst=0)
        dist.destroy_process_group()
        if local_rank == 0:
            merged_repr_dict = {"cls_repr": []}
            if return_atomic_reprs:
                merged_repr_dict.update(
                    {"atomic_symbol": [], "atomic_coords": [], "atomic_reprs": []}
                )

            for rd in gathered_list:
                merged_repr_dict["cls_repr"].extend(rd["cls_repr"])
                if return_atomic_reprs:
                    merged_repr_dict["atomic_symbol"].extend(rd.get("atomic_symbol", []))
                    merged_repr_dict["atomic_coords"].extend(rd.get("atomic_coords", []))
                    merged_repr_dict["atomic_reprs"].extend(rd.get("atomic_reprs", []))

            merged_repr_dict["cls_repr"] = merged_repr_dict["cls_repr"][: len(dataset)]
            if return_atomic_reprs:
                for key in ["atomic_symbol", "atomic_coords", "atomic_reprs"]:
                    merged_repr_dict[key] = merged_repr_dict[key][: len(dataset)]

            shared_queue.put(merged_repr_dict)
        return repr_dict

    def inference_without_ddp(
        self,
        model,
        dataset,
        return_repr=False,
        return_atomic_reprs=False,
        feature_name=None,
    ):
        """
        Runs inference on the given dataset using the provided model without DistributedDataParallel (DDP).

        :param model: The neural network model to be used for inference.
        :param dataset: The dataset on which inference is to be performed.
        :param return_repr: (bool, optional) If True, returns class-level representations. Defaults to False.
        :param return_atomic_reprs: (bool, optional) If True, returns atomic-level representations. Defaults to False.
        :param feature_name: (str, optional) Name of the feature used for data loading. Defaults to None.

        :return: A dictionary containing different types of representations based on the model's output and the
                 specified parameters. This can include class-level representations, atomic coordinates,
                 atomic representations, and atomic symbols.
        """
        model = model.to(self.device)
        dataloader = NNDataLoader(
            feature_name=feature_name,
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=model.batch_collate_fn,
            distributed=False,
        )
        model = model.eval()
        repr_dict = {
            "cls_repr": [],
            "atomic_coords": [],
            "atomic_reprs": [],
            "atomic_symbol": [],
        }
        for batch in tqdm(dataloader):
            net_input, _ = self.decorate_batch(batch, feature_name)
            with torch.no_grad():
                outputs = model(
                    **net_input,
                    return_repr=return_repr,
                    return_atomic_reprs=return_atomic_reprs,
                )
                assert isinstance(outputs, dict)
                repr_dict["cls_repr"].extend(
                    item.cpu().numpy() for item in outputs["cls_repr"]
                )
                if return_atomic_reprs:
                    repr_dict["atomic_symbol"].extend(outputs["atomic_symbol"])
                    repr_dict['atomic_coords'].extend(
                        item.cpu().numpy() for item in outputs['atomic_coords']
                    )
                    repr_dict['atomic_reprs'].extend(
                        item.cpu().numpy() for item in outputs['atomic_reprs']
                    )

        return repr_dict

    def set_seed(self, seed):
        """
        Sets a random seed for torch and numpy to ensure reproducibility.
        :param seed: (int) The seed number to be set.
        """
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)


class EarlyStopper:
    def __init__(self, patience, dump_dir, fold, metrics, metrics_str):
        """
        Initializes the EarlyStopper class.

        :param patience: The number of epochs to wait for an improvement before stopping.
        :param dump_dir: Directory to save the model state.
        :param fold: The current fold number in a cross-validation setting.
        """
        self.patience = patience
        self.dump_dir = dump_dir
        self.fold = fold
        self.metrics = metrics
        self.metrics_str = metrics_str
        self.wait = 0
        self.min_loss = float("inf")
        self.max_loss = float("-inf")
        self.is_early_stop = False

    def early_stop_choice(self, model, epoch, loss, metric_score=None):
        """
        Determines if early stopping criteria are met, based on either loss improvement or custom metric score.

        :param model: The model being trained.
        :param epoch: The current epoch number.
        :param loss: The current loss value.
        :param metric_score: The current metric score.

        :return: A boolean indicating whether early stopping should occur.
        """
        if not isinstance(self.metrics_str, str) or self.metrics_str in [
            'loss',
            'none',
            '',
        ]:
            return self._judge_early_stop_loss(loss, model, epoch)
        else:
            is_early_stop, min_score, wait, max_score = self.metrics._early_stop_choice(
                self.wait,
                self.min_loss,
                metric_score,
                self.max_loss,
                model,
                self.dump_dir,
                self.fold,
                self.patience,
                epoch,
            )
            self.min_loss = min_score
            self.max_loss = max_score
            self.wait = wait
            self.is_early_stop = is_early_stop
            return self.is_early_stop

    def _judge_early_stop_loss(self, loss, model, epoch):
        """
        Determines whether early stopping should be triggered based on the loss comparison.

        :param loss: The current loss value of the model.
        :param model: The neural network model being trained.
        :param epoch: The current epoch number.

        :return: A boolean indicating whether early stopping should occur.
        """
        if loss <= self.min_loss:
            self.min_loss = loss
            self.wait = 0
            if isinstance(model, DistributedDataParallel):
                model = model.module
            info = {'model_state_dict': model.state_dict()}
            os.makedirs(self.dump_dir, exist_ok=True)
            torch.save(info, os.path.join(self.dump_dir, f'model_{self.fold}.pth'))
        else:
            self.wait += 1
            if self.wait >= self.patience:
                logger.warning(f'Early stopping at epoch: {epoch+1}')
                self.is_early_stop = True
        return self.is_early_stop


def NNDataLoader(
    feature_name=None,
    dataset=None,
    batch_size=None,
    shuffle=False,
    collate_fn=None,
    drop_last=False,
    distributed=False,
    valid_mode=False,
):
    """
    Creates a DataLoader for neural network training or inference. This
    function is a wrapper around the standard PyTorch DataLoader, allowing
    for custom feature handling and additional configuration.

    :param feature_name: (str, optional) Name of the feature used for data loading.
                            This can be used to specify a particular type of data processing. Defaults to None.
    :param dataset: (Dataset, optional) The dataset from which to load the data. Defaults to None.
    :param batch_size: (int, optional) Number of samples per batch to load. Defaults to None.
    :param shuffle: (bool, optional) Whether to shuffle the data at every epoch. Defaults to False.
    :param collate_fn: (callable, optional) Merges a list of samples to form a mini-batch. Defaults to None.
    :param drop_last: (bool, optional) Set to True to drop the last incomplete batch. Defaults to False.
    :param distributed: (bool, optional) Set to True to enable distributed data loading. Defaults to False.

    :return: DataLoader configured according to the provided parameters.
    """

    if distributed:
        sampler = DistributedSampler(dataset, shuffle=shuffle)
        g = get_ddp_generator()
    else:
        sampler = None
        g = None

    if valid_mode:
        g = None

    dataloader = TorchDataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        drop_last=drop_last,
        # num_workers=4,
        pin_memory=True,
        sampler=sampler,
        generator=g,
    )
    return dataloader


def get_ddp_generator(seed=3407):
    local_rank = dist.get_rank()
    g = torch.Generator()
    g.manual_seed(seed + local_rank)
    return g


# source from https://github.com/huggingface/transformers/blob/main/src/transformers/optimization.py#L108C1-L132C54
def _get_linear_schedule_with_warmup_lr_lambda(
    current_step: int, *, num_warmup_steps: int, num_training_steps: int
):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    return max(
        0.0,
        float(num_training_steps - current_step)
        / float(max(1, num_training_steps - num_warmup_steps)),
    )


def get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps, num_training_steps, last_epoch=-1
):
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0, after
    a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    lr_lambda = partial(
        _get_linear_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)
