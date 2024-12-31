# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from unicore import metrics
from unicore.losses import UnicoreLoss, register_loss


@register_loss("finetune_mse")
class FinetuneMSELoss(UnicoreLoss):
    def __init__(self, task):
        super().__init__(task)

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(
            **sample["net_input"],
            features_only=True,
            classification_head_name=self.args.classification_head_name,
        )
        reg_output = net_output[0]
        loss = self.compute_loss(model, reg_output, sample, reduce=reduce)
        sample_size = sample["target"]["finetune_target"].size(0)
        if not self.training:
            if self.task.mean and self.task.std:
                targets_mean = torch.tensor(self.task.mean, device=reg_output.device)
                targets_std = torch.tensor(self.task.std, device=reg_output.device)
                reg_output = reg_output * targets_std + targets_mean
            logging_output = {
                "loss": loss.data,
                "predict": reg_output.view(-1, self.args.num_classes).data,
                "target": sample["target"]["finetune_target"]
                .view(-1, self.args.num_classes)
                .data,
                "smi_name": sample["smi_name"],
                "sample_size": sample_size,
                "num_task": self.args.num_classes,
                "conf_size": self.args.conf_size,
                "bsz": sample["target"]["finetune_target"].size(0),
            }
        else:
            logging_output = {
                "loss": loss.data,
                "sample_size": sample_size,
                "bsz": sample["target"]["finetune_target"].size(0),
            }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        predicts = net_output.view(-1, self.args.num_classes).float()
        targets = (
            sample["target"]["finetune_target"].view(-1, self.args.num_classes).float()
        )
        if self.task.mean and self.task.std:
            targets_mean = torch.tensor(self.task.mean, device=targets.device)
            targets_std = torch.tensor(self.task.std, device=targets.device)
            targets = (targets - targets_mean) / targets_std
        loss = F.mse_loss(
            predicts,
            targets,
            reduction="sum" if reduce else "none",
        )
        return loss

    @staticmethod
    def reduce_metrics(logging_outputs, split="valid") -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        if "valid" in split or "test" in split:
            predicts = torch.cat([log.get("predict") for log in logging_outputs], dim=0)
            if predicts.size(-1) == 1:
                # single label regression task, add aggregate acc and loss score
                targets = torch.cat(
                    [log.get("target", 0) for log in logging_outputs], dim=0
                )
                smi_list = [
                    item for log in logging_outputs for item in log.get("smi_name")
                ]
                df = pd.DataFrame(
                    {
                        "predict": predicts.view(-1).cpu(),
                        "target": targets.view(-1).cpu(),
                        "smi": smi_list,
                    }
                )
                mae = np.abs(df["predict"] - df["target"]).mean()
                mse = ((df["predict"] - df["target"]) ** 2).mean()
                df = df.groupby("smi").mean()
                agg_mae = np.abs(df["predict"] - df["target"]).mean()
                agg_mse = ((df["predict"] - df["target"]) ** 2).mean()

                metrics.log_scalar(f"{split}_mae", mae, sample_size, round=3)
                metrics.log_scalar(f"{split}_mse", mse, sample_size, round=3)
                metrics.log_scalar(f"{split}_agg_mae", agg_mae, sample_size, round=3)
                metrics.log_scalar(f"{split}_agg_mse", agg_mse, sample_size, round=3)
                metrics.log_scalar(
                    f"{split}_agg_rmse", np.sqrt(agg_mse), sample_size, round=4
                )

    @staticmethod
    def logging_outputs_can_be_summed(is_train) -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return is_train


@register_loss("finetune_mae")
class FinetuneMAELoss(FinetuneMSELoss):
    def __init__(self, task):
        super().__init__(task)

    def compute_loss(self, model, net_output, sample, reduce=True):
        predicts = net_output.view(-1, self.args.num_classes).float()
        targets = (
            sample["target"]["finetune_target"].view(-1, self.args.num_classes).float()
        )
        if self.task.mean and self.task.std:
            targets_mean = torch.tensor(self.task.mean, device=targets.device)
            targets_std = torch.tensor(self.task.std, device=targets.device)
            targets = (targets - targets_mean) / targets_std
        loss = F.l1_loss(
            predicts,
            targets,
            reduction="sum" if reduce else "none",
        )
        return loss


@register_loss("finetune_smooth_mae")
class FinetuneSmoothMAELoss(FinetuneMSELoss):
    def __init__(self, task):
        super().__init__(task)

    def compute_loss(self, model, net_output, sample, reduce=True):
        predicts = net_output.view(-1, self.args.num_classes).float()
        targets = (
            sample["target"]["finetune_target"].view(-1, self.args.num_classes).float()
        )
        if self.task.mean and self.task.std:
            targets_mean = torch.tensor(self.task.mean, device=targets.device)
            targets_std = torch.tensor(self.task.std, device=targets.device)
            targets = (targets - targets_mean) / targets_std
        loss = F.smooth_l1_loss(
            predicts,
            targets,
            reduction="sum" if reduce else "none",
        )
        return loss

    @staticmethod
    def reduce_metrics(logging_outputs, split="valid") -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        if "valid" in split or "test" in split:
            num_task = logging_outputs[0].get("num_task", 0)
            conf_size = logging_outputs[0].get("conf_size", 0)
            y_true = (
                torch.cat([log.get("target", 0) for log in logging_outputs], dim=0)
                .view(-1, conf_size, num_task)
                .cpu()
                .numpy()
                .mean(axis=1)
            )
            y_pred = (
                torch.cat([log.get("predict") for log in logging_outputs], dim=0)
                .view(-1, conf_size, num_task)
                .cpu()
                .numpy()
                .mean(axis=1)
            )
            agg_mae = np.abs(y_pred - y_true).mean()
            metrics.log_scalar(f"{split}_agg_mae", agg_mae, sample_size, round=4)


@register_loss("finetune_mse_pocket")
class FinetuneMSEPocketLoss(FinetuneMSELoss):
    def __init__(self, task):
        super().__init__(task)

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(
            **sample["net_input"],
            features_only=True,
            classification_head_name=self.args.classification_head_name,
        )
        reg_output = net_output[0]
        loss = self.compute_loss(model, reg_output, sample, reduce=reduce)
        sample_size = sample["target"]["finetune_target"].size(0)
        if not self.training:
            if self.task.mean and self.task.std:
                targets_mean = torch.tensor(self.task.mean, device=reg_output.device)
                targets_std = torch.tensor(self.task.std, device=reg_output.device)
                reg_output = reg_output * targets_std + targets_mean
            logging_output = {
                "loss": loss.data,
                "predict": reg_output.view(-1, self.args.num_classes).data,
                "target": sample["target"]["finetune_target"]
                .view(-1, self.args.num_classes)
                .data,
                "sample_size": sample_size,
                "num_task": self.args.num_classes,
                "bsz": sample["target"]["finetune_target"].size(0),
            }
        else:
            logging_output = {
                "loss": loss.data,
                "sample_size": sample_size,
                "bsz": sample["target"]["finetune_target"].size(0),
            }
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs, split="valid") -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        if "valid" in split or "test" in split:
            predicts = torch.cat([log.get("predict") for log in logging_outputs], dim=0)
            if predicts.size(-1) == 1:
                # single label regression task
                targets = torch.cat(
                    [log.get("target", 0) for log in logging_outputs], dim=0
                )
                df = pd.DataFrame(
                    {
                        "predict": predicts.view(-1).cpu(),
                        "target": targets.view(-1).cpu(),
                    }
                )
                mse = ((df["predict"] - df["target"]) ** 2).mean()
                metrics.log_scalar(f"{split}_mse", mse, sample_size, round=3)
                metrics.log_scalar(f"{split}_rmse", np.sqrt(mse), sample_size, round=4)
