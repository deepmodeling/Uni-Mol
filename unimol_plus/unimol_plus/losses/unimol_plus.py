from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from unicore import metrics
from unicore.losses import UnicoreLoss, register_loss
from scipy.spatial.transform import Rotation as R
from typing import List, Callable, Any, Dict
import os


@register_loss("unimol_plus")
class UnimolPlusLoss(UnicoreLoss):
    """
    Implementation for the loss used in masked graph model (MGM) training.
    """

    def __init__(self, task):
        super().__init__(task)
        self.args = task.args

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """

        with torch.no_grad():
            sample_size = sample["batched_data"]["atom_feat"].shape[0]
            natoms = sample["batched_data"]["atom_feat"].shape[1]

        # add gaussian noise
        (
            graph_output,
            pos_pred,
            dist_pred,
        ) = model(**sample)
        targets = sample["batched_data"]["target"].float().view(-1)
        graph_output = graph_output.float().view(-1)
        per_data_loss = torch.nn.L1Loss(reduction="none")(graph_output.float(), targets)
        loss = per_data_loss.sum()
        per_data_pred = graph_output
        per_data_label = targets

        atom_mask = sample["batched_data"]["atom_mask"].float()
        pos_mask = atom_mask.unsqueeze(-1)
        pos_target = sample["batched_data"]["pos_target"].float() * pos_mask

        def get_pos_loss(pos_pred):
            pos_pred = pos_pred.float() * pos_mask
            center_loss = pos_pred.mean(dim=-2).square().sum()
            pos_loss = torch.nn.L1Loss(reduction="none")(
                pos_pred,
                pos_target,
            ).sum(dim=(-1, -2))
            pos_cnt = pos_mask.squeeze(-1).sum(dim=-1) + 1e-10
            pos_loss = (pos_loss / pos_cnt).sum()
            return pos_loss, center_loss

        (pos_loss, center_loss) = get_pos_loss(pos_pred)

        pair_mask = atom_mask.unsqueeze(-1) * atom_mask.unsqueeze(-2).float()
        dist_target = (pos_target.unsqueeze(-2) - pos_target.unsqueeze(-3)).norm(dim=-1)
        dist_target = dist_target * pair_mask
        dist_cnt = pair_mask.sum(dim=(-1, -2)) + 1e-10

        def get_dist_loss(dist_pred, return_sum=True):
            dist_pred = dist_pred.float() * pair_mask
            dist_loss = torch.nn.L1Loss(reduction="none")(
                dist_pred,
                dist_target,
            ).sum(dim=(-1, -2))
            if return_sum:
                return (dist_loss / dist_cnt).sum()
            else:
                return dist_loss / dist_cnt

        dist_loss = get_dist_loss(dist_pred)

        total_loss = (
            loss + dist_loss + self.args.pos_loss_weight * (pos_loss + center_loss)
        )
        logging_output = {
            "loss": loss.data,
            "dist_loss": dist_loss.data,
            "pos_loss": pos_loss.data,
            "center_loss": center_loss.data,
            "total_loss": total_loss.data,
            "sample_size": sample_size,
            "nsentences": sample_size,
            "ntokens": natoms,
            "bsz": sample_size,
            "n_atoms": natoms * sample_size,
        }
        if not torch.is_grad_enabled():
            logging_output["id"] = sample["batched_data"]["id"].cpu().numpy()
            logging_output["pred"] = per_data_pred.detach().cpu().numpy()
            logging_output["label"] = per_data_label.detach().cpu().numpy()
        logging_output["total_loss"] = total_loss.data
        return total_loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs, split="valid") -> None:
        """Aggregate logging outputs from data parallel training."""
        if split != "train":
            id = np.concatenate([log["id"] for log in logging_outputs])
            pred = np.concatenate([log["pred"] for log in logging_outputs])
            label = np.concatenate([log["label"] for log in logging_outputs])
            df = pd.DataFrame(
                {
                    "id": id,
                    "pred": pred,
                    "label": label,
                }
            )
            df_grouped = df.groupby(["id"])
            df_mean = df_grouped.agg("mean")
            df_median = df_grouped.agg("median")

            def get_mae_loss(df):
                return np.abs(df["pred"] - df["label"]).mean()

            metrics.log_scalar("loss_by_mean", get_mae_loss(df_mean), 1, round=6)
            metrics.log_scalar("loss_by_median", get_mae_loss(df_median), 1, round=6)

        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        n_atoms = sum(log.get("n_atoms", 0) for log in logging_outputs)
        for key in logging_outputs[0].keys():
            if "loss" in key or "metric" in key:
                total_loss_sum = sum(log.get(key, 0) for log in logging_outputs)
                metrics.log_scalar(
                    key, total_loss_sum / sample_size, sample_size, round=6
                )
        metrics.log_scalar("n_atoms", n_atoms / sample_size, sample_size, round=6)

    @staticmethod
    def logging_outputs_can_be_summed(is_train) -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return is_train
