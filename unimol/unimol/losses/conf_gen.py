# Copyright (c) DP Techonology, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
import numpy as np
from unicore import metrics
from unicore.losses import UnicoreLoss, register_loss
from scipy.spatial.transform import Rotation as R


@register_loss("mol_confG")
class MolConfGLoss(UnicoreLoss):
    def __init__(self, task):
        super().__init__(task)
        self.padding_idx = task.dictionary.pad()
        self.eos_idx = task.dictionary.eos()
        self.bos_idx = task.dictionary.bos()

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        distance_loss, coord_loss = self.compute_loss(
            model, net_output, sample, reduce=reduce
        )
        sample_size = sample["target"]["coord_target"].size(0)
        loss = (
            self.args.coord_loss * coord_loss + self.args.distance_loss * distance_loss
        )
        logging_output = {
            "loss": loss.data,
            "distance_loss": distance_loss.data,
            "coord_loss": coord_loss.data,
            "bsz": sample["target"]["coord_target"].size(0),
            "sample_size": 1,
            "coord_predict": net_output[-1].data,
            "coord_target": sample["target"]["coord_target"].data,
            "distance_predict": net_output[0].data,
        }
        if not self.training:
            logging_output["smi_name"] = sample["smi_name"]

        return loss, 1, logging_output

    # reaglin coord in coord loss
    def compute_loss(self, model, net_output, sample, reduce=True):
        distance_predict, coord_predict = net_output[0], net_output[-1]
        token_mask = sample["net_input"]["src_tokens"].ne(self.padding_idx)  # B,L
        token_mask &= sample["net_input"]["src_tokens"].ne(self.eos_idx)
        token_mask &= sample["net_input"]["src_tokens"].ne(self.bos_idx)
        distance_mask, coord_mask = calc_mask(token_mask)
        mean_coord = (coord_mask * coord_predict).sum(dim=1) / token_mask.sum(
            dim=1, keepdims=True
        )
        coord_predict = coord_predict - mean_coord.unsqueeze(dim=1)

        # distance loss
        distance_predict = distance_predict[distance_mask]
        distance_target = sample["target"]["distance_target"][distance_mask]
        distance_loss = F.l1_loss(
            distance_predict.float(),
            distance_target.float(),
            reduction="mean",
        )

        # coord loss
        coord_target = sample["target"]["coord_target"]  # B, L, 3
        new_coord_target = realign_coord(coord_predict, coord_target, token_mask)
        coord_predict = coord_predict[coord_mask]
        new_coord_target = new_coord_target[coord_mask]
        coord_loss = F.l1_loss(
            coord_predict.float(),
            new_coord_target.float(),
            reduction="mean",
        )

        return distance_loss, coord_loss

    @staticmethod
    def reduce_metrics(logging_outputs, split="valid") -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar("loss", loss_sum / sample_size, sample_size, round=5)
        distance_loss = sum(log.get("distance_loss", 0) for log in logging_outputs)
        if distance_loss > 0:
            metrics.log_scalar(
                "distance_loss", distance_loss / sample_size, sample_size, round=5
            )
        coord_loss = sum(log.get("coord_loss", 0) for log in logging_outputs)
        if coord_loss > 0:
            metrics.log_scalar(
                "coord_loss", coord_loss / sample_size, sample_size, round=5
            )

    @staticmethod
    def logging_outputs_can_be_summed(is_train) -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return is_train


def realign_coord(coord_predict, coord_target, token_mask):
    new_coord_target = torch.zeros_like(coord_target).type_as(coord_target)
    bs = token_mask.size(0)

    for i in range(bs):
        _coord_predict = coord_predict[i]
        _coord_target = coord_target[i]
        _token_mask = token_mask[i]

        _coord_predict = _coord_predict[_token_mask].detach().cpu().numpy()
        _coord_target = _coord_target[_token_mask].detach().cpu().numpy()

        _coord_predict = _coord_predict - _coord_predict.mean(axis=0)
        _coord_target = _coord_target - _coord_target.mean(axis=0)

        _r = (
            R.align_vectors(_coord_target, _coord_predict)[0]
            .as_matrix()
            .astype(np.float32)
        )
        _new_coord_target = torch.from_numpy(np.dot(_coord_target, _r)).type_as(
            coord_target
        )
        new_coord_target[i, _token_mask, :] = _new_coord_target

    return new_coord_target


def calc_mask(token_mask):
    sz = token_mask.size()
    distance_mask = torch.zeros(sz[0], sz[1], sz[1]).type_as(token_mask)
    distance_mask = token_mask.unsqueeze(-1) & token_mask.unsqueeze(1)
    coord_mask = torch.zeros(sz[0], sz[1], 3).type_as(token_mask)
    coord_mask.masked_fill_(token_mask.unsqueeze(-1), True)
    return distance_mask, coord_mask
