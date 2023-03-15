# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn.functional as F
from unicore import metrics
from unicore.losses import UnicoreLoss, register_loss


@register_loss("docking_pose")
class DockingPossLoss(UnicoreLoss):
    def __init__(self, task):
        super().__init__(task)

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_outputs = model(**sample["net_input"])
        cross_distance_predict, holo_distance_predict = net_outputs[0], net_outputs[1]

        ### distance loss
        distance_mask = sample["target"]["distance_target"].ne(0)  # 0 is padding
        if self.args.dist_threshold > 0:
            distance_mask &= (
                sample["target"]["distance_target"] < self.args.dist_threshold
            )
        distance_predict = cross_distance_predict[distance_mask]
        distance_target = sample["target"]["distance_target"][distance_mask]
        distance_loss = F.mse_loss(
            distance_predict.float(), distance_target.float(), reduction="mean"
        )

        ### holo distance loss
        holo_distance_mask = sample["target"]["holo_distance_target"].ne(
            0
        )  # 0 is padding
        holo_distance_predict_train = holo_distance_predict[holo_distance_mask]
        holo_distance_target = sample["target"]["holo_distance_target"][
            holo_distance_mask
        ]
        holo_distance_loss = F.smooth_l1_loss(
            holo_distance_predict_train.float(),
            holo_distance_target.float(),
            reduction="mean",
            beta=1.0,
        )

        loss = distance_loss + holo_distance_loss
        sample_size = sample["target"]["holo_coord"].size(0)
        logging_output = {
            "loss": loss.data,
            "cross_loss": distance_loss.data,
            "holo_loss": holo_distance_loss.data,
            "bsz": sample_size,
            "sample_size": 1,
        }
        if not self.training:
            logging_output["smi_name"] = sample["smi_name"]
            logging_output["pocket_name"] = sample["pocket_name"]
            logging_output[
                "cross_distance_predict"
            ] = cross_distance_predict.data.detach().cpu()
            logging_output[
                "holo_distance_predict"
            ] = holo_distance_predict.data.detach().cpu()
            logging_output["atoms"] = (
                sample["net_input"]["mol_src_tokens"].data.detach().cpu()
            )
            logging_output["pocket_atoms"] = (
                sample["net_input"]["pocket_src_tokens"].data.detach().cpu()
            )
            logging_output["holo_center_coordinates"] = (
                sample["holo_center_coordinates"].data.detach().cpu()
            )
            logging_output["holo_coordinates"] = (
                sample["target"]["holo_coord"].data.detach().cpu()
            )
            logging_output["pocket_coordinates"] = (
                sample["net_input"]["pocket_src_coord"].data.detach().cpu()
            )

        return loss, 1, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs, split="valid") -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar("loss", loss_sum / sample_size, sample_size, round=4)
        metrics.log_scalar(
            f"{split}_loss", loss_sum / sample_size, sample_size, round=4
        )
        cross_loss_sum = sum(log.get("cross_loss", 0) for log in logging_outputs)
        metrics.log_scalar(
            "cross_loss", cross_loss_sum / sample_size, sample_size, round=4
        )
        holo_loss_sum = sum(log.get("holo_loss", 0) for log in logging_outputs)
        metrics.log_scalar(
            "holo_loss", holo_loss_sum / sample_size, sample_size, round=4
        )

    @staticmethod
    def logging_outputs_can_be_summed(is_train) -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return is_train
