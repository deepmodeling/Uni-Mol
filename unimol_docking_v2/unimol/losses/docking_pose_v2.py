# Copyright (c) DP Techonology, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
import numpy as np
from unicore import metrics
from unicore.losses import UnicoreLoss, register_loss


@register_loss("docking_pose_v2")
class DockingPosseV2Loss(UnicoreLoss):
    def __init__(self, task):
        super().__init__(task)
        self.eos_idx = task.dictionary.eos()
        self.bos_idx = task.dictionary.bos()
        self.padding_idx = task.dictionary.pad()

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        cross_distance_predict, holo_distance_predict, coord_predict, prmsd_predict = net_output[:4]

        ### distance loss
        distance_mask = sample["target"]["distance_target"].ne(0) # 0 is padding
        if self.args.dist_threshold > 0:
            distance_mask &= sample["target"]["distance_target"] < self.args.dist_threshold
        distance_predict = cross_distance_predict[distance_mask]
        distance_target =  sample["target"]["distance_target"][distance_mask]
        distance_loss = F.mse_loss(
            distance_predict.float(), 
            distance_target.float(), 
            reduction="mean")
        
        ### holo distance loss
        token_mask = sample["net_input"]["mol_src_tokens"].ne(self.padding_idx) & \
                     sample["net_input"]["mol_src_tokens"].ne(self.eos_idx) & \
                     sample["net_input"]["mol_src_tokens"].ne(self.bos_idx)
        holo_distance_mask = token_mask.unsqueeze(-1) & token_mask.unsqueeze(1)
        holo_distance_predict = holo_distance_predict[holo_distance_mask]
        holo_distance_target =  sample["target"]["holo_distance_target"][holo_distance_mask]
        holo_distance_loss = F.smooth_l1_loss(
            holo_distance_predict.float(), 
            holo_distance_target.float(),
            reduction="mean",
            beta=1.0,
            )

        ### coord loss
        coord_target = sample["target"]["holo_coord"]
        coord_mask = coord_target.ne(0)  # 0 is padding
        coord_loss = (((coord_predict - coord_target)**2).sum(dim=[1,2]) / coord_mask[:,:,0].sum(dim=-1)).sqrt().mean()

        ### prmsd loss
        tick = 0.25
        max_bins = 32 
        token_mask = coord_mask[:,:,0]
        prmsd_target = ((coord_predict - coord_target)**2 * coord_mask).sum(dim=-1).sqrt()
        prmsd_target = (prmsd_target / tick).long()
        prmsd_target[prmsd_target >= (max_bins - 1)] = max_bins - 1
        prmsd_target[prmsd_target < 0] = 0
        prmsd_logit = F.softmax(prmsd_predict.float(), dim=-1)   # BS, N, MAX_BINS
        prmsd_predict = F.log_softmax(prmsd_predict.float(), dim=-1)   # BS, N, MAX_BINS
        prmsd_loss = F.nll_loss(
            prmsd_predict[token_mask],
            prmsd_target[token_mask],
            reduction="mean",
        )

        loss = distance_loss + holo_distance_loss + coord_loss + + prmsd_loss*0.1

        weight = torch.arange(max_bins,).type_as(prmsd_logit).unsqueeze(0) + tick / 2
        prmsd_score = (prmsd_logit * weight).sum(dim=-1).mean(dim=-1)

        sample_size = sample["target"]["holo_coord"].size(0)
        logging_output = {
            "loss": loss.data,
            "cross_distance_loss": distance_loss.data,
            "distance_loss": holo_distance_loss.data,
            "coord_loss": coord_loss.data,
            "prmsd_loss": prmsd_loss.data,
            "prmsd_score": prmsd_score.data,
            "bsz": sample_size,
            "sample_size": 1,
            "coord_predict": coord_predict.data,   # last iteration
            "coord_target": sample["target"]["holo_coord"].data,
        }
        if not self.training:
            logging_output["smi_name"] = sample["smi_name"]
            logging_output["pocket_name"] = sample["pocket_name"]
            logging_output["coord_predict"] = coord_predict.data.detach().cpu()
            logging_output["prmsd_score"] = prmsd_score.data.detach().cpu()
            logging_output["atoms"] = sample["net_input"]["mol_src_tokens"].data.detach().cpu()
            logging_output["pocket_atoms"] = sample["net_input"]["pocket_src_tokens"].data.detach().cpu()
            logging_output["coordinates"] = sample["net_input"]["mol_src_coord"].data.detach().cpu()
            logging_output["holo_coordinates"] = sample["target"]["holo_coord"].data.detach().cpu()
            logging_output["pocket_coordinates"] = sample["net_input"]["pocket_src_coord"].data.detach().cpu()
            logging_output["holo_center_coordinates"] = sample["holo_center_coordinates"].data.detach().cpu()

        return loss, sample_size, logging_output
        

    @staticmethod
    def reduce_metrics(logging_outputs, split='valid') -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size, sample_size, round=3
        )
        metrics.log_scalar(
            f"{split}_loss", loss_sum / sample_size, sample_size, round=3
        )
        cross_distance_loss = sum(log.get("cross_distance_loss", 0) for log in logging_outputs)
        if cross_distance_loss > 0:
            metrics.log_scalar(
                "cross_distance_loss", cross_distance_loss / sample_size, sample_size, round=3
            )
        distance_loss = sum(log.get("distance_loss", 0) for log in logging_outputs)
        if distance_loss > 0:
            metrics.log_scalar(
                "distance_loss", distance_loss / sample_size, sample_size, round=3
            )
        coord_loss = sum(log.get("coord_loss", 0) for log in logging_outputs)
        if coord_loss > 0:
            coord_predict = [log.get("coord_predict")[i].cpu().numpy() for log in logging_outputs for i in range(log.get("coord_predict").size(0))]
            coord_target = [log.get("coord_target")[i].cpu().numpy() for log in logging_outputs for i in range(log.get("coord_target").size(0))]
            metrics.log_scalar(
                "coord_loss", coord_loss / sample_size, sample_size, round=3
            )
            rmsd_list = [RMSD(_predict, _target) for _predict,_target in zip(coord_predict, coord_target)]
            metrics.log_scalar(
                "RMSD", np.mean(rmsd_list), sample_size, round=3
            )
        prmsd_loss = sum(log.get("prmsd_loss", 0) for log in logging_outputs)
        if prmsd_loss > 0:
            metrics.log_scalar(
                "prmsd_loss", prmsd_loss / sample_size, sample_size, round=3
            )

    @staticmethod
    def logging_outputs_can_be_summed(is_train) -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return False


def RMSD(coord_predict, coord_target):
    mask = coord_target != 0
    rmsd = np.sqrt(np.sum(((coord_predict - coord_target) ** 2) * mask) / (mask[:,0].sum()))
    return rmsd
