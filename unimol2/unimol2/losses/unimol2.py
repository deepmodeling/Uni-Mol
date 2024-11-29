# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
from unicore import metrics
from unicore.losses import UnicoreLoss, register_loss


@register_loss("unimol2")
class UniMol2Loss(UnicoreLoss):
    def __init__(self, task):
        super().__init__(task)
        self.padding_idx = 0
        self.seed = task.seed

    def forward(self, model, sample, reduce=True):
        input_key = "batched_data"
        # used for lm head
        masked_tokens = sample[input_key]["target_token"].ne(self.padding_idx)
        sample_size = masked_tokens.long().sum()

        (
            logits_encoder,
            encoder_distance,
            encoder_coord,
        ) = model(**sample, encoder_masked_tokens=masked_tokens)
        target = sample[input_key]["target_token"]
        if masked_tokens is not None:
            target = target[masked_tokens]
        # calculate masked token loss...
        masked_token_loss = F.nll_loss(
            F.log_softmax(logits_encoder, dim=-1, dtype=torch.float32),
            target,
            ignore_index=self.padding_idx,
            reduction="mean",
        )
        masked_pred = logits_encoder.argmax(dim=-1)
        masked_hit = (masked_pred == target).long().sum()
        masked_hit_at6 = (masked_pred == 6).long().sum()
        target_hit_at6 = (target == 6).long().sum()
        masked_cnt = sample_size
        loss = masked_token_loss * self.args.masked_token_loss

        logging_output = {
            "sample_size": 1,
            "bsz": sample[input_key]["src_token"].size(0),
            "seq_len": sample[input_key]["src_token"].size(1)
            * sample[input_key]["src_token"].size(0),
            "masked_token_loss": masked_token_loss.data,
            "masked_token_hit": masked_hit.data,
            "masked_hit_at6": masked_hit_at6.data,
            "target_hit_at6": target_hit_at6.data,
            "masked_token_cnt": masked_cnt,
        }

        if self.args.masked_coord_loss > 0:
            batch_size = sample["batched_data"]["atom_mask"].shape[0]
            atom_mask = sample["batched_data"]["atom_mask"].float()
            pos_mask = atom_mask.unsqueeze(-1)
            pos_target = sample["batched_data"]["target_pos"].float() * pos_mask

            def get_pos_loss(pos_pred):
                pos_pred = pos_pred.float() * pos_mask
                pos_loss = torch.nn.L1Loss(reduction="none")(
                    pos_pred,
                    pos_target,
                ).sum(dim=(-1, -2))
                pos_cnt = pos_mask.squeeze(-1).sum(dim=-1) + 1e-10
                pos_loss = (pos_loss / pos_cnt).sum()
                return pos_loss

            pos_loss = get_pos_loss(encoder_coord) / batch_size
            loss = loss + pos_loss * self.args.masked_coord_loss
            logging_output["masked_coord_loss"] = pos_loss.data

        if self.args.masked_dist_loss > 0:

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

            dist_loss = get_dist_loss(encoder_distance) / batch_size
            loss = loss + dist_loss * self.args.masked_dist_loss
            logging_output["masked_dist_loss"] = dist_loss.data

        logging_output["loss"] = loss.data
        return loss, 1, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs, split="valid") -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        bsz = sum(log.get("bsz", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        seq_len = sum(log.get("seq_len", 0) for log in logging_outputs)
        metrics.log_scalar("loss", loss_sum / sample_size, sample_size, round=5)
        metrics.log_scalar("seq_len", seq_len / bsz, 1, round=3)

        masked_loss = sum(log.get("masked_token_loss", 0) for log in logging_outputs)
        metrics.log_scalar(
            "masked_token_loss", masked_loss / sample_size, sample_size, round=5
        )

        masked_acc = sum(
            log.get("masked_token_hit", 0) for log in logging_outputs
        ) / sum(log.get("masked_token_cnt", 0) for log in logging_outputs)
        metrics.log_scalar("masked_acc", masked_acc, sample_size, round=5)

        masked_hit_at6 = sum(
            log.get("masked_hit_at6", 0) for log in logging_outputs
        ) / sum(log.get("masked_token_cnt", 0) for log in logging_outputs)
        metrics.log_scalar("masked_hit_at6", masked_hit_at6, sample_size, round=3)

        target_hit_at6 = sum(
            log.get("target_hit_at6", 0) for log in logging_outputs
        ) / sum(log.get("masked_token_cnt", 0) for log in logging_outputs)
        metrics.log_scalar("target_hit_at6", target_hit_at6, sample_size, round=3)

        masked_coord_loss = sum(
            log.get("masked_coord_loss", 0) for log in logging_outputs
        )
        if masked_coord_loss > 0:
            metrics.log_scalar(
                "masked_coord_loss",
                masked_coord_loss / sample_size,
                sample_size,
                round=5,
            )

        masked_dist_loss = sum(
            log.get("masked_dist_loss", 0) for log in logging_outputs
        )
        if masked_dist_loss > 0:
            metrics.log_scalar(
                "masked_dist_loss", masked_dist_loss / sample_size, sample_size, round=5
            )

        x_norm_loss = sum(log.get("x_norm_loss", 0) for log in logging_outputs)
        if x_norm_loss > 0:
            metrics.log_scalar(
                "x_norm_loss", x_norm_loss / sample_size, sample_size, round=5
            )

        delta_pair_repr_norm_loss = sum(
            log.get("delta_pair_repr_norm_loss", 0) for log in logging_outputs
        )
        if delta_pair_repr_norm_loss > 0:
            metrics.log_scalar(
                "delta_pair_repr_norm_loss",
                delta_pair_repr_norm_loss / sample_size,
                sample_size,
                round=3,
            )

    @staticmethod
    def logging_outputs_can_be_summed(is_train) -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True



@register_loss("unimol_infer")
class UniMolInferLoss(UnicoreLoss):
    def __init__(self, task):
        super().__init__(task)
        self.padding_idx = task.dictionary.pad()

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        input_key = "net_input"
        target_key = "target"
        src_tokens = sample[input_key]["src_tokens"].ne(self.padding_idx)
        (
            encoder_rep,
            encoder_pair_rep,
        ) = model(**sample[input_key], features_only=True)
        sample_size = sample[input_key]["src_tokens"].size(0)
        encoder_pair_rep_list = []
        for i in range(sample_size):  # rm padding token
            encoder_pair_rep_list.append(encoder_pair_rep[i][src_tokens[i], :][:, src_tokens[i]].data.cpu().numpy())
        logging_output = {
                "mol_repr_cls": encoder_rep[:, 0, :].data.cpu().numpy(),  # get cls token
                "pair_repr": encoder_pair_rep_list,
                "smi_name": sample[target_key]["smi_name"],
                "bsz": sample[input_key]["src_tokens"].size(0),
            }
        return 0, sample_size, logging_output
