#!/usr/bin/env python3 -u
# Copyright (c) DP Techonology, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sys
import pickle
import torch
import lmdb
import gzip
import numpy as np
from unicore import checkpoint_utils, distributed_utils, options, utils
from unicore.logging import progress_bar
from unicore import tasks

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("unimol.inference")


def predicted_lddt(plddt_logits: torch.Tensor) -> torch.Tensor:
    """Computes per-residue pLDDT from logits.
    Args:
        logits: [num_res, num_bins] output from the PredictedLDDTHead.
    Returns:
        plddt: [num_res] per-residue pLDDT.
    """
    num_bins = plddt_logits.shape[-1]
    bin_probs = torch.nn.functional.softmax(plddt_logits.float(), dim=-1)
    bin_width = 1.0 / num_bins
    bounds = torch.arange(
        start=0.5 * bin_width, end=1.0, step=bin_width, device=plddt_logits.device
    )
    plddt = torch.sum(
        bin_probs * bounds.view(*((1,) * len(bin_probs.shape[:-1])), *bounds.shape),
        dim=-1,
    )
    return plddt


def masked_mean(mask, value, dim, eps=1e-10, keepdim=False):
    mask = mask.expand(*value.shape)
    return torch.sum(mask * value, dim=dim, keepdim=keepdim) / (
        eps + torch.sum(mask, dim=dim, keepdim=keepdim)
    )


def main(args):

    assert (
        args.batch_size is not None
    ), "Must specify batch size either with --batch-size"

    use_fp16 = args.fp16
    use_cuda = torch.cuda.is_available() and not args.cpu

    if use_cuda:
        torch.cuda.set_device(args.device_id)

    if args.distributed_world_size > 1:
        data_parallel_world_size = distributed_utils.get_data_parallel_world_size()
        data_parallel_rank = distributed_utils.get_data_parallel_rank()
    else:
        data_parallel_world_size = 1
        data_parallel_rank = 0

    # Load model
    logger.info("loading model(s) from {}".format(args.path))
    state = checkpoint_utils.load_checkpoint_to_cpu(args.path)
    task = tasks.setup_task(args)
    model = task.build_model(args)
    model.load_state_dict(state["ema"]["params"], strict=True)

    if use_cuda:
        model.cuda()

    model.eval()

    # Print args
    logger.info(args)

    # Build loss
    loss = task.build_loss(args)
    loss.eval()
    if data_parallel_world_size > 1:
        tmp = distributed_utils.all_gather_list(
            [torch.tensor(0)],
            max_size=10000,
            group=distributed_utils.get_data_parallel_group(),
        )
    for subset in args.valid_subset.split(","):
        try:
            task.load_dataset(subset, combine=False, epoch=1, force_valid=True)
            dataset = task.dataset(subset)
        except KeyError:
            raise Exception("Cannot find dataset: " + subset)

        itr = task.get_batch_iterator(
            dataset=dataset,
            batch_size=args.batch_size,
            ignore_invalid_inputs=True,
            required_batch_size_multiple=args.required_batch_size_multiple,
            seed=args.seed,
            num_shards=data_parallel_world_size,
            shard_id=data_parallel_rank,
            num_workers=args.num_workers,
            data_buffer_size=args.data_buffer_size,
        ).next_epoch_itr(shuffle=False)
        progress = progress_bar.progress_bar(
            itr,
            log_format=args.log_format,
            log_interval=args.log_interval,
            prefix=f"valid on '{subset}' subset",
            default_log_format=("tqdm" if not args.no_progress_bar else "simple"),
        )
        outputs = []
        for i, sample in enumerate(progress):
            sample = utils.move_to_cuda(sample) if use_cuda else sample
            if len(sample) == 0:
                continue
            with torch.no_grad():
                (graph_output, pos_pred) = model(**sample)[:2]
            id = sample["batched_data"]["id"]
            gap_pred = graph_output.cpu().numpy()
            targets = sample["batched_data"]["target"].float().view(-1).cpu().numpy()
            assert len(id) == len(pos_pred)
            id = id.cpu().numpy()
            pos_pred = pos_pred.cpu().numpy()
            outputs.append(
                (
                    id,
                    pos_pred,
                    gap_pred,
                    targets,
                )
            )
            progress.log({}, step=i)
        pickle.dump(
            outputs,
            open(
                os.path.join(
                    args.results_path, subset + "_{}.pkl".format(data_parallel_rank)
                ),
                "wb",
            ),
        )
        print("Finished {} subset, rank {}".format(subset, data_parallel_rank))
        if data_parallel_world_size > 1:
            tmp = distributed_utils.all_gather_list(
                [torch.tensor(0)],
                max_size=10000,
                group=distributed_utils.get_data_parallel_group(),
            )

    return None


def cli_main():
    parser = options.get_validation_parser()
    options.add_model_args(parser)
    args = options.parse_args_and_arch(parser)
    distributed_utils.call_main(args, main)


if __name__ == "__main__":
    cli_main()
