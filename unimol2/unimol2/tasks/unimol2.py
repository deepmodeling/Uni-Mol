# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os

import numpy as np
from unicore.data import (
    NestedDictionaryDataset,
    EpochShuffleDataset,
)
from unimol2.data import (
    Add2DConformerDataset,
    ConformerSampleDataset,
    CroppingDataset,
    IndexAtomDataset,
    KeyDataset,
    LMDBDataset,
    NormalizeDataset,
    RemoveHydrogenDataset,
    Unimol2FeatureDataset,
)
from unicore.tasks import UnicoreTask, register_task


logger = logging.getLogger(__name__)


@register_task("unimol2")
class UniMol2_Task(UnicoreTask):
    """Task for training transformer auto-encoder models."""

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument(
            "data",
            help="colon separated path to data directories list, \
                            will be iterated upon during epochs in round-robin manner",
        )
        parser.add_argument(
            "--mask-prob",
            default=0.15,
            type=float,
            help="probability of replacing a token with mask",
        )
        parser.add_argument(
            "--drop-feat-prob",
            default=0.5,
            type=float,
            help="probability of dropout the atom / bond feat",
        )
        parser.add_argument(
            "--mask-token-prob",
            default=0.15,
            type=float,
            help="probability of mask token",
        )
        parser.add_argument(
            "--leave-unmasked-prob",
            default=0.0,
            type=float,
            help="probability that a masked token is unmasked",
        )
        parser.add_argument(
            "--random-token-prob",
            default=0.0,
            type=float,
            help="probability of replacing a token with a random token",
        )
        parser.add_argument(
            "--noise-type",
            default="uniform",
            choices=["trunc_normal", "uniform", "normal", "none"],
            help="noise type in coordinate noise",
        )
        parser.add_argument(
            "--noise",
            default=1.0,
            type=float,
            help="coordinate noise for masked atoms",
        )
        parser.add_argument(
            "--max-atoms",
            type=int,
            default=256,
            help="selected maximum number of atoms in a molecule",
        )

    def __init__(self, args, dictionary=None):
        super().__init__(args)
        self.dictionary = dictionary
        self.seed = args.seed
        # set atom type dim 128, lastdim is maskidx
        self.pad_idx = 0
        self.mask_idx = 127

        from multiprocessing.pool import ThreadPool
        self.lmdb_copy_thread = ThreadPool(processes=1)

    @classmethod
    def setup_task(cls, args, **kwargs):
        return cls(args, dictionary=None)

    def load_dataset(self, split, combine=False, **kwargs):
        """Load a given dataset split.
        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        epoch = kwargs['epoch']

        def one_dataset(seed):
            if self.args.mode =='train':
                raw_dataset = LMDBDataset(self.args.data, split, epoch, self.args.max_epoch, self.lmdb_copy_thread)
                # smi2_2Dcoords, 10 3d + 1 2d
                raw_dataset = Add2DConformerDataset(
                    raw_dataset, "smi", "atoms", "coordinates"
                )
            else:
                # for validation
                raw_dataset = LMDBDataset(self.args.data, split, 1, 1, self.lmdb_copy_thread)

            smi_dataset = KeyDataset(raw_dataset, "smi")
            # sampler from 10 + 1 conformer
            dataset = ConformerSampleDataset(
                raw_dataset, seed, "atoms", "coordinates", "coordinates_2d"
            )
            # remove H
            dataset = RemoveHydrogenDataset(
                dataset,
                "atoms",
                "coordinates",
                "coordinates_2d",
                True,
            )
            # cropping atom to max_atoms...
            dataset = CroppingDataset(
                dataset, seed, "atoms", "coordinates", "coordinates_2d", self.args.max_atoms
            )
            dataset = NormalizeDataset(dataset, "coordinates", "coordinates_2d", normalize_coord=True)
            # dataset -> coordinates, coordinates_2d

            token_dataset = KeyDataset(dataset, "atoms")
            origin_token_dataset = IndexAtomDataset(
                smi_dataset, token_dataset,
            )
            coord_dataset = KeyDataset(dataset, "coordinates")
            coordinates_2d = KeyDataset(dataset, "coordinates_2d")

            v2_feat = Unimol2FeatureDataset(
                smi_dataset=smi_dataset,
                token_dataset=origin_token_dataset,
                mask_token_prob=self.args.mask_token_prob,
                pad_idx=self.pad_idx,
                mask_idx=self.mask_idx,

                src_pos_dataset=coord_dataset,
                mask_pos_prob=self.args.mask_prob,
                noise=self.args.noise,
                noise_type=self.args.noise_type,
                drop_feat_prob=self.args.drop_feat_prob,
                seed=seed,
                src_2d_pos_dataset=coordinates_2d
            )

            return v2_feat

        dataset = {
            "batched_data": one_dataset(self.args.seed),
        }

        dataset = NestedDictionaryDataset(dataset)
        if split in ["train", "train.small"]:
            dataset = EpochShuffleDataset(dataset, len(dataset), self.args.seed)
        self.datasets[split] = dataset

    def build_model(self, args):
        from unicore import models

        model = models.build_model(args, self)
        return model
