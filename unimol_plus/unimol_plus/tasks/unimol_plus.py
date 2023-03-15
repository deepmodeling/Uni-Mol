# Copyright (c) DP Techonology, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os

import numpy as np
from unicore.data import (
    NestedDictionaryDataset,
    EpochShuffleDataset,
)
from unimol_plus.data import (
    KeyDataset,
    LMDBDataset,
    ConformationSampleDataset,
    ConformationExpandDataset,
    UnimolPlusFeatureDataset,
)
from unicore.tasks import UnicoreTask, register_task

logger = logging.getLogger(__name__)


@register_task("unimol_plus")
class UnimolPlusTask(UnicoreTask):
    """Task for training transformer auto-encoder models."""

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument("data", help="downstream data path")

    def __init__(self, args):
        super().__init__(args)
        self.seed = args.seed

    @classmethod
    def setup_task(cls, args, **kwargs):
        return cls(args)

    def load_dataset(self, split, force_valid=False, **kwargs):
        """Load a given dataset split.
        Args:
            split (str): name of the data scoure (e.g., bppp)
        """
        split_path = os.path.join(self.args.data, split + ".lmdb")
        dataset = LMDBDataset(split_path)
        is_train = (split == "train") and not force_valid
        if is_train:
            sample_dataset = ConformationSampleDataset(
                dataset,
                self.seed,
                "input_pos",
                "label_pos",
            )
        else:
            sample_dataset = ConformationExpandDataset(
                dataset,
                self.seed,
                "input_pos",
                "label_pos",
            )
        raw_coord_dataset = KeyDataset(sample_dataset, "coordinates")
        tgt_coord_dataset = KeyDataset(sample_dataset, "target_coordinates")
        graph_features = UnimolPlusFeatureDataset(
            sample_dataset,
            raw_coord_dataset,
            tgt_coord_dataset if split in ["train", "valid_our"] else None,
            is_train=is_train,
            label_prob=self.args.label_prob,
            mid_prob=self.args.mid_prob,
            mid_lower=self.args.mid_lower,
            mid_upper=self.args.mid_upper,
            noise=self.args.noise_scale,
            seed=self.seed + 2,
        )

        nest_dataset = NestedDictionaryDataset(
            {
                "batched_data": graph_features,
            },
        )
        if is_train:
            nest_dataset = EpochShuffleDataset(
                nest_dataset, len(nest_dataset), self.seed
            )
        self.datasets[split] = nest_dataset

    def build_model(self, args):
        from unicore import models

        model = models.build_model(args, self)
        return model
