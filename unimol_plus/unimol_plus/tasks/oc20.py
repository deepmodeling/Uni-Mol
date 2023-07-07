# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
#

import os
import numpy as np
from unicore.data import (
    NestedDictionaryDataset,
    EpochShuffleDataset,
)
from unimol_plus.data import (
    LMDBDataset,
    StackedLMDBDataset,
    Is2reDataset,
)

from unicore.tasks import UnicoreTask, register_task


@register_task("oc20")
class MDTask(UnicoreTask):
    @staticmethod
    def add_args(parser):
        parser.add_argument("data", metavar="FILE", help="file prefix for data")
        parser.add_argument(
            "--train-with-valid-data",
            default=False,
            action="store_true",
        )

    def __init__(self, args):
        super().__init__(args)
        self.seed = args.seed

    @classmethod
    def setup_task(cls, args, **kwargs):
        return cls(args)

    def load_dataset(self, split, combine=False, **kwargs):
        assert split in [
            "train",
            "val_id",
            "val_ood_ads",
            "val_ood_cat",
            "val_ood_both",
            "test_id",
            "test_ood_ads",
            "test_ood_cat",
            "test_ood_both",
            "test_sumbit",
        ], "invalid split: {}!".format(split)
        print(" > Loading {} ...".format(split))

        if self.args.train_with_valid_data and split == "train":
            datasets = []
            for cur_split in [
                "train",
                "val_id",
                "val_ood_ads",
                "val_ood_cat",
                "val_ood_both",
            ]:
                db_path = os.path.join(self.args.data, cur_split, "data.lmdb")
                lmdb_dataset = LMDBDataset(db_path, key_to_id=False, gzip=False)
                datasets.append(lmdb_dataset)
            lmdb_dataset = StackedLMDBDataset(datasets)
        else:
            db_path = os.path.join(self.args.data, split, "data.lmdb")
            lmdb_dataset = LMDBDataset(db_path, key_to_id=False, gzip=False)

        is_train = split == "train"
        is2re_dataset = Is2reDataset(lmdb_dataset, self.args, is_train=is_train)
        nest_dataset = NestedDictionaryDataset(
            {
                "batched_data": is2re_dataset,
            },
        )

        if is_train:
            nest_dataset = EpochShuffleDataset(
                nest_dataset, len(nest_dataset), self.seed
            )
        self.datasets[split] = nest_dataset

        print("| Loaded {} with {} samples".format(split, len(nest_dataset)))

        self.datasets[split] = nest_dataset

    def build_model(self, args):
        from unicore import models

        model = models.build_model(args, self)
        return model
