# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os

import numpy as np
from unicore.data import (
    NestedDictionaryDataset,
    EpochShuffleDataset,
    LMDBDataset,
    RawLabelDataset,
    RawArrayDataset,
)
from unimol2.data import (
    KeyDataset,
    ConformerSampleDataset,
    RemoveHydrogenDataset,
    NormalizeDataset,
    CroppingDataset,
    Add2DConformerDataset,
    MoleculeFeatureDataset,
    Unimol2FinetuneFeatureDataset,
    IndexAtomDataset
)

from unimol2.data.tta_dataset import TTADataset
from unicore.tasks import UnicoreTask, register_task


logger = logging.getLogger(__name__)

task_metainfo = {
    "qm9dft_v2": {
        "mean": [-0.23997669940621352, 0.011123767412331285, 0.2511003712141015],
        "std": [0.02213143402267657, 0.046936069870866196, 0.04751888787058615],
        "target_name": ["homo", "lumo", "gap"],
    },
     "qm9dft_v2_alpha": {
        "mean": 75.19129618702617,
        "std": 8.187762224050584,
        "target_name": "alpha",
    },
    "qm9dft_v2_cv": {
        "mean": 31.600675893490678,
        "std": 4.062456253369289,
        "target_name": "cv",
    },
    "qm9dft_v2_mu": {
        "mean": 2.7060374694700675,
        "std": 1.530388280934567,
        "target_name": "mu",
    },
    "qm9dft_v2_r2": {
        "mean": 1189.5274499667628,
        "std": 279.7561272394077,
        "target_name": "r2",
    },
    "qm9dft_v2_zpve": {
        "mean": 0.14852438909511897,
        "std": 0.03327377213900081,
        "target_name": "zpve",
    },
    "qm9dft_v2_g": {
        "mean": -1629.388193917963,
        "std": 220.20626683425812,
        "target_name": "g",
    },
    "qm9dft_v2_h": {
        "mean": -1771.5469283884158,
        "std": 243.1501571556723,
        "target_name": "h",
    },
    "qm9dft_v2_u": {
        "mean": -1761.4806474164875,
        "std":  241.435201920648,
        "target_name": "u",
    },
    "qm9dft_v2_u0": {
        "mean": -1750.8129967646425 ,
        "std": 239.3124800445088,
        "target_name": "u0",
    }
}


@register_task("mol_finetune")
class UniMolFinetuneTask(UnicoreTask):
    """Task for training transformer auto-encoder models."""

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument("data", help="downstream data path")
        parser.add_argument("--task-name", type=str, help="downstream task name")
        parser.add_argument(
            "--classification-head-name",
            default="classification",
            help="finetune downstream task name",
        )
        parser.add_argument(
            "--num-classes",
            default=1,
            type=int,
            help="finetune downstream task classes numbers",
        )
        parser.add_argument("--reg", action="store_true", help="regression task")
        parser.add_argument("--no-shuffle", action="store_true", help="shuffle data")
        parser.add_argument(
            "--conf-size",
            default=10,
            type=int,
            help="number of conformers generated with each molecule",
        )
        parser.add_argument(
            "--remove-hydrogen",
            action="store_true",
            help="remove hydrogen atoms",
        )
        parser.add_argument(
            "--drop-feat-prob",
            default=0.0,
            type=float,
            help="probability of dropout the atom / bond feat",
        )
        parser.add_argument(
            "--use-2d-pos-prob",
            default=0.0,
            type=float,
            help="probability of dropout the atom / bond feat",
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
        if self.args.task_name in task_metainfo:
            # for regression task, pre-compute mean and std
            self.mean = task_metainfo[self.args.task_name]["mean"]
            self.std = task_metainfo[self.args.task_name]["std"]

    @classmethod
    def setup_task(cls, args, **kwargs):
        return cls(args, dictionary=None)

    def load_dataset(self, split, **kwargs):
        """Load a given dataset split.
        Args:
            split (str): name of the data scoure (e.g., train)
        """
        split_path = os.path.join(self.args.data, self.args.task_name, split + ".lmdb")
        raw_dataset = LMDBDataset(split_path)
        if split == "train":
            tgt_dataset = KeyDataset(raw_dataset, "target")
            smi_dataset = KeyDataset(raw_dataset, "smi")
            raw_dataset = Add2DConformerDataset(
                raw_dataset, "smi", "atoms", "coordinates"
            )
            dataset = ConformerSampleDataset(
                raw_dataset, self.args.seed, "atoms", "coordinates", "coordinates_2d"
            )
        else:
            raw_dataset = TTADataset(
                raw_dataset, self.args.seed, "atoms", "coordinates", self.args.conf_size
            )
            tgt_dataset = KeyDataset(raw_dataset, "target")
            smi_dataset = KeyDataset(raw_dataset, "smi")

            dataset = Add2DConformerDataset(
                raw_dataset, "smi", "atoms", "coordinates"
            )

        molecule_dataset = MoleculeFeatureDataset(
                dataset=raw_dataset,
                drop_feat_prob=self.args.drop_feat_prob,
                smi_key='smi'
        )

        dataset = RemoveHydrogenDataset(
            dataset,
            "atoms",
            "coordinates",
            "coordinates_2d",
            True,
        )
        dataset = CroppingDataset(
            dataset, self.seed, "atoms", "coordinates", "coordinates_2d", self.args.max_atoms
        )
        dataset = NormalizeDataset(dataset, "coordinates", "coordinates_2d", normalize_coord=True)
        src_dataset = KeyDataset(dataset, "atoms")

        src_dataset = IndexAtomDataset(
            smi_dataset, src_dataset,
        )
        if self.args.use_2d_pos_prob:
            coord_dataset = KeyDataset(dataset, "coordinates_2d")
        else:
            coord_dataset = KeyDataset(dataset, "coordinates")

        finetune_feat = Unimol2FinetuneFeatureDataset(
            smi_dataset=smi_dataset,
            token_dataset=src_dataset,
            src_pos_dataset=coord_dataset,
            molecule_dataset=molecule_dataset,
            seed=self.seed,
        )

        nest_dataset = NestedDictionaryDataset(
            {
                "net_input": {
                   "batched_data": finetune_feat
                },
                "target": {
                    "finetune_target": RawLabelDataset(tgt_dataset),
                },
                "smi_name": RawArrayDataset(smi_dataset),
            },
        )
        if not self.args.no_shuffle and split == "train":
            nest_dataset = EpochShuffleDataset(nest_dataset, len(nest_dataset), self.args.seed)

        self.datasets[split] = nest_dataset

    def build_model(self, args):
        from unicore import models

        model = models.build_model(args, self)
        model.register_classification_head(
            self.args.classification_head_name,
            num_classes=self.args.num_classes,
        )
        return model
