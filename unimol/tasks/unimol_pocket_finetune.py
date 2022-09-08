# Copyright (c) DP Techonology, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os

from unicore.data import (
    Dictionary,
    NestedDictionaryDataset,
    LMDBDataset,
    AppendTokenDataset,
    PrependTokenDataset,
    RightPadDataset,
    TokenizeDataset,
    RightPadDataset2D,
    RawLabelDataset,
    FromNumpyDataset,
    EpochShuffleDataset,
)

from unimol.data import (
    KeyDataset,
    ConformerSamplePocketFinetuneDataset,
    DistanceDataset,
    EdgeTypeDataset,
    NormalizeDataset,
    RightPadDatasetCoord,
    CroppingResiduePocketDataset,
    RemoveHydrogenResiduePocketDataset,
    FromStrLabelDataset,
)

from unicore.tasks import UnicoreTask, register_task


logger = logging.getLogger(__name__)

task_metainfo = {
    "Score": {
        "mean": -0.02113608960384876,
        "std": 0.14467607204629246,
    },
    "Druggability Score": {
        "mean": 0.04279187401338044,
        "std": 0.1338187819653573,
    },
    "Total SASA": {
        "mean": 118.7343246335413,
        "std": 59.82260887999069,
    },
    "Hydrophobicity score": {
        "mean": 16.824823092535517,
        "std": 18.16340833552264,
    },
}


@register_task("pocket_finetune")
class UniMolPocketFinetuneTask(UnicoreTask):
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
            default=2,
            type=int,
            help="finetune downstream task classes numbers",
        )
        parser.add_argument(
            "--remove-hydrogen",
            action="store_true",
            help="remove hydrogen atoms",
        )
        parser.add_argument(
            "--max-atoms",
            type=int,
            default=256,
            help="selected maximum number of atoms in a molecule",
        )
        parser.add_argument(
            "--dict-name",
            default="dict_pkt.txt",
            help="dictionary file",
        )
        parser.add_argument(
            "--fpocket-score",
            default="Druggability Score",
            help="Select one of the 4 Fpocket scores as the target",
            choices=[
                "Score",
                "Druggability Score",
                "Total SASA",
                "Hydrophobicity score",
            ],
        )

    def __init__(self, args, dictionary):
        super().__init__(args)
        self.dictionary = dictionary
        self.seed = args.seed
        # add mask token
        self.mask_idx = dictionary.add_symbol("[MASK]", is_special=True)
        if self.args.task_name == "drugabbility":
            if self.args.fpocket_score in task_metainfo:
                # for regression task, pre-compute mean and std
                self.mean = task_metainfo[self.args.fpocket_score]["mean"]
                self.std = task_metainfo[self.args.fpocket_score]["std"]
        else:
            self.mean, self.std = None, None

    @classmethod
    def setup_task(cls, args, **kwargs):
        dictionary = Dictionary.load(os.path.join(args.data, args.dict_name))
        logger.info("dictionary: {} types".format(len(dictionary)))
        return cls(args, dictionary)

    def load_dataset(self, split, **kwargs):
        """Load a given dataset split.
        Args:
            split (str): name of the data scoure (e.g., bppp)
        """
        split_path = os.path.join(self.args.data, self.args.task_name, split + ".lmdb")
        dataset = LMDBDataset(split_path)
        if self.args.task_name == "druggability":
            tgt_dataset_inner = KeyDataset(dataset, "target")
            tgt_dataset = KeyDataset(tgt_dataset_inner, self.args.fpocket_score)
            tgt_dataset = FromStrLabelDataset(tgt_dataset)
        else:
            tgt_dataset = KeyDataset(dataset, "target")
            tgt_dataset = RawLabelDataset(tgt_dataset)

        dataset = ConformerSamplePocketFinetuneDataset(
            dataset, self.seed, "atoms", "residue", "coordinates"
        )
        dataset = RemoveHydrogenResiduePocketDataset(
            dataset, "atoms", "residue", "coordinates", self.args.remove_hydrogen
        )
        dataset = CroppingResiduePocketDataset(
            dataset, self.seed, "atoms", "residue", "coordinates", self.args.max_atoms
        )
        dataset = NormalizeDataset(dataset, "coordinates")
        src_dataset = KeyDataset(dataset, "atoms")
        src_dataset = TokenizeDataset(
            src_dataset, self.dictionary, max_seq_len=self.args.max_seq_len
        )
        coord_dataset = KeyDataset(dataset, "coordinates")

        def PrependAndAppend(dataset, pre_token, app_token):
            dataset = PrependTokenDataset(dataset, pre_token)
            return AppendTokenDataset(dataset, app_token)

        src_dataset = PrependAndAppend(
            src_dataset, self.dictionary.bos(), self.dictionary.eos()
        )
        edge_type = EdgeTypeDataset(src_dataset, len(self.dictionary))
        coord_dataset = FromNumpyDataset(coord_dataset)
        coord_dataset = PrependAndAppend(coord_dataset, 0.0, 0.0)
        distance_dataset = DistanceDataset(coord_dataset)

        nest_dataset = NestedDictionaryDataset(
            {
                "net_input": {
                    "src_tokens": RightPadDataset(
                        src_dataset,
                        pad_idx=self.dictionary.pad(),
                    ),
                    "src_coord": RightPadDatasetCoord(
                        coord_dataset,
                        pad_idx=0,
                    ),
                    "src_distance": RightPadDataset2D(
                        distance_dataset,
                        pad_idx=0,
                    ),
                    "src_edge_type": RightPadDataset2D(
                        edge_type,
                        pad_idx=0,
                    ),
                },
                "target": {
                    "finetune_target": tgt_dataset,
                },
            },
        )

        if split.startswith("train"):
            nest_dataset = EpochShuffleDataset(
                nest_dataset, len(nest_dataset), self.args.seed
            )
        self.datasets[split] = nest_dataset

    def build_model(self, args):
        from unicore import models

        model = models.build_model(args, self)
        model.register_classification_head(
            self.args.classification_head_name,
            num_classes=self.args.num_classes,
        )
        return model
