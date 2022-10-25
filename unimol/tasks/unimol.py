# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os

import numpy as np
from unicore.data import (
    Dictionary,
    NestedDictionaryDataset,
    AppendTokenDataset,
    PrependTokenDataset,
    RightPadDataset,
    EpochShuffleDataset,
    TokenizeDataset,
    RightPadDataset2D,
    FromNumpyDataset,
    RawArrayDataset,
)
from unimol.data import (
    KeyDataset,
    ConformerSampleDataset,
    DistanceDataset,
    EdgeTypeDataset,
    MaskPointsDataset,
    RemoveHydrogenDataset,
    AtomTypeDataset,
    NormalizeDataset,
    CroppingDataset,
    RightPadDatasetCoord,
    Add2DConformerDataset,
    LMDBDataset,
)
from unicore.tasks import UnicoreTask, register_task


logger = logging.getLogger(__name__)


@register_task("unimol")
class UniMolTask(UnicoreTask):
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
            "--leave-unmasked-prob",
            default=0.05,
            type=float,
            help="probability that a masked token is unmasked",
        )
        parser.add_argument(
            "--random-token-prob",
            default=0.05,
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
            "--remove-hydrogen",
            action="store_true",
            help="remove hydrogen atoms",
        )
        parser.add_argument(
            "--remove-polar-hydrogen",
            action="store_true",
            help="remove polar hydrogen atoms",
        )
        parser.add_argument(
            "--max-atoms",
            type=int,
            default=256,
            help="selected maximum number of atoms in a molecule",
        )
        parser.add_argument(
            "--dict-name",
            default="dict.txt",
            help="dictionary file",
        )
        parser.add_argument(
            "--only-polar",
            default=1,
            type=int,
            help="1: only polar hydrogen ; -1: all hydrogen ; 0: remove all hydrogen ",
        )

    def __init__(self, args, dictionary):
        super().__init__(args)
        self.dictionary = dictionary
        self.seed = args.seed
        # add mask token
        self.mask_idx = dictionary.add_symbol("[MASK]", is_special=True)
        if self.args.only_polar > 0:
            self.args.remove_polar_hydrogen = True
        elif args.only_polar < 0:
            self.args.remove_polar_hydrogen = False
        else:
            self.args.remove_hydrogen = True

    @classmethod
    def setup_task(cls, args, **kwargs):
        dictionary = Dictionary.load(os.path.join(args.data, args.dict_name))
        logger.info("dictionary: {} types".format(len(dictionary)))
        return cls(args, dictionary)

    def load_dataset(self, split, combine=False, **kwargs):
        """Load a given dataset split.
        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        split_path = os.path.join(self.args.data, split + ".lmdb")

        raw_dataset = LMDBDataset(split_path)

        def one_dataset(raw_dataset, coord_seed, mask_seed):
            if self.args.mode =='train':
                raw_dataset = Add2DConformerDataset(
                    raw_dataset, "smi", "atoms", "coordinates"
                )
            smi_dataset = KeyDataset(raw_dataset, "smi")
            dataset = ConformerSampleDataset(
                raw_dataset, coord_seed, "atoms", "coordinates"
            )
            dataset = AtomTypeDataset(raw_dataset, dataset)
            dataset = RemoveHydrogenDataset(
                dataset,
                "atoms",
                "coordinates",
                self.args.remove_hydrogen,
                self.args.remove_polar_hydrogen,
            )
            dataset = CroppingDataset(
                dataset, self.seed, "atoms", "coordinates", self.args.max_atoms
            )
            dataset = NormalizeDataset(dataset, "coordinates", normalize_coord=True)
            token_dataset = KeyDataset(dataset, "atoms")
            token_dataset = TokenizeDataset(
                token_dataset, self.dictionary, max_seq_len=self.args.max_seq_len
            )
            coord_dataset = KeyDataset(dataset, "coordinates")
            expand_dataset = MaskPointsDataset(
                token_dataset,
                coord_dataset,
                self.dictionary,
                pad_idx=self.dictionary.pad(),
                mask_idx=self.mask_idx,
                noise_type=self.args.noise_type,
                noise=self.args.noise,
                seed=mask_seed,
                mask_prob=self.args.mask_prob,
                leave_unmasked_prob=self.args.leave_unmasked_prob,
                random_token_prob=self.args.random_token_prob,
            )

            def PrependAndAppend(dataset, pre_token, app_token):
                dataset = PrependTokenDataset(dataset, pre_token)
                return AppendTokenDataset(dataset, app_token)

            encoder_token_dataset = KeyDataset(expand_dataset, "atoms")
            encoder_target_dataset = KeyDataset(expand_dataset, "targets")
            encoder_coord_dataset = KeyDataset(expand_dataset, "coordinates")

            src_dataset = PrependAndAppend(
                encoder_token_dataset, self.dictionary.bos(), self.dictionary.eos()
            )
            tgt_dataset = PrependAndAppend(
                encoder_target_dataset, self.dictionary.pad(), self.dictionary.pad()
            )
            encoder_coord_dataset = PrependAndAppend(encoder_coord_dataset, 0.0, 0.0)
            encoder_distance_dataset = DistanceDataset(encoder_coord_dataset)

            edge_type = EdgeTypeDataset(src_dataset, len(self.dictionary))
            coord_dataset = FromNumpyDataset(coord_dataset)
            coord_dataset = PrependAndAppend(coord_dataset, 0.0, 0.0)
            distance_dataset = DistanceDataset(coord_dataset)
            return {
                "src_tokens": RightPadDataset(
                    src_dataset,
                    pad_idx=self.dictionary.pad(),
                ),
                "src_coord": RightPadDatasetCoord(
                    encoder_coord_dataset,
                    pad_idx=0,
                ),
                "src_distance": RightPadDataset2D(
                    encoder_distance_dataset,
                    pad_idx=0,
                ),
                "src_edge_type": RightPadDataset2D(
                    edge_type,
                    pad_idx=0,
                ),
            }, {
                "tokens_target": RightPadDataset(
                    tgt_dataset, pad_idx=self.dictionary.pad()
                ),
                "distance_target": RightPadDataset2D(distance_dataset, pad_idx=0),
                "coord_target": RightPadDatasetCoord(coord_dataset, pad_idx=0),
                "smi_name": RawArrayDataset(smi_dataset),
            }

        net_input, target = one_dataset(raw_dataset, self.args.seed, self.args.seed)
        dataset = {"net_input": net_input, "target": target}
        dataset = NestedDictionaryDataset(dataset)
        if split in ["train", "train.small"]:
            dataset = EpochShuffleDataset(dataset, len(dataset), self.args.seed)
        self.datasets[split] = dataset

    def build_model(self, args):
        from unicore import models

        model = models.build_model(args, self)
        return model
