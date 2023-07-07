# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import lmdb
import os
import numpy as np
import gzip
import pickle
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)


class LMDBDataset:
    def __init__(self, db_path, key_to_id=True, gzip=True):
        self.db_path = db_path
        assert os.path.isfile(self.db_path), "{} not found".format(self.db_path)
        env = self.connect_db(self.db_path)
        with env.begin() as txn:
            self._keys = list(txn.cursor().iternext(values=False))
        self.key_to_id = key_to_id
        self.gzip = gzip

    def connect_db(self, lmdb_path, save_to_self=False):
        env = lmdb.open(
            lmdb_path,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=256,
        )
        if not save_to_self:
            return env
        else:
            self.env = env

    def __len__(self):
        return len(self._keys)

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        if not hasattr(self, "env"):
            self.connect_db(self.db_path, save_to_self=True)
        key = self._keys[idx]
        datapoint_pickled = self.env.begin().get(key)
        if self.gzip:
            datapoint_pickled = gzip.decompress(datapoint_pickled)
        data = pickle.loads(datapoint_pickled)
        if self.key_to_id:
            data["id"] = int.from_bytes(key, "big")
        return data


class StackedLMDBDataset:
    def __init__(self, datasets):
        self._len = 0
        self.datasets = []
        self.idx_to_file = {}
        self.idx_offset = []
        for dataset in datasets:
            self.datasets.append(dataset)
            for i in range(len(dataset)):
                self.idx_to_file[i + self._len] = len(self.datasets) - 1
            self.idx_offset.append(self._len)
            self._len += len(dataset)

    def __len__(self):
        return self._len

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        file_idx = self.idx_to_file[idx]
        sub_idx = idx - self.idx_offset[file_idx]
        return self.datasets[file_idx][sub_idx]
