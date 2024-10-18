# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import lmdb
import os
import pickle
from functools import lru_cache
import logging
import shutil
import time

logger = logging.getLogger(__name__)


# async ckp copy
def lmdb_data_copy_fun(src_path_dir, target_path_dir, epoch, split):
    db_path_src = os.path.join(src_path_dir, "{}_part_{}.lmdb".format(split, epoch))
    db_path_tgt = os.path.join(target_path_dir, "{}_part_{}.lmdb".format(split, epoch))

    if os.path.exists(db_path_tgt):
        return

    if not os.path.exists(db_path_src):
        logger.warning(f"please not that {db_path_src} not exists.")
        return

    shutil.copyfile(db_path_src, db_path_tgt)

    last_db_path_tgt = os.path.join(target_path_dir, "{}_part_{}.lmdb".format(split, epoch-2))
    if os.path.exists(last_db_path_tgt):
        os.remove(last_db_path_tgt)

    logger.info(f"finished async copy file from {db_path_src} to {db_path_tgt}")
    return

class LMDBDataset():
    def __init__(self, db_dir, split, epoch, max_epoch, lmdb_copy_thread=None, tmp_data_dir="/temp/"):
        self.db_dir = db_dir
        if not os.path.exists(tmp_data_dir):
            os.makedirs(tmp_data_dir, exist_ok=True)

        self.tmp_data_dir = tmp_data_dir
        self.lmdb_copy_thread = lmdb_copy_thread
        self.split = split
        self.max_epoch = max_epoch
        self.content = self.load_data(self.tmp_data_dir, epoch, split)

    def load_data(self, data_dir, epoch, split):
        self.db_path_tgt = os.path.join(data_dir, "{}_part_{}.lmdb".format(split, epoch-1))

        self.db_path_tgt_lock = os.path.join(data_dir, "{}_part_{}.lmdb.lock".format(split, epoch-1))
        if not os.path.exists(self.db_path_tgt):
            self.db_path_src = os.path.join(self.db_dir, "{}_part_{}.lmdb".format(split, epoch-1))
            if not os.path.exists(self.db_path_src):
                raise FileNotFoundError(f"{0} not found, please make sure the max-epoch were setting right.".format(self.db_path_src))
            os.system(f"touch {self.db_path_tgt_lock}")
            shutil.copyfile(self.db_path_src, self.db_path_tgt)
            logger.info(f"{self.db_path_tgt} not exist, copy file from {self.db_path_src}")
            os.system(f"rm -rf {self.db_path_tgt_lock}")
        else:
            while os.path.exists(self.db_path_tgt_lock):
                time.sleep(1)

        env = lmdb.open(
            self.db_path_tgt,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=256,
        )
        content = []
        with env.begin() as txn:
            self._keys = list(range(txn.stat()['entries']))
            for idx in self._keys:
                datapoint_pickled = txn.get( idx.to_bytes(4, byteorder="big") )
                content.append(datapoint_pickled)

        if self.lmdb_copy_thread is not None:
            self.lmdb_copy_thread.apply_async(lmdb_data_copy_fun,
                                            (self.db_dir, self.tmp_data_dir, epoch, self.split))
        return content

    def __len__(self):
        return len(self._keys)

    def set_epoch(self, epoch):
        if epoch is not None and epoch < self.max_epoch:
            self.content = self.load_data(self.tmp_data_dir, epoch, self.split)

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        datapoint_pickled = self.content[idx]
        data = pickle.loads(datapoint_pickled)
        return data
