import gzip
import os, sys
import pickle
from tqdm import tqdm
from multiprocessing import Pool
import lmdb
from rdkit import Chem
import torch

split = torch.load("split_dict.pt")
train_index = split["train"]


os.system("rm -f label_3D.lmdb")

env_new = lmdb.open(
    "label_3D.lmdb",
    subdir=False,
    readonly=False,
    lock=False,
    readahead=False,
    meminit=False,
    max_readers=1,
    map_size=int(100e9),
)
txn_write = env_new.begin(write=True)

i = 0
with open("pcqm4m-v2-train.sdf", "r") as input:
    cur_content = ""
    for line in input:
        cur_content += line
        if line == "$$$$\n":
            ret = gzip.compress(pickle.dumps(cur_content))
            a = txn_write.put(int(train_index[i]).to_bytes(4, byteorder="big"), ret)
            i += 1
            cur_content = ""
            if i % 10000 == 0:
                txn_write.commit()
                txn_write = env_new.begin(write=True)
                print("processed {} molecules".format(i))

txn_write.commit()
env_new.close()
