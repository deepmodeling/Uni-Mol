import os
import lmdb
import pickle
from multiprocessing import Pool
from tqdm import tqdm
import numpy as np
import argparse
from multiprocessing import cpu_count

nthreads = cpu_count()


def inner_read(cursor):
    key, value = cursor
    data = pickle.loads(value)

    if "y_relaxed" in data:
        ret_data = {
            "cell": data["cell"].numpy().astype(np.float32),
            "pos": data["pos"].numpy().astype(np.float32),
            "atomic_numbers": data["atomic_numbers"].numpy().astype(np.int8),
            "tags": data["tags"].numpy().astype(np.int8),
            "pos_relaxed": data["pos_relaxed"].numpy().astype(np.float32),
            "y_relaxed": data["y_relaxed"],
            "sid": data["sid"],
        }
    else:
        ret_data = {
            "cell": data["cell"].numpy().astype(np.float32),
            "pos": data["pos"].numpy().astype(np.float32),
            "atomic_numbers": data["atomic_numbers"].numpy().astype(np.int8),
            "tags": data["tags"].numpy().astype(np.int8),
            "sid": data["sid"],
        }
    return data["sid"], pickle.dumps(ret_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="generate lmdb file")
    parser.add_argument("--input-path", type=str, help="initial oc20 data file path")
    parser.add_argument("--out-path", type=str, help="output path")
    parser.add_argument("--split", type=str, help="train/valid/test")
    args = parser.parse_args()

    train_list = ["train"]
    valid_list = ["val_id", "val_ood_ads", "val_ood_both", "val_ood_cat"]
    test_list = ["test_id", "test_ood_ads", "test_ood_both", "test_ood_cat"]
    path = args.input_path
    out_path = args.out_path

    if args.split == "train":
        name_list = train_list
    elif args.split == "valid":
        name_list = valid_list
    elif args.split == "test":
        name_list = test_list

    file_list = [os.path.join(path, name, "data.lmdb") for name in name_list]
    with Pool(nthreads) as pool:
        for filename, outname in zip(file_list, name_list):
            i = 0
            env = lmdb.open(
                filename,
                subdir=False,
                readonly=True,
                lock=False,
                readahead=True,
                meminit=False,
                max_readers=nthreads,
                map_size=int(1000e9),
            )
            txn = env.begin()

            out_dir = os.path.join(out_path, outname)
            if not os.path.exists(out_dir):
                os.mkdir(out_dir)
            outputfilename = os.path.join(out_dir, "data.lmdb")
            try:
                os.remove(outputfilename)
            except:
                pass

            env_new = lmdb.open(
                outputfilename,
                subdir=False,
                readonly=False,
                lock=False,
                readahead=False,
                meminit=False,
                max_readers=1,
                map_size=int(1000e9),
            )
            txn_write = env_new.begin(write=True)
            for inner_output in tqdm(
                pool.imap(inner_read, txn.cursor()), total=env.stat()["entries"]
            ):
                txn_write.put(f"{inner_output[0]}".encode("ascii"), inner_output[1])
                i += 1
                if i % 1000 == 0:
                    txn_write.commit()
                    txn_write = env_new.begin(write=True)
            txn_write.commit()
            env_new.close()
            env.close()
