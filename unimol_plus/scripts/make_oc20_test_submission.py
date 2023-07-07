import numpy as np
import torch
import pickle
import os, sys
import glob

input_folder = sys.argv[1]

subsets = ["test_id", "test_ood_ads", "test_ood_both", "test_ood_cat"]


def flatten(d, index):
    res = []
    for x in d:
        res.extend(x[index])
    return np.array(res)


def one_ckp(folder, subset):
    s = f"{folder}/" + subset + "*.pkl"
    files = sorted(glob.glob(s))
    data = []
    for file in files:
        with open(file, "rb") as f:
            try:
                data.extend(pickle.load(f))
            except Exception as e:
                print("Error in file: ", file)
                raise e

    id = flatten(data, 0)
    y_pred = flatten(data, 2)

    return np.array(id), np.array(y_pred)


submission_file = {}

for subset in subsets:
    id, y_pred = one_ckp(input_folder, subset)
    prefix = "_".join(subset.split("_")[1:])
    submission_file[prefix + "_ids"] = id
    submission_file[prefix + "_energy"] = y_pred

np.savez_compressed(sys.argv[2], **submission_file)
