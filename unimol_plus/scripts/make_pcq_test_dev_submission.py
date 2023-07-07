import numpy as np
import torch
import pickle
import os, sys
import glob
import pandas as pd

input_folder = sys.argv[1]
subset = sys.argv[2]


split = torch.load("./split_dict.pt")
valid_index = split[subset]


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
    # index 1 is the predicted position
    gap_pred = flatten(data, 2)

    df = pd.DataFrame(
        {
            "id": id,
            "data_index": id,
            "pred": gap_pred,
        }
    )
    df_grouped = df.groupby(["id"])
    df_mean = df_grouped.agg("mean")
    return df_mean.sort_values(by="data_index")


def save_test_submission(input_dict, dir_path: str, mode: str):
    """
    save test submission file at dir_path
    """
    assert "y_pred" in input_dict
    assert mode in ["test-dev", "test-challenge"]

    y_pred = input_dict["y_pred"]

    if mode == "test-dev":
        filename = os.path.join(dir_path, "y_pred_pcqm4m-v2_test-dev")
        assert y_pred.shape == (147037,)
    elif mode == "test-challenge":
        filename = os.path.join(dir_path, "y_pred_pcqm4m-v2_test-challenge")
        assert y_pred.shape == (147432,)

    assert isinstance(filename, str)
    assert isinstance(y_pred, np.ndarray) or isinstance(y_pred, torch.Tensor)

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.numpy()
    y_pred = y_pred.astype(np.float32)
    np.savez_compressed(filename, y_pred=y_pred)


df_mean = one_ckp(input_folder, subset)
pred_id = df_mean["data_index"].values
for i in range(len(valid_index)):
    assert valid_index[i] == pred_id[i]
pred = df_mean["pred"].values
print(pred.shape)
save_test_submission({"y_pred": pred}, "./", subset)
