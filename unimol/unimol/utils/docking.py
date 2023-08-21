# Copyright (c) DP Techonology, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import numpy as np
import pandas as pd
from multiprocessing import Pool
from tqdm import tqdm
import glob
import argparse
from docking_utils import (
    docking_data_pre,
    ensemble_iterations,
    print_results,
    rmsd_func,
)
import warnings

warnings.filterwarnings(action="ignore")


def result_log(dir_path):
    ### result logging ###
    output_dir = os.path.join(dir_path, "cache")
    rmsd_results = []
    for path in glob.glob(os.path.join(output_dir, "*.docking.pkl")):
        (
            mol,
            bst_predict_coords,
            holo_coords,
            bst_loss,
            smi,
            pocket,
            pocket_coords,
        ) = pd.read_pickle(path)
        rmsd = rmsd_func(holo_coords, bst_predict_coords, mol=mol)
        rmsd_results.append(rmsd)
    rmsd_results = np.array(rmsd_results)
    print_results(rmsd_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="docking")
    parser.add_argument(
        "--reference-file",
        type=str,
        default="./protein_ligand_binding_pose_prediction/test.lmdb",
        help="Location of the reference set",
    )
    parser.add_argument("--nthreads", type=int, default=40, help="num of threads")
    parser.add_argument(
        "--predict-file",
        type=str,
        default="./infer_pose/save_pose_test.out.pkl",
        help="Location of the prediction file",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="./protein_ligand_binding_pose_prediction",
        help="Location of the docking output path",
    )
    parser.add_argument(
        "--optimization-model",
        type=str,
        default="conformer",
        help="Optimize coordinates ('coordinate') or ligand internal torsions ('conformer')",
        choices=["coordinate", "conformer"],
    )
    args = parser.parse_args()

    raw_data_path, predict_path, dir_path, nthreads, model_choice = (
        args.reference_file,
        args.predict_file,
        args.output_path,
        args.nthreads,
        args.optimization_model,
    )
    tta_times = 10
    (
        mol_list,
        smi_list,
        pocket_list,
        pocket_coords_list,
        distance_predict_list,
        holo_distance_predict_list,
        holo_coords_list,
        holo_center_coords_list,
    ) = docking_data_pre(raw_data_path, predict_path)
    iterations = ensemble_iterations(
        mol_list,
        smi_list,
        pocket_list,
        pocket_coords_list,
        distance_predict_list,
        holo_distance_predict_list,
        holo_coords_list,
        holo_center_coords_list,
        tta_times=tta_times,
    )
    sz = len(mol_list) // tta_times
    new_pocket_list = pocket_list[::tta_times]
    output_dir = os.path.join(dir_path, "cache")
    os.makedirs(output_dir, exist_ok=True)

    def dump(content):
        pocket = content[3]
        output_name = os.path.join(output_dir, "{}.pkl".format(pocket))
        try:
            os.remove(output_name)
        except:
            pass
        pd.to_pickle(content, output_name)
        return True

    # skip step if repeat
    with Pool(nthreads) as pool:
        for inner_output in tqdm(pool.imap_unordered(dump, iterations), total=sz):
            if not inner_output:
                print("fail to dump")

    def single_docking(pocket_name):
        input_name = os.path.join(output_dir, "{}.pkl".format(pocket_name))
        output_name = os.path.join(output_dir, "{}.docking.pkl".format(pocket_name))
        output_ligand_name = os.path.join(
            output_dir, "{}.ligand.sdf".format(pocket_name)
        )
        try:
            os.remove(output_name)
        except:
            pass
        try:
            os.remove(output_ligand_name)
        except:
            pass

        cmd = "python ./unimol/utils/{}_model.py --input {} --output {} --output-ligand {}".format(
            model_choice, input_name, output_name, output_ligand_name
        )
        os.system(cmd)
        return True


    with Pool(nthreads) as pool:
        for inner_output in tqdm(
            pool.imap_unordered(single_docking, new_pocket_list), total=len(new_pocket_list)
        ):
            if not inner_output:
                print("fail to docking")

    result_log(args.output_path)
