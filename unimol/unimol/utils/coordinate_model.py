# Copyright (c) DP Techonology, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import torch
import pandas as pd
from rdkit import Chem
import pickle
import argparse
from docking_utils import rmsd_func
import warnings

warnings.filterwarnings(action="ignore")


def single_SF_loss(
    predict_coords,
    pocket_coords,
    distance_predict,
    holo_distance_predict,
    dist_threshold=4.5,
):
    dist = torch.norm(predict_coords.unsqueeze(1) - pocket_coords.unsqueeze(0), dim=-1)
    holo_dist = torch.norm(
        predict_coords.unsqueeze(1) - predict_coords.unsqueeze(0), dim=-1
    )
    distance_mask = distance_predict < dist_threshold
    cross_dist_score = (
        (dist[distance_mask] - distance_predict[distance_mask]) ** 2
    ).mean()
    dist_score = ((holo_dist - holo_distance_predict) ** 2).mean()
    loss = cross_dist_score * 1.0 + dist_score * 5.0
    return loss


def scoring(
    predict_coords,
    pocket_coords,
    distance_predict,
    holo_distance_predict,
    dist_threshold=4.5,
):
    predict_coords = predict_coords.detach()
    dist = torch.norm(predict_coords.unsqueeze(1) - pocket_coords.unsqueeze(0), dim=-1)
    holo_dist = torch.norm(
        predict_coords.unsqueeze(1) - predict_coords.unsqueeze(0), dim=-1
    )
    distance_mask = distance_predict < dist_threshold
    cross_dist_score = (
        (dist[distance_mask] - distance_predict[distance_mask]) ** 2
    ).mean()
    dist_score = ((holo_dist - holo_distance_predict) ** 2).mean()
    return cross_dist_score.numpy(), dist_score.numpy()


def dock_with_gradient(
    coords,
    pocket_coords,
    distance_predict_tta,
    holo_distance_predict_tta,
    loss_func=single_SF_loss,
    holo_coords=None,
    iterations=20000,
    early_stoping=5,
):
    bst_loss, bst_coords, bst_meta_info = 10000.0, coords, None
    for i, (distance_predict, holo_distance_predict) in enumerate(
        zip(distance_predict_tta, holo_distance_predict_tta)
    ):
        new_coords = copy.deepcopy(coords)
        _coords, _loss, _meta_info = single_dock_with_gradient(
            new_coords,
            pocket_coords,
            distance_predict,
            holo_distance_predict,
            loss_func=loss_func,
            holo_coords=holo_coords,
            iterations=iterations,
            early_stoping=early_stoping,
        )
        if bst_loss > _loss:
            bst_coords = _coords
            bst_loss = _loss
            bst_meta_info = _meta_info
    return bst_coords, bst_loss, bst_meta_info


def single_dock_with_gradient(
    coords,
    pocket_coords,
    distance_predict,
    holo_distance_predict,
    loss_func=single_SF_loss,
    holo_coords=None,
    iterations=20000,
    early_stoping=5,
):
    coords = torch.from_numpy(coords).float()
    pocket_coords = torch.from_numpy(pocket_coords).float()
    distance_predict = torch.from_numpy(distance_predict).float()
    holo_distance_predict = torch.from_numpy(holo_distance_predict).float()

    if holo_coords is not None:
        holo_coords = torch.from_numpy(holo_coords).float()

    coords.requires_grad = True
    optimizer = torch.optim.LBFGS([coords], lr=1.0)
    bst_loss, times = 10000.0, 0
    for i in range(iterations):

        def closure():
            optimizer.zero_grad()
            loss = loss_func(
                coords, pocket_coords, distance_predict, holo_distance_predict
            )
            loss.backward()
            return loss

        loss = optimizer.step(closure)
        if loss.item() < bst_loss:
            bst_loss = loss.item()
            times = 0
        else:
            times += 1
            if times > early_stoping:
                break

    meta_info = scoring(coords, pocket_coords, distance_predict, holo_distance_predict)
    return coords.detach().numpy(), loss.detach().numpy(), meta_info


def set_coord(mol, coords):
    for i in range(coords.shape[0]):
        mol.GetConformer(0).SetAtomPosition(i, coords[i].tolist())
    return mol


def add_coord(mol, xyz):
    x, y, z = xyz
    conf = mol.GetConformer(0)
    pos = conf.GetPositions()
    pos[:, 0] += x
    pos[:, 1] += y
    pos[:, 2] += z
    for i in range(pos.shape[0]):
        conf.SetAtomPosition(
            i, Chem.rdGeometry.Point3D(pos[i][0], pos[i][1], pos[i][2])
        )
    return mol


def single_docking(input_path, output_path, output_ligand_path):
    content = pd.read_pickle(input_path)
    (
        init_coords_tta,
        mol,
        smi,
        pocket,
        pocket_coords,
        distance_predict_tta,
        holo_distance_predict_tta,
        holo_coords,
        holo_cener_coords,
    ) = content
    sample_times = len(init_coords_tta)
    bst_predict_coords, bst_loss, bst_meta_info = None, 1000.0, None
    for i in range(sample_times):
        init_coords = init_coords_tta[i]
        predict_coords, loss, meta_info = dock_with_gradient(
            init_coords,
            pocket_coords,
            distance_predict_tta,
            holo_distance_predict_tta,
            holo_coords=holo_coords,
            loss_func=single_SF_loss,
        )
        if loss < bst_loss:
            bst_loss = loss
            bst_predict_coords = predict_coords
            bst_meta_info = meta_info

    _rmsd = round(rmsd_func(holo_coords, bst_predict_coords, mol), 4)
    _cross_score = round(float(bst_meta_info[0]), 4)
    _self_score = round(float(bst_meta_info[1]), 4)
    print(f"{pocket}-{smi}-RMSD:{_rmsd}-{_cross_score}-{_self_score}")
    mol = Chem.RemoveHs(mol)
    mol = set_coord(mol, bst_predict_coords)

    if output_path is not None:
        with open(output_path, "wb") as f:
            pickle.dump(
                [mol, bst_predict_coords, holo_coords, bst_loss, smi, pocket, pocket_coords],
                f,
            )
    if output_ligand_path is not None:
        mol = add_coord(mol, holo_cener_coords.numpy())
        Chem.MolToMolFile(mol, output_ligand_path)

    return True


if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.manual_seed(0)
    parser = argparse.ArgumentParser(description="Docking with gradient")
    parser.add_argument("--input", type=str, help="input file.")
    parser.add_argument("--output", type=str, default=None, help="output path.")
    parser.add_argument(
        "--output-ligand", type=str, default=None, help="output ligand sdf path."
    )
    args = parser.parse_args()

    single_docking(args.input, args.output, args.output_ligand)
