# Copyright (c) DP Techonology, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
import torch as th
import pandas as pd
from copy import deepcopy
from rdkit import Chem
from rdkit.Chem import AllChem
import pickle
import argparse
import warnings
from docking_utils import rmsd_func
from typing import List


warnings.filterwarnings(action="ignore")

# Utils

def rot_from_axis_angle(axis: th.Tensor, angle: th.Tensor) -> th.Tensor:
    """ ((...), 3), ((...),) -> ((...), 3, 3) """
    # ((...), D) -> ((...),)
    v1, v2, v3 = th.nn.functional.normalize(axis, dim=-1).unbind(dim=-1)
    zero = th.zeros_like(v1)
    # ((...),) -> ((...), 3, 3)
    cross_matrix = th.stack(
        (
            th.stack((zero, -v3, v2), dim=-1),
            th.stack((v3, zero, -v1), dim=-1),
            th.stack((-v2, v1, zero), dim=-1),
        ),
        dim=-2,
    )
    ide = th.eye(3, device=v1.device, dtype=v1.dtype).repeat(
        *(1,) * len(v1.shape), 1, 1
    )
    angle = angle.unsqueeze(dim=-1).unsqueeze(dim=-1)
    return (
            ide
            + th.sin(angle) * cross_matrix
            + (1 - th.cos(angle)) * (cross_matrix @ cross_matrix)
    )

def rot_from_euler(alpha_beta_gamma: th.Tensor) -> th.Tensor:
    """ rotation from euler angles. ((...), 3) -> ((...), 3, 3) """
    alpha, beta, gamma = alpha_beta_gamma.clone().unbind(dim=-1)
    zeros = th.zeros_like(alpha)
    Rx_tensor = th.stack((
        (alpha + 1) / (alpha + 1), zeros, zeros,
        zeros, th.cos(alpha), - th.sin(alpha),
        zeros, th.sin(alpha), th.cos(alpha)
    ), axis=-1).reshape(*alpha.shape, 3, 3)
    Ry_tensor = th.stack((
        th.cos(beta), zeros, - th.sin(beta),
        zeros, (beta + 1) / (beta + 1), zeros,
        th.sin(beta), zeros, th.cos(beta)
    ), axis=-1).reshape(*beta.shape, 3, 3)
    Rz_tensor = th.stack((
        th.cos(gamma), -th.sin(gamma), zeros,
        th.sin(gamma), th.cos(gamma), zeros,
        zeros, zeros, (gamma + 1) / (gamma + 1)
    ), axis=-1).reshape(*gamma.shape, 3, 3)

    R = (Rx_tensor @ Ry_tensor) @ Rz_tensor
    return R

def get_dihedral(
        c1: th.Tensor, c2: th.Tensor, c3: th.Tensor, c4: th.Tensor, eps: float = 1e-7
) -> th.Tensor:
    """ Dihedral angle in radians. atan2 formula from:
    https://en.wikipedia.org/wiki/Dihedral_angle#In_polymer_physics
    Inputs: c1, c2, c3, c4 are all ((...), 3,)
    * eps: float. small number to avoid division by zero.
    Outputs: ((...),) tensor
    """
    u1 = c2 - c1
    u2 = c3 - c2
    u3 = c4 - c3

    u2u3_cross = th.cross(u2, u3, dim=-1)
    u2norm = u2.square().sum(dim=-1, keepdim=True).add(eps).sqrt()

    return th.atan2(
        (u2norm * u1 * u2u3_cross).sum(-1),
        (th.cross(u1, u2, dim=-1) * u2u3_cross).sum(-1),
    )

def get_flexible_torsions(mol: Chem.Mol) -> th.Tensor:
    """ Gets a unique set of ligand torsions which are rotatable. Shape: (T, 4) """
    # get 3-hop connected atoms, directionally so no repeats
    dist_mat = th.from_numpy(Chem.GetDistanceMatrix(mol))
    # get rotatable bonds
    torsionSmarts = "[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]"
    torsionQuery = Chem.MolFromSmarts(torsionSmarts)
    matches = set(mol.GetSubstructMatches(torsionQuery))
    # get 3-hop connected atoms, directionally so no repeats
    i_, l_ = (dist_mat.triu() == 3).bool().nonzero().T.tolist()
    # Shortest path, where rotatable bond in the middle is a torsion
    flex_unique_torsions = []
    for i, l in zip(i_, l_):
        i, j, k, l = Chem.GetShortestPath(mol, i, l)
        if {(j, k), (k, j)}.intersection(matches):
            # torsion in the direction that leaves lesser atoms to later rotate: towards the periphery
            if (dist_mat[j] < dist_mat[k]).sum() > (dist_mat[j] > dist_mat[k]).sum():
                flex_unique_torsions.append([i, j, k, l])
            else:
                flex_unique_torsions.append([l, k, j, i])
    return th.tensor(flex_unique_torsions)


def rotate_along_axis(x: th.Tensor, origin: th.Tensor, axis: th.Tensor, angle: th.Tensor) -> th.Tensor:
    """ Rotates a point cloud around an axis given an origin
    Inputs:
    * x: ((...), N, 3)
    * origin: ((...), N_or_1, 3)
    * axis: ((...), 3)
    * angle: (,) th.Tensor
    Outputs: ((...), N, 3) rotated coordinates
    """
    rot_mat = rot_from_axis_angle(axis, angle)
    return th.einsum('...rc,...nc -> ...nr', rot_mat, x - origin) + origin


def update_dihedral(coords: th.Tensor, idxs: List[int], value: float, dist_mat: th.Tensor = None) -> th.Tensor:
    """Modifies a dihedral/torsion for a molecule with the given value.
    Analog to rdkit.Chem.rdMolTransforms.SetDihedralRad, but differentiable.
    WARNING! Assumes bond between j-k is rotatble
    Inputs:
    * coords: ((...), N, 3)
    * idxs: (4,) List or th.Tensor of dtype th.long. indexes to define the torsion
    * value: float or th.Tensor of single value or ((...),). New value for the torsion (in radians)
    * dist_mat: (N, N) length of shortest path for each i-j.
    Outputs: ((...), N, 3) updated coords
    """
    i, j, k, l = idxs
    if not isinstance(value, th.Tensor):
        value = th.tensor(value, dtype=coords.dtype, device=coords.device)

    # atoms whose coords will be updated - closer to k than j
    mask_rotate = dist_mat[k] < dist_mat[j]

    # amount to rotate is the difference between current and desired
    coords[..., mask_rotate, :] = rotate_along_axis(
        x=coords[..., mask_rotate, :],
        origin=coords[..., [j], :],
        axis=coords[..., k, :] - coords[..., j, :],
        angle=value - get_dihedral(*coords[..., idxs, :].unbind(dim=-2)),
    )
    return coords


# Docking functions

def single_SF_loss(
    predict_coords: th.Tensor,
    pocket_coords: th.Tensor,
    cross_distance_predict: th.Tensor,
    self_distance_predict: th.Tensor,
    dist_threshold: float = 4.5,
    cross_dist_weight: float = 1.0,
    self_dist_weight: float = 2.0,
    reduce_batch: bool = True,
):
    """ Calculates loss function
    Args:
        predict_coords: ((...), N, 3) predicted molecule coordinates
        pocket_coords:  ((...), P, 3) pocket coordinates
        cross_distance_predict: ((...), N, P) predicted (molecule-pocket) distance matrix
        self_distance_predict: ((...), N, N) predicted (molecule-molecule) distance
        dist_threshold: max dist to consider molecule-pocket interactions in the loss
        cross_dist_weight: weight of cross distance loss
        self_dist_weight: weight of self distance loss
        reduce_batch: whether to reduce the batch dimension

    Returns:
        cross_dist_score: cross distance score. scalar. numpy
        dist_score: distance score. scalar. numpy
        clash_score. clash score. informative. scalar. numpy.
        loss: loss value. scalar. has gradients
    """
    # ((...), N, 1, 3) - ((...), 1, P, 3) -> ((...), N, P)
    cross_dist = (predict_coords[..., None, :] - pocket_coords[..., None, :, :]).norm(dim=-1)
    # ((...), N, 1, 3) - ((...), 1, N, 3) -> ((...), N, N)
    self_dist = (predict_coords[..., None, :] - predict_coords[..., None, :, :]).norm(dim=-1)
    # only consider local molecule-pocket interactions
    dist_mask = cross_distance_predict < dist_threshold
    # ((...), N, N) -> ((...),)
    cross_dist_score = ((cross_dist - cross_distance_predict)**2 * dist_mask).sum() / dist_mask.sum(dim=(-1, -2))
    dist_score = ((self_dist - self_distance_predict) ** 2).mean(dim=(-1, -2))
    # weight different loss terms
    loss = cross_dist_score * cross_dist_weight + dist_score * self_dist_weight
    # penalize clashes - informative
    clash_pl_score = ((cross_dist - 3.).clamp(max=0) * 5.).square()
    clash_pl_score = clash_pl_score.sum(dim=(-1, -2)) / dist_mask.sum(dim=(-1, -2))
    if reduce_batch:
        return cross_dist_score.detach().mean().numpy(), dist_score.detach().mean().numpy(), 0., loss.mean()
    return cross_dist_score.detach().numpy(), dist_score.detach().numpy(), clash_pl_score.detach().numpy(), loss



def dock_with_gradient(
    coords: np.ndarray,
    pocket_coords: np.ndarray,
    distance_predict_tta: np.ndarray,
    holo_distance_predict_tta: np.ndarray,
    mol: Chem.Mol,
    conf_coords: np.ndarray,
    loss_func=single_SF_loss,
    holo_coords: np.ndarray = None,
    iterations: int =400,
    early_stoping: int = 5,
):
    """ Docking with gradient descent, optimizing the conformer.

    Args:
        coords: (N, 3) initial molecule coordinates
        pocket_coords: (P, 3) pocket coordinates
        distance_predict_tta: (?, T, N, P) predicted (molecule-pocket) distance matrix
        holo_distance_predict_tta: (?, T, N, N) predicted (molecule-molecule) distance matrix
        mol: rdkit molecule
        conf_coords: (?, N, 3) initial molecule conformers coordinates
        loss_func: function to calculate loss
        holo_coords: (?, T, N, 3) holo molecule coordinates
        iterations: max number of iterations
        early_stoping: stop if loss does not improve for this number of iterations

    Returns:
        bst_coords: (N, 3) optimized molecule coordinates
        bst_loss: loss value. scalar. has gradients
        bst_meta_info: dict with additional info
    """
    bst_loss, bst_coords, bst_meta_info = 10000.0, coords, None
    for i, (distance_predict, holo_distance_predict) in enumerate(
        zip(distance_predict_tta, holo_distance_predict_tta)
    ):
        new_coords = deepcopy(coords)
        _coords, _loss, _meta_info = single_dock_with_gradient(
            new_coords,
            pocket_coords,
            distance_predict,
            holo_distance_predict,
            mol=mol,
            conf_coords=deepcopy(np.array(conf_coords)),
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


def kabsch(x: th.Tensor, y: th.Tensor, weight: Optional[th.Tensor] = None) -> th.Tensor:
    """ Aligns x onto y. x, y are ((...), N, 3) tensors. Weights is ((...), N)
    If rotation fails, at least bring to X to Y's COM
    """
    if weight is None:
        weight = th.ones_like(x[..., 0])

    weight = weight / weight.sum(dim=-1, keepdim=True)
    x_mean = (x * weight[..., None]).sum(dim=-2, keepdim=True)
    y_mean = (y * weight[..., None]).sum(dim=-2, keepdim=True)
    x = x - x_mean
    y = y - y_mean

    # if rotation fails (SVD might fail if matrix is ill-behaved), just bring to same COM
    try:
        # ((...), N, 3) -> ((...), 1, 3, 3)
        cov = th.einsum("...ni,...nj->...ij", x, y * weight[..., None])[..., None, :, :]
        u, s, v = th.linalg.svd(cov)
        # Flip the sign of bottom row of each matrix if det product < 0
        det = th.det(v) * th.det(u)
        u_flip = th.ones_like(u)
        u_flip[det < 0, :, -1] = -1.0
        u = u * u_flip
        rot = u @ v
        # align to rotation
        x = rot @ x
    except:
        pass
    return x + y_mean



def single_dock_with_gradient(
    coords: np.ndarray,
    pocket_coords: np.ndarray,
    distance_predict: np.ndarray,
    holo_distance_predict: np.ndarray,
    mol: Chem.Mol,
    conf_coords: np.ndarray,
    loss_func=single_SF_loss,
    holo_coords: np.ndarray = None,
    iterations: int = 20000,
    early_stoping: int = 5,
):
    """ Strategy: create multiple conformers, align to coordinates, optimize the conformer
    to minimize the loss function. Then pick the conformer with the lowest loss

    Args:
        coords: (N, 3) initial molecule coordinates
        pocket_coords: (P, 3) pocket coordinates
        distance_predict: (N, P) predicted (molecule-pocket) distance matrix
        holo_distance_predict: (N, N) predicted (molecule-molecule) distance matrix
        mol: rdkit mol object. to extract graph connectivity
        conf_coords: (B, N, 3) initial conformer coordinates
        loss_func: function to calculate loss
        holo_coords: (N, 3) holo molecule coordinates
        iterations: max number of iterations
        early_stoping: stop if loss does not improve for this number of iterations

    Returns:
        coords: (N, 3) optimized molecule coordinates
        loss: loss value. scalar. numpy
        meta_info: dict with additional info
    """
    # convert to torch
    coords = th.from_numpy(coords).float()
    pocket_coords = th.from_numpy(pocket_coords).float()
    distance_predict = th.from_numpy(distance_predict).float()
    holo_distance_predict = th.from_numpy(holo_distance_predict).float()
    conf_coords = th.from_numpy(conf_coords).float()

    if holo_coords is not None:
        holo_coords = th.from_numpy(holo_coords).float()

    # prepare optimization params
    num_conformers = conf_coords.shape[0]
    torsion_idxs = get_flexible_torsions(mol)  # (T, 4)
    graph_dist_mat = th.from_numpy(Chem.GetDistanceMatrix(mol)).long()  # (N, N)

    # (B, 3)
    euler = th.randn(num_conformers, 3) * 1e-3
    # init translation approx at ligand Center of Mass: (B, 1, 3)
    trans = th.randn(num_conformers, 1, 3) + coords.mean(dim=-2)[None, None]
    # (B, T)
    if torsion_idxs.shape[-1] > 0:
        torsions = get_dihedral(*conf_coords[..., torsion_idxs, :].unbind(dim=-2))
        torsions += th.randn_like(torsions) * 1e-3
    else:
        torsions = th.zeros(num_conformers, 0)

    # add batch dim to labels
    pocket_coords = pocket_coords[None].repeat(num_conformers, 1, 1)
    distance_predict = distance_predict[None].repeat(num_conformers, 1, 1)
    holo_distance_predict = holo_distance_predict[None].repeat(num_conformers, 1, 1)

    # set gradients and optimizer
    euler.requires_grad = True
    trans.requires_grad = True
    torsions.requires_grad = True

    optimizer = th.optim.LBFGS(params=[euler, trans, torsions], lr=0.5)
    bst_loss, times = 10000.0, 0
    for i in range(iterations):
        def closure():
            optimizer.zero_grad()
            # parametrize ligand with 6+K
            aux_coords = conf_coords.detach().clone() + trans
            # frame update
            com = aux_coords.mean(dim=-2, keepdim=True)
            rot = rot_from_euler(euler)
            aux_coords = th.einsum('...rc,...nc->...nr', rot, aux_coords - com) + com
            pre_aux_coords = aux_coords.clone()
            # torsion update + kabsch -> makes 6 & T orthogonal in the tangent space
            for t, vals in zip(torsion_idxs, torsions.unbind(dim=-1)):
                aux_coords = update_dihedral(coords=aux_coords, idxs=t.tolist(), value=vals, dist_mat=graph_dist_mat)
            aux_coords = kabsch(aux_coords, pre_aux_coords)

            _, _, _, loss = loss_func(
                aux_coords, pocket_coords, distance_predict, holo_distance_predict
            )
            loss.backward()
            return loss

        loss = optimizer.step(closure)
        # print(f"Iter: {i} and loss: {loss}")
        if loss.item() < bst_loss:
            bst_loss = loss.item()
            times = 0
        else:
            times += 1
            if times > early_stoping:
                break

    # pick the conformer with lowest loss
    aux_coords = conf_coords.detach().clone() + trans
    # frame update
    com = aux_coords.mean(dim=-2, keepdim=True)
    rot = rot_from_euler(euler)
    aux_coords = th.einsum('...rc,...nc->...nr', rot, aux_coords - com) + com
    pre_aux_coords = aux_coords.clone()
    # torsion update + kabsch -> makes 6 & T orthogonal in the tangent space
    for t, vals in zip(torsion_idxs, torsions.unbind(dim=-1)):
        aux_coords = update_dihedral(coords=aux_coords, idxs=t.tolist(), value=vals, dist_mat=graph_dist_mat)
    aux_coords = kabsch(aux_coords, pre_aux_coords)

    cross_score, self_score, clash_score, loss = loss_func(
        aux_coords, pocket_coords, distance_predict, holo_distance_predict, reduce_batch=False
    )
    best_idx = loss.argmax(dim=-1).item()
    return aux_coords[best_idx].detach().numpy(), loss[best_idx].detach().numpy(), (
        cross_score[best_idx], self_score[best_idx], clash_score[best_idx]
    )


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


def single_docking(input_path: str, output_path: str, output_ligand_path: str):
    """ Performs docking based on UniMol predictions.

    Args:
        input_path: path to the input file
        output_path: path to the output file
        output_ligand_path: path to the output ligand file
        sym_rmsd: whether to use symmetric RMSD: consider best of symmetric atoms

    Returns:
        True
    """
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
            distance_predict_tta[i][None],
            holo_distance_predict_tta[i][None],
            mol=mol,
            conf_coords=init_coords_tta[i][None],
            holo_coords=holo_coords,
            loss_func=single_SF_loss,
        )
        if loss < bst_loss:
            bst_loss = loss
            bst_predict_coords = predict_coords
            bst_meta_info = meta_info

    _rmsd = round(rmsd_func(holo_coords, bst_predict_coords, mol=mol), 4)
    _cross_score = round(float(bst_meta_info[0]), 4)
    _self_score = round(float(bst_meta_info[1]), 4)
    _clash_score = round(float(bst_meta_info[2]), 4)
    print(f"{pocket}-{smi}-RMSD:{_rmsd}-CROSSSCORE:{_cross_score}-SELFSCORE:{_self_score}-CLASHSCORE:{_clash_score}")
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
    th.set_num_threads(1)
    th.manual_seed(0)
    parser = argparse.ArgumentParser(description="Docking with gradient")
    parser.add_argument("--input", type=str, help="input file.")
    parser.add_argument("--output", type=str, default=None, help="output path.")
    parser.add_argument(
        "--output-ligand", type=str, default=None, help="output ligand sdf path."
    )
    args = parser.parse_args()

    single_docking(args.input, args.output, args.output_ligand)
