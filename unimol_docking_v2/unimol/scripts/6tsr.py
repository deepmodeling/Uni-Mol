# Copyright (c) DP Techonology, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" This script serves the purpose of refining ML docking outputs.
It removes any ligand internal geometry issue caused by direct cartesian
optimization by superimposing a conformer optimized with 6+T+S format
(See UMD-fit, NeurIPS 2023 GenBio workshop for ref)
And optionally removes steric clashes wrt a given pocket(+ions(+cofactors(+waters)))
while keeping coordinates close to original prediction through a harmonic potential
"""

import copy
import torch as th
from rdkit import Chem
from rdkit.Chem import rdMolAlign as MA
from rdkit.Chem import AllChem
import argparse
import warnings
import numpy as np
from typing import Callable, Optional, Tuple, List

warnings.filterwarnings(action="ignore")

# Utils - adapted from public Unimol: unimol.utils.conf_gen_cal_metrics

def single_conf_gen(tgt_mol, num_confs=1000, seed=0):
    mol = copy.deepcopy(tgt_mol)
    mol = Chem.AddHs(mol)
    ps = AllChem.ETKDGv3()
    ps.randomSeed = seed
    ps.numThreads = 0
    ps.useRandomCoords = True
    AllChem.EmbedMolecule(mol, ps)
    try:
        AllChem.MMFFOptimizeMolecule(mol, confId=0)
    except:
        pass
    mol = Chem.RemoveHs(mol)
    return mol


# Utils - from public Unimol: unimol.utils.docking_utils

def rmsd_func(holo_coords: np.ndarray, predict_coords: np.ndarray, mol: Optional[Chem.Mol] = None) -> float:
    """ Symmetric RMSD for molecules. """
    if predict_coords is not np.nan:
        sz = holo_coords.shape
        if mol is not None:
            # get stereochem-unaware permutations: (P, N)
            base_perms = np.array(mol.GetSubstructMatches(mol, uniquify=False))
            # filter for valid stereochem only
            chem_order = np.array(list(Chem.rdmolfiles.CanonicalRankAtoms(mol, breakTies=False)))
            perms_mask = (chem_order[base_perms] == chem_order[None]).sum(-1) == mol.GetNumAtoms()
            base_perms = base_perms[perms_mask]
            noh_mask = np.array([a.GetAtomicNum() != 1 for a in mol.GetAtoms()])
            # (N, 3), (N, 3) -> (P, N, 3), ((), N, 3) -> (P,) -> min((P,))
            best_rmsd = np.inf
            for perm in base_perms:
                rmsd = np.sqrt(np.sum((predict_coords[perm[noh_mask]] - holo_coords) ** 2) / sz[-2])
                if rmsd < best_rmsd:
                    best_rmsd = rmsd

            rmsd = best_rmsd
        else:
            rmsd = np.sqrt(np.sum((predict_coords - holo_coords) ** 2) / sz[-2])
        return rmsd
    return 1000.0


# Utils - from public Unimol: unimol.utils.conformer_model

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


def kabsch(
    x: th.Tensor,
    y: th.Tensor,
    weights: Optional[th.Tensor] = None,
) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
    """Kabsch alignment of X into Y (solution to least squares of point cloud rototranslation).
    Assumes X,Y are both ((...), N, D) - usually ((...), N, 3)
    Inputs:
    * x: ((...), N, D) th.Tensor
    * y: ((...), N, D) th.Tensor
    * weights: (..., N) th.Tensor. Optional. Only 0s and 1s to keep the algo meaningful
    Outputs:
    * x_: (..., N, D) th.Tensor
    """
    # (create and) ensure weights tensor is same shape as point clouds
    if weights is None:
        weights = th.ones_like(x[..., 0])

    try:
        # (..., n) -> (..., n, 1)
        weights = (weights / weights.sum(dim=-1, keepdim=True))[..., None]
        # calculate COM (...nd, ...n -> ... () d) and center
        x_mean = (x * weights).sum(dim=-2, keepdim=True)
        y_mean = (y * weights).sum(dim=-2, keepdim=True)
        x_ = x - x_mean
        y_ = y - y_mean

        # Optimal rotation matrix via SVD of covariance matrix (..., 3, 3)
        C = th.einsum("... n i, ... n j -> ... i j", y_ * weights, x_)
        U, S, V = th.linalg.svd(C)
        # Flip the sign of bottom row of each matrix if det product < 0
        det = th.det(V) * th.det(U)
        U_flip = th.ones_like(U)
        if det < 0:
            U_flip[:, -1] = -1.0
        U = U * U_flip

        # beware! th.linalg.svd(C).V.t() == th.svd(C).V
        R = U @ V
        # Note: R @ x == x @ R^(-1), and R^(-1) == Rt
        rt_x = th.einsum('...rc, ...nc -> ...nr', R, x_) + y_mean
    except:
        rt_x = x_ + y_mean
    return rt_x


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
    flex_unique_torsions =[]
    selected = set()
    for i, l in zip(i_, l_):
        i, j, k, l = Chem.GetShortestPath(mol, i, l)
        if {(j, k), (k, j)}.intersection(matches) and not {(j, k), (k, j)}.intersection(selected):
            # torsion in the direction that leaves lesser atoms to later rotate: towards the periphery
            if (dist_mat[j] < dist_mat[k]).sum() > (dist_mat[j] > dist_mat[k]).sum():
                flex_unique_torsions.append([i, j, k, l])
            else:
                flex_unique_torsions.append([l, k, j, i])
            selected.add((j, k))
            selected.add((k, j))

    return th.tensor(flex_unique_torsions)


# Main computation
def opt_6t(
    this_mol: Chem.Mol,
    this_coords: th.Tensor,
    other_coords: Optional[th.Tensor] = None,
    custom_func: Optional[Callable] = None,
    opt_steps: int = 350,
    verbose: bool = False,
    lr: float = 0.5,
) -> th.Tensor:
    """Flexible alignment of two ligands, only difference should come from
    cycles, non-rotatable structures and angles, etc.
    Assumes both molecules are the same (order included), except for the torsions.
    Parametrizes ligand as optimizable overall (rotation, translation) and torsions
    Should also serve as a template to be adapted for an arbitrary loss function
    which may not need a template ligand, or may need additional inputs.
    Inputs:
    * other: Optional. Ligand. Ligand to copy torsions from.
    * custom_func: Optional. Callable. Custom function to optimize (mutually exclusive with other)
    * opt_steps: int. Optimization steps to match ligand RMSD. 6+T parametrization
    * verbose: bool. Verbose optimization.
    Outputs: Ligand aligned to other
    """
    # (T, 4)
    torsion_idxs = get_flexible_torsions(this_mol)
    k_deg_adj_mat = th.from_numpy(Chem.GetDistanceMatrix(this_mol))

    # kabsch is optimal between 2 point clouds if no torsions
    if len(torsion_idxs) == 0 and other_coords is not None:
        coords = kabsch(this_coords, other_coords)
        return coords

    if len(torsion_idxs):
        # (T, 4) -> 4 x (T,)
        i, j, k, l = torsion_idxs.unbind(dim=-1)
        # (T, N, 1)
        torsion_masks = (k_deg_adj_mat[k] < k_deg_adj_mat[j])[..., None]

        # order from smallest to biggest so that we can update in order
        torsion_order = torsion_masks.sum(dim=(-1, -2)).argsort(dim=-1)
        torsion_idxs = torsion_idxs[torsion_order]
        torsion_masks = torsion_masks[torsion_order].to(this_coords)
        i, j, k, l = torsion_idxs.unbind(dim=-1)

        if custom_func is None:
            torsions = get_dihedral(*other_coords[torsion_idxs].unbind(dim=-2))
        elif custom_func is not None:
            torsions = get_dihedral(*this_coords[torsion_idxs].unbind(dim=-2))
    else:
        torsions = th.zeros(0, 4, dtype=th.double)

    def mse(x: th.Tensor, y: th.Tensor) -> th.Tensor:
        return x, (x - y).square().sum(dim=-1).mean(dim=-1)

    def update_6t(
        c: th.Tensor, _t: th.Tensor, _r: th.Tensor, t: th.Tensor
    ) -> th.Tensor:
        """Updates 6+T parameters to minimize RMSD between two ligands.
        Inputs:
        * c: ((...), N, 3) th.Tensor. Coordinates.
        * _t: ((...), 1, 3,) th.Tensor. Translation.
        * _r: ((...), 1, 3,) th.Tensor. Rotation.
        * t: ((...), T) th.Tensor. Torsions.
        Outputs: (N, 3) th.Tensor. Updated coordinates.
        """
        coords = c.detach()
        # 6 update - only if no model coordinates (ex. arbitrary loss function)
        if other_coords is None:
            coords = coords + trans
            com = coords.mean(dim=-2, keepdim=True)
            rot = rot_from_euler(_r)
            coords = th.einsum("...rc,...nc->...nr", rot, coords - com) + com
        else:
            coords = kabsch(coords, other_coords)

        # print("after kabsch mse", mse(coords, other_coords)[1].item() ** 0.5)

        # T update
        coords_pre = coords.clone()
        if len(torsion_idxs):
            axis = coords[k] - coords[j]
            delta_t = t - get_dihedral(coords[i], coords[j], coords[k], coords[l])
            # (T, 3, 3)
            torsion_rots = rot_from_axis_angle(axis, delta_t)
            for k_, t_rot, t_mask in zip(k, torsion_rots, torsion_masks):
                offset = coords[k_][..., None, :]
                rot_coords = th.einsum("...rc,...nc->...nr", t_rot, coords - offset) + offset
                coords = coords * (1 - t_mask) + t_mask * rot_coords
            # make 6 & T orthogonal in the tangent space
            coords = kabsch(coords, coords_pre)
        return coords

    if other_coords is not None:
        coords, msd = mse(this_coords, other_coords)
    else:
        coords, msd = custom_func(this_coords)


    exp_coords = coords.clone()
    # init at identity and optimize to minimize RMSD
    trans = th.randn(1, 3) * 1e-4
    angles = th.randn(3) * 1e-4

    trans.requires_grad = True
    angles.requires_grad = True
    torsions.requires_grad = True
    opt = th.optim.LBFGS(lr=lr, params=[torsions, trans, angles])

    iter_no_improvement, last_msd = 0, th.inf
    for o in range(opt_steps):
        def closure():
            opt.zero_grad()
            coords = update_6t(exp_coords, trans, angles, torsions)
            if other_coords is not None:
                coords, loss = mse(coords, other_coords)
            elif custom_func is not None:
                coords, loss = custom_func(coords)
            loss = loss.mean()
            loss.backward()
            return loss

        loss = opt.step(closure)

        iter_no_improvement += 1
        if loss.amin() < last_msd:
            last_msd = loss.amin().detach()
            iter_no_improvement = 0
        if iter_no_improvement > 5:
            if verbose:
                print(f"Early Stopped at RMSD={loss.amin() ** 0.5}.")
            break

    # apply to output
    coords = update_6t(exp_coords, trans, angles, torsions)

    return coords

def set_coord(mol: Chem.Mol, coords: th.Tensor) -> Chem.Mol:
    if mol.GetNumConformers() == 0:
        mol.AddConformer(Chem.Conformer(mol.GetNumAtoms()))
    coords = coords.tolist()
    for i in range(len(coords)):
        mol.GetConformer(0).SetAtomPosition(i, coords[i])
    return mol


def get_coord(mol: Chem.Mol) -> th.Tensor:
    conf = mol.GetConformer(0)
    pos = th.as_tensor(conf.GetPositions(), dtype=th.float32)
    return pos


# Utils for clash avoidance

def create_subset_molecule(
    original_molecule: Chem.Mol,
    indices_to_keep: Optional[List[int]] = None,
    mask: Optional[List[bool]] = None
) -> Chem.Mol:
    """
    Create a new molecule containing a subset of atoms from the original molecule.
    Args:
        original_molecule (rdkit.Chem.Mol): The original molecule.
        indices_to_keep (list): List of atom indices to keep.
        mask (list): List of booleans indicating which atoms to keep.

    Returns:
        rdkit.Chem.Mol: The subset molecule.
    """
    assert indices_to_keep is not None or mask is not None, f""
    original_molecule = copy.deepcopy(original_molecule)
    original_molecule.SetProp("is_clean", "True")
    subset_molecule = Chem.RWMol()

    # Copy selected atoms and bonds to the subset molecule
    indices_to_keep = set(indices_to_keep) if indices_to_keep is not None else None
    prev2new = {}
    news = 0
    pos = get_coord(original_molecule)

    for i in range(original_molecule.GetNumAtoms()):
        if (indices_to_keep is not None and i in indices_to_keep) or (mask is not None and mask[i]):
            atom = original_molecule.GetAtomWithIdx(i)
            subset_molecule.AddAtom(atom)
            prev2new[i] = news
            news += 1

    for bond in original_molecule.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        if (indices_to_keep is not None and i in indices_to_keep and j in indices_to_keep) or \
                (mask is not None and mask[i] and mask[j]):
            subset_molecule.AddBond(prev2new[i], prev2new[j], bond.GetBondType())

    final_mol = subset_molecule.GetMol()
    final_mol.SetProp("is_clean", "True")

    # Copy 3D coordinates from the original molecule to the subset molecule
    new_coords = th.zeros(final_mol.GetNumAtoms(), 3)
    for prev, new in prev2new.items():
        new_coords[new] = pos[prev]

    return set_coord(final_mol, new_coords)


def clash_relax(this_mol: Chem.Mol, this_coords: th.Tensor, env_mol: Chem.Mol, tol: float = 0.) -> th.Tensor:
    """Modifies a ligand to avoid clashes with the surrounding environment.
    Inputs:
    * self: Ligand. Center of coordinates to optimize from, while removing steric clashes.
    * env: Chem.Mol. Environment to avoid clashes with.
    * tol: float. Maximum clash loss value to allow before starting to optimize.
    Outputs: Ligand. Optimized ligand to avoid steric clashes while keeping close to initial coordinates.
    """
    lig_coords = this_coords.clone()
    env_coords = get_coord(env_mol)

    covr_lig = th.tensor([
        AllChem.GetPeriodicTable().GetRcovalent(a.GetSymbol())
        for a in this_mol.GetAtoms()
    ], dtype=th.float)
    vdwr_lig = th.tensor([
        AllChem.GetPeriodicTable().GetRvdw(a.GetSymbol())
        for a in this_mol.GetAtoms()
    ], dtype=th.float)
    vdwr_pocket = th.tensor([
        AllChem.GetPeriodicTable().GetRvdw(a.GetSymbol())
        for a in env_mol.GetAtoms()
    ], dtype=th.float)


    # (P,), (L,) -> (L, P)
    vdw_min_dists = 1.05 * (vdwr_pocket[..., None] + vdwr_lig[None])

    adjk_mat = th.tensor(Chem.GetDistanceMatrix(this_mol))
    adj_mat = adjk_mat == 1
    # (L,), (L,) -> (L, L)
    cov_self_min_dists = adj_mat.float() * (
        covr_lig[..., None] + covr_lig[None]
    )
    vdw_self_min_dists = vdwr_lig[..., None] + vdwr_lig[None]
    vdw_self_min_dists.diagonal(dim1=-1, dim2=-2)[:] = 0.0
    vdw_self_min_dists[adjk_mat == 2] *= 0.65
    vdw_self_min_dists[adjk_mat == 3] *= 0.9
    vdw_self_min_dists = (
        cov_self_min_dists + (~adj_mat).float() * vdw_self_min_dists
    )

    def custom_func(coords: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """Minimize clashes (clash_score, self_clash_score) while holding coords close
        to initial pred by a harmonic potential (harm_score).
        Inputs:
        * coords: (N, 3) lig coordinates to optimize.
        Outputs: (N, 3), (,) th.Tensors
        """
        # (N, 3), (C, 3) -> (N, C)
        clash_score = 10.0 * (
            (env_coords[:, None] - coords[None]).norm(dim=-1) - vdw_min_dists
        ).clamp(max=0.0)
        # (N, C) -> (,)
        clash_score = clash_score.square().sum(dim=-2).mean(dim=-1)
        # (N, 3) -> (N, N)
        self_clash_score = 10.0 * (
            (coords[:, None] - coords[None]).norm(dim=-1) - vdw_self_min_dists
        ).clamp(max=0.0)
        # (N, N) -> (,)
        self_clash_score = self_clash_score.square().sum(dim=-2).mean(dim=-1)
        # (C, 3), (C, 3) -> (,)
        harm_score = 10 * (coords - lig_coords).norm(dim=-1)
        harm_score = th.maximum(harm_score.square(), harm_score).mean(dim=-1)
        return (
            coords,
            harm_score + (clash_score + 0.5 * self_clash_score) * coords.shape[0],
        )

    if custom_func(this_coords)[-1] > tol:
        return opt_6t(this_mol=this_mol, this_coords = this_coords, custom_func=custom_func)
    else:
        print(f"clashing, minimizing")
        return this_coords


# global integration

def single_refine(
    input_ligand_path: str,
    output_ligand_path: str,
    label_ligand_path: Optional[str],
    pocket_mol_path: Optional[str],
    superimpose: bool = True,
    num_6t_trials: int = 1,
    min_clash: bool = True,
    clash_tol: float = 0.75,
) -> bool:
    """ Refine a single ligand, optionally 6+T and steric clash avoidance (relax). """
    if not superimpose and not min_clash:
        raise ValueError("At least one of superimpose and min_clash must be True.")

    in_lig = Chem.MolFromMolFile(input_ligand_path, sanitize=False)
    Chem.SanitizeMol(in_lig, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL, catchErrors=True)
    conf_mol = copy.deepcopy(in_lig)
    conf_mol = Chem.RemoveHs(conf_mol)
    if superimpose:
        best_rmsd, best_coords = th.inf, None
        # (N, 3) -> (N, 3)
        for i in range(3*num_6t_trials):
            conf_mol = single_conf_gen(conf_mol, num_confs=1, seed=i)
            # (N, 3)
            coords = get_coord(conf_mol)
            label_coords = get_coord(in_lig)
            opt_coords = opt_6t(
                this_mol=in_lig, this_coords=coords, other_coords=label_coords,
            ).detach()
            rmsd = (opt_coords - label_coords).square().sum(dim=-1).mean(dim=-1).sqrt()
            if rmsd < best_rmsd:
                best_rmsd = rmsd
                best_coords = opt_coords

        opt_coords = best_coords
        # write output
        conf_mol = set_coord(conf_mol, opt_coords)
        Chem.MolToMolFile(conf_mol, output_ligand_path, kekulize=False)
        in_lig = conf_mol

    if min_clash:
        if not pocket_mol_path:
            print(f"No pocket provided for {input_ligand_path}, skipping clash minimization")
            return True
        # (N, 3)
        Chem.SanitizeMol(conf_mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL, catchErrors=True)
        conf_mol = Chem.RemoveHs(conf_mol)
        opt_coords = get_coord(conf_mol)
        # TODO: this might be very slow! maybe needs to be accelerated. BioPandas? We just need atom types and coords
        prot_mol = Chem.MolFromPDBFile(pocket_mol_path, sanitize=False)
        dist_mask = th.cdist(opt_coords, get_coord(prot_mol)).amin(dim=0) < 7.5
        pocket_mol = create_subset_molecule(prot_mol, mask=dist_mask.tolist())
        opt_coords = clash_relax(
            this_mol=conf_mol, this_coords=opt_coords, env_mol=pocket_mol, tol=clash_tol
        ).detach()
        # write output
        new_mol = set_coord(conf_mol, opt_coords)
        Chem.MolToMolFile(new_mol, output_ligand_path, kekulize=False)

    if label_ligand_path is not None:
        label_lig = Chem.MolFromMolFile(label_ligand_path, sanitize=False)
        Chem.SanitizeMol(label_lig, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL, catchErrors=True)
        label_lig = Chem.RemoveHs(label_lig)
        label_coords = get_coord(label_lig)
        _rmsd = round(rmsd_func(label_coords.numpy(), opt_coords.numpy(), mol=conf_mol), 4)
        clean_smi = Chem.MolToSmiles(Chem.RemoveHs(conf_mol))
        print(f"{input_ligand_path}-{clean_smi}-RMSD:{_rmsd}")

    return True


if __name__ == "__main__":
    th.set_num_threads(1)
    th.manual_seed(0)
    parser = argparse.ArgumentParser(description="Docking with gradient")
    parser.add_argument(
        "--input-ligand", type=str, default=None,
        help="input ligand sdf path. Usually from coordinates optimization"
    )
    parser.add_argument(
        "--output-ligand", type=str, default=None, help="output ligand sdf path."
    )
    parser.add_argument(
        "--label-ligand", type=str, default=None, help="to compute RMSD"
    )
    parser.add_argument(
        "--superimpose", type=int, default=1,
        help="Superimpose a mol to the given mol by 6+T")
    parser.add_argument(
        "--num-6t-trials", type=int, default=1,
        help="Number of trials for confgen + 6t superimposition. Keeps the lowest RMSD")
    parser.add_argument(
        "--min-clash", type=int, default=1,
        help="Minimize steric clashes as postprocessing, needs a pocket-mol")
    parser.add_argument(
        "--clash-tol", type=float, default=0.0,
        help="Clash tolerance for minimization. Triggers opt if higher. Posebusters invalids are all >= 100")
    parser.add_argument(
        "--pocket-mol", type=str, default=None,
        help="(Optional) pocket PDB file (protein, protein+metals, protein+metals+water, "
        "protein+metals+water+cofactors). Required to minimize clashes",
    )
    args = parser.parse_args()

    single_refine(
        args.input_ligand, args.output_ligand, args.label_ligand, args.pocket_mol,
        bool(args.superimpose), args.num_6t_trials, bool(args.min_clash), args.clash_tol)



