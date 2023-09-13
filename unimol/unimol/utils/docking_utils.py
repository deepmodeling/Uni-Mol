# Copyright (c) DP Techonology, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")
import warnings

warnings.filterwarnings(action="ignore")
from rdkit.Chem import rdMolTransforms
import copy
import lmdb
import pickle
import pandas as pd
from typing import Dict, List, Optional
from unimol.utils.conf_gen_cal_metrics import clustering, single_conf_gen


def add_all_conformers_to_mol(mol: Chem.Mol, conformers: List[np.ndarray]) -> Chem.Mol:
    mol = copy.deepcopy(mol)
    mol.RemoveAllConformers()
    for i, conf_pos in enumerate(conformers):
        conf = Chem.Conformer(mol.GetNumAtoms())
        mol.AddConformer(conf, assignId=True)

        conf = mol.GetConformer(i)
        positions = conf_pos.tolist()
        for j in range(mol.GetNumAtoms()):
            conf.SetAtomPosition(j, positions[j])
    return mol


def get_torsions(m: Chem.Mol, removeHs=True) -> List:
    if removeHs:
        m = Chem.RemoveHs(m)
    torsionList = []
    torsionSmarts = "[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]"
    torsionQuery = Chem.MolFromSmarts(torsionSmarts)
    matches = m.GetSubstructMatches(torsionQuery)
    for match in matches:
        idx2 = match[0]
        idx3 = match[1]
        bond = m.GetBondBetweenAtoms(idx2, idx3)
        jAtom = m.GetAtomWithIdx(idx2)
        kAtom = m.GetAtomWithIdx(idx3)
        for b1 in jAtom.GetBonds():
            if b1.GetIdx() == bond.GetIdx():
                continue
            idx1 = b1.GetOtherAtomIdx(idx2)
            for b2 in kAtom.GetBonds():
                if (b2.GetIdx() == bond.GetIdx()) or (b2.GetIdx() == b1.GetIdx()):
                    continue
                idx4 = b2.GetOtherAtomIdx(idx3)
                # skip 3-membered rings
                if idx4 == idx1:
                    continue
                # skip torsions that include hydrogens
                if (m.GetAtomWithIdx(idx1).GetAtomicNum() == 1) or (
                    m.GetAtomWithIdx(idx4).GetAtomicNum() == 1
                ):
                    continue
                if m.GetAtomWithIdx(idx4).IsInRing():
                    torsionList.append((idx4, idx3, idx2, idx1))
                    break
                else:
                    torsionList.append((idx1, idx2, idx3, idx4))
                    break
            break
    return torsionList


def load_lmdb_data(lmdb_path, key):
    env = lmdb.open(
        lmdb_path,
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=256,
    )
    txn = env.begin()
    _keys = list(txn.cursor().iternext(values=False))
    collects = []
    for idx in range(len(_keys)):
        datapoint_pickled = txn.get(f"{idx}".encode("ascii"))
        data = pickle.loads(datapoint_pickled)
        collects.append(data[key])
    return collects


def reprocess_content(content: Dict, base_mol: Optional[Chem.Mol] = None, M: int = 2000, N: int = 10, mmff: bool = False, seed: int = 42, stereo_from3d: bool = True) -> Dict:
    """ Reprocess a data point in the LMDB schema for Docking usage. Ensures correct stereochemistry.
    Basic principle is to perceive stereochem from label molecule's 3D and keep it intact.
    Use default values for best results

    Args:
        content: A dictionary of the LMDB schema. (atoms, holo_mol, mol_list, cooredinates, etc.)
        base_mol: The molecule to replace the holo_mol with, if passed
        M: The number of conformers to generate
        N: The number of clusters to group conformers and pick a representative from
        mmff: Whether to use MMFF minimization after conformer generation
        seed: The random seed to use for conformer generation
        stereo_from3d: Whether to perceive stereochemistry from the 3D coordinates of the label molecule

    Returns:
        A copy of the original, with the holo_mol replaced with the base_mol, and coordinates added.
    """
    if base_mol is None:
        base_mol = content["holo_mol"]
    # Copy so we don't change inputs
    content = copy.deepcopy(content)
    base_mol = copy.deepcopy(base_mol)
    base_mol = Chem.AddHs(base_mol, addCoords=True)
    # assign stereochem from 3d
    if stereo_from3d and base_mol.GetNumConformers() > 0:
        Chem.AssignStereochemistryFrom3D(base_mol)
    ori_smiles = Chem.MolToSmiles(base_mol)
    # create new, clean molecule
    remol = Chem.MolFromSmiles(ori_smiles)
    # reorder to match and add Hs
    idxs = remol.GetSubstructMatches(Chem.RemoveHs(base_mol))
    if isinstance(idxs[0], tuple):
        idxs = idxs[0]
    idxs = list(map(int, idxs))
    remol = Chem.RenumberAtoms(remol, idxs)
    remol = Chem.AddHs(remol, addCoords=True)
    # overwrite - write the diverse conformer set for potential later reuse
    content["coordinates"] = [x for x in clustering(remol, M=M, N=N, seed=seed, removeHs=False, mmff=mmff)]
    content["mol_list"] = [
        Chem.AddHs(
            copy.deepcopy(add_all_conformers_to_mol(
                Chem.RemoveHs(remol), content["coordinates"]
            )), addCoords=True
        ) for i in range(N)
    ]
    content["holo_mol"] = copy.deepcopy(base_mol)
    content["atoms"] = [a.GetSymbol() for a in base_mol.GetAtoms()]
    return content


def docking_data_pre(raw_data_path, predict_path):
    mol_list = load_lmdb_data(raw_data_path, "mol_list")
    mol_list = [Chem.RemoveHs(mol) for items in mol_list for mol in items]
    predict = pd.read_pickle(predict_path)
    (
        smi_list,
        pocket_list,
        pocket_coords_list,
        distance_predict_list,
        holo_distance_predict_list,
        holo_coords_list,
        holo_center_coords_list,
    ) = ([], [], [], [], [], [], [])
    for batch in predict:
        sz = batch["atoms"].size(0)
        for i in range(sz):
            smi_list.append(batch["smi_name"][i])
            pocket_list.append(batch["pocket_name"][i])

            distance_predict = batch["cross_distance_predict"][i]
            token_mask = batch["atoms"][i] > 2
            pocket_token_mask = batch["pocket_atoms"][i] > 2
            distance_predict = distance_predict[token_mask][:, pocket_token_mask]
            pocket_coords = batch["pocket_coordinates"][i]
            pocket_coords = pocket_coords[pocket_token_mask, :]

            holo_distance_predict = batch["holo_distance_predict"][i]
            holo_distance_predict = holo_distance_predict[token_mask][:, token_mask]

            holo_coordinates = batch["holo_coordinates"][i]
            holo_coordinates = holo_coordinates[token_mask, :]
            holo_center_coordinates = batch["holo_center_coordinates"][i][:3]

            pocket_coords = pocket_coords.numpy().astype(np.float32)
            distance_predict = distance_predict.numpy().astype(np.float32)
            holo_distance_predict = holo_distance_predict.numpy().astype(np.float32)
            holo_coords = holo_coordinates.numpy().astype(np.float32)

            pocket_coords_list.append(pocket_coords)
            distance_predict_list.append(distance_predict)
            holo_distance_predict_list.append(holo_distance_predict)
            holo_coords_list.append(holo_coords)
            holo_center_coords_list.append(holo_center_coordinates)

    return (
        mol_list,
        smi_list,
        pocket_list,
        pocket_coords_list,
        distance_predict_list,
        holo_distance_predict_list,
        holo_coords_list,
        holo_center_coords_list,
    )


def ensemble_iterations(
    mol_list,
    smi_list,
    pocket_list,
    pocket_coords_list,
    distance_predict_list,
    holo_distance_predict_list,
    holo_coords_list,
    holo_center_coords_list,
    tta_times=10,
    seed=42,
):
    sz = len(mol_list)
    for i in range(sz // tta_times):
        start_idx, end_idx = i * tta_times, (i + 1) * tta_times
        distance_predict_tta = distance_predict_list[start_idx:end_idx]
        holo_distance_predict_tta = holo_distance_predict_list[start_idx:end_idx]

        mol = copy.deepcopy(mol_list[start_idx])
        rdkit_mol = single_conf_gen(mol, num_confs=tta_times, seed=seed)
        sz = len(rdkit_mol.GetConformers())
        initial_coords_list = [
            rdkit_mol.GetConformers()[i].GetPositions().astype(np.float32)
            for i in range(sz)
        ]

        yield [
            initial_coords_list,
            mol,
            smi_list[start_idx],
            pocket_list[start_idx],
            pocket_coords_list[start_idx],
            distance_predict_tta,
            holo_distance_predict_tta,
            holo_coords_list[start_idx],
            holo_center_coords_list[start_idx],
        ]


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


def print_results(rmsd_results):
    print("RMSD < 1.0 : ", np.mean(rmsd_results < 1.0))
    print("RMSD < 1.5 : ", np.mean(rmsd_results < 1.5))
    print("RMSD < 2.0 : ", np.mean(rmsd_results < 2.0))
    print("RMSD < 3.0 : ", np.mean(rmsd_results < 3.0))
    print("RMSD < 5.0 : ", np.mean(rmsd_results < 5.0))
    print("avg RMSD : ", np.mean(rmsd_results))
