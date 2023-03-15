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


def get_torsions(m, removeHs=True):
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


def SetDihedral(conf, atom_idx, new_vale):
    rdMolTransforms.SetDihedralRad(
        conf, atom_idx[0], atom_idx[1], atom_idx[2], atom_idx[3], new_vale
    )


def single_conf_gen_bonds(tgt_mol, num_confs=1000, seed=42, removeHs=True):
    mol = copy.deepcopy(tgt_mol)
    mol = Chem.AddHs(mol)
    allconformers = AllChem.EmbedMultipleConfs(
        mol, numConfs=num_confs, randomSeed=seed, clearConfs=True
    )
    if removeHs:
        mol = Chem.RemoveHs(mol)
    rotable_bonds = get_torsions(mol, removeHs=removeHs)
    for i in range(len(allconformers)):
        np.random.seed(i)
        values = 3.1415926 * 2 * np.random.rand(len(rotable_bonds))
        for idx in range(len(rotable_bonds)):
            SetDihedral(mol.GetConformers()[i], rotable_bonds[idx], values[idx])
        Chem.rdMolTransforms.CanonicalizeConformer(mol.GetConformers()[i])
    return mol


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
):
    sz = len(mol_list)
    for i in range(sz // tta_times):
        start_idx, end_idx = i * tta_times, (i + 1) * tta_times
        distance_predict_tta = distance_predict_list[start_idx:end_idx]
        holo_distance_predict_tta = holo_distance_predict_list[start_idx:end_idx]

        mol = copy.deepcopy(mol_list[start_idx])
        rdkit_mol = single_conf_gen_bonds(
            mol, num_confs=tta_times, seed=42, removeHs=True
        )
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


def rmsd_func(holo_coords, predict_coords):
    if predict_coords is not np.nan:
        sz = holo_coords.shape
        rmsd = np.sqrt(np.sum((predict_coords - holo_coords) ** 2) / sz[0])
        return rmsd
    return 1000.0


def print_results(rmsd_results):
    print("RMSD < 1.0 : ", np.mean(rmsd_results < 1.0))
    print("RMSD < 1.5 : ", np.mean(rmsd_results < 1.5))
    print("RMSD < 2.0 : ", np.mean(rmsd_results < 2.0))
    print("RMSD < 3.0 : ", np.mean(rmsd_results < 3.0))
    print("RMSD < 5.0 : ", np.mean(rmsd_results < 5.0))
    print("avg RMSD : ", np.mean(rmsd_results))
