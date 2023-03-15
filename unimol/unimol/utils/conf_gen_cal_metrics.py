# Copyright (c) DP Techonology, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pandas as pd
import numpy as np
import os
import copy
import pickle
import lmdb
from rdkit import Chem
from tqdm import tqdm
from rdkit.Chem import rdMolTransforms
from rdkit.Chem import AllChem
from rdkit.Chem.rdMolAlign import GetBestRMS
from rdkit.Chem.rdForceFieldHelpers import MMFFOptimizeMolecule
from rdkit.Chem import rdMolAlign as MA
from scipy.spatial.transform import Rotation
from multiprocessing import Pool
from sklearn.cluster import KMeans
import argparse


def get_torsions(m):
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


def GetDihedral(conf, atom_idx):
    return rdMolTransforms.GetDihedralRad(
        conf, atom_idx[0], atom_idx[1], atom_idx[2], atom_idx[3]
    )


def single_conf_gen_bonds(tgt_mol, num_confs=1000, seed=42):
    mol = copy.deepcopy(tgt_mol)
    mol = Chem.AddHs(mol)
    allconformers = AllChem.EmbedMultipleConfs(
        mol, numConfs=num_confs, randomSeed=seed, clearConfs=True
    )
    mol = Chem.RemoveHs(mol)
    rotable_bonds = get_torsions(mol)
    for i in range(len(allconformers)):
        np.random.seed(i)
        values = 3.1415926 * 2 * np.random.rand(len(rotable_bonds))
        for idx in range(len(rotable_bonds)):
            SetDihedral(mol.GetConformers()[i], rotable_bonds[idx], values[idx])
        Chem.rdMolTransforms.CanonicalizeConformer(mol.GetConformers()[i])
    return mol


def single_conf_gen(tgt_mol, num_confs=1000, seed=42):
    mol = copy.deepcopy(tgt_mol)
    mol = Chem.AddHs(mol)
    allconformers = AllChem.EmbedMultipleConfs(
        mol, numConfs=num_confs, randomSeed=seed, clearConfs=True
    )
    sz = len(allconformers)
    for i in range(sz):
        try:
            AllChem.MMFFOptimizeMolecule(mol, confId=i)
        except:
            continue
    mol = Chem.RemoveHs(mol)
    return mol


def single_conf_gen_no_MMFF(tgt_mol, num_confs=1000, seed=42):
    mol = copy.deepcopy(tgt_mol)
    mol = Chem.AddHs(mol)
    allconformers = AllChem.EmbedMultipleConfs(
        mol, numConfs=num_confs, randomSeed=seed, clearConfs=True
    )
    mol = Chem.RemoveHs(mol)
    return mol


def clustering(mol, M=1000, N=100):
    rdkit_mol = single_conf_gen(mol, num_confs=M)
    rdkit_mol = Chem.RemoveHs(rdkit_mol)
    total_sz = 0
    sz = len(rdkit_mol.GetConformers())
    tgt_coords = rdkit_mol.GetConformers()[0].GetPositions().astype(np.float32)
    tgt_coords = tgt_coords - np.mean(tgt_coords, axis=0)
    rdkit_coords_list = []
    for i in range(sz):
        _coords = rdkit_mol.GetConformers()[i].GetPositions().astype(np.float32)
        _coords = _coords - _coords.mean(axis=0)  # need to normalize first
        _R, _score = Rotation.align_vectors(_coords, tgt_coords)
        rdkit_coords_list.append(np.dot(_coords, _R.as_matrix()))
    total_sz += sz

    ### add no MMFF optimize conformers
    rdkit_mol = single_conf_gen_no_MMFF(mol, num_confs=int(M // 4), seed=43)
    rdkit_mol = Chem.RemoveHs(rdkit_mol)
    sz = len(rdkit_mol.GetConformers())
    for i in range(sz):
        _coords = rdkit_mol.GetConformers()[i].GetPositions().astype(np.float32)
        _coords = _coords - _coords.mean(axis=0)  # need to normalize first
        _R, _score = Rotation.align_vectors(_coords, tgt_coords)
        rdkit_coords_list.append(np.dot(_coords, _R.as_matrix()))
    total_sz += sz

    ### add uniform rotation bonds conformers
    rdkit_mol = single_conf_gen_bonds(mol, num_confs=int(M // 4), seed=43)
    rdkit_mol = Chem.RemoveHs(rdkit_mol)
    sz = len(rdkit_mol.GetConformers())
    for i in range(sz):
        _coords = rdkit_mol.GetConformers()[i].GetPositions().astype(np.float32)
        _coords = _coords - _coords.mean(axis=0)  # need to normalize first
        _R, _score = Rotation.align_vectors(_coords, tgt_coords)
        rdkit_coords_list.append(np.dot(_coords, _R.as_matrix()))
    total_sz += sz

    rdkit_coords_flatten = np.array(rdkit_coords_list).reshape(total_sz, -1)
    cluster_size = N
    ids = (
        KMeans(n_clusters=cluster_size, random_state=42)
        .fit_predict(rdkit_coords_flatten)
        .tolist()
    )
    coords_list = [rdkit_coords_list[ids.index(i)] for i in range(cluster_size)]
    return coords_list


def single_process_data(content):
    smi, tgt_mol_list = content[0], content[1]
    M = min(20 * len(tgt_mol_list), 2000)
    N = 2 * len(tgt_mol_list)
    tgt_mol = copy.deepcopy(tgt_mol_list[0])
    tgt_mol = Chem.RemoveHs(tgt_mol)
    rdkit_cluster_coords_list = clustering(tgt_mol, M=M, N=N)
    atoms = [atom.GetSymbol() for atom in tgt_mol.GetAtoms()]
    sz = len(rdkit_cluster_coords_list)
    ## check target molecule atoms is the same as the input molecule
    for _mol in tgt_mol_list:
        _mol = Chem.RemoveHs(_mol)
        _atoms = [atom.GetSymbol() for atom in _mol.GetAtoms()]
        assert _atoms == atoms, print(smi)

    tgt_coords = tgt_mol.GetConformer().GetPositions().astype(np.float32)
    dump_list = []
    for i in range(sz):
        dump_list.append(
            {
                "atoms": atoms,
                "coordinates": [rdkit_cluster_coords_list[i]],
                "smi": smi,
                "target": tgt_coords,
            }
        )
    return dump_list


def write_lmdb(content_list, output_dir, name, nthreads=16):

    os.makedirs(output_dir, exist_ok=True)
    output_name = os.path.join(output_dir, f"{name}.lmdb")
    print(output_name)
    try:
        os.remove(output_name)
    except:
        pass
    env_new = lmdb.open(
        output_name,
        subdir=False,
        readonly=False,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=1,
        map_size=int(100e9),
    )
    txn_write = env_new.begin(write=True)
    with Pool(nthreads) as pool:
        i = 0
        for inner_output in tqdm(pool.imap(inner_process, content_list)):
            if inner_output is not None:
                for item in inner_output:
                    txn_write.put(
                        f"{i}".encode("ascii"), pickle.dumps(item, protocol=-1)
                    )
                    i += 1
        print("{} process {} lines".format(output_name, i))
        txn_write.commit()
        env_new.close()


def inner_process(content):
    try:
        return single_process_data(content)
    except:
        return None


def data_pre(predict_path, data_path, normalize=True):

    predict = pd.read_pickle(predict_path)
    data = pd.read_pickle(data_path)
    data = data.groupby("smi")["mol"].apply(list).reset_index()
    smi_list, predict_list = [], []
    for batch in predict:
        sz = batch["bsz"]
        for i in range(sz):
            smi_list.append(batch["smi_name"][i])
            coord_predict = batch["coord_predict"][i]
            coord_target = batch["coord_target"][i]
            coord_mask = coord_target[:, 0].ne(0)
            coord_predict = coord_predict[coord_mask, :].cpu().numpy()
            if normalize:
                coord_predict = coord_predict - coord_predict.mean(axis=0)

            predict_list.append(coord_predict)

    predict_df = pd.DataFrame({"smi": smi_list, "predict_coord": predict_list})
    predict_df = predict_df.groupby("smi")["predict_coord"].apply(list).reset_index()

    df = pd.merge(data, predict_df, on="smi", how="left")
    print("preprocessing 1...")
    ref_mols_list, gen_mols_list = [], []
    for smi, mol_list, pos_list in zip(df["smi"], df["mol"], df["predict_coord"]):
        if "." in smi:
            print(smi)
            continue
        ref_mols_list.append(mol_list)
        gen_mols = [set_rdmol_positions(mol_list[0], pos) for pos in pos_list]
        gen_mols_list.append(gen_mols)
    print("preprocessing 2...")
    return ref_mols_list, gen_mols_list


def get_rmsd_min(ref_mols, gen_mols, use_ff=False, threshold=0.5):
    rmsd_mat = np.zeros([len(ref_mols), len(gen_mols)], dtype=np.float32)
    for i, gen_mol in enumerate(gen_mols):
        gen_mol_c = copy.deepcopy(gen_mol)
        if use_ff:
            MMFFOptimizeMolecule(gen_mol_c)
        for j, ref_mol in enumerate(ref_mols):
            ref_mol_c = copy.deepcopy(ref_mol)
            rmsd_mat[j, i] = get_best_rmsd(gen_mol_c, ref_mol_c)
    rmsd_mat_min = rmsd_mat.min(-1)
    return (rmsd_mat_min <= threshold).mean(), rmsd_mat_min.mean()


def get_best_rmsd(gen_mol, ref_mol):
    gen_mol = Chem.RemoveHs(gen_mol)
    ref_mol = Chem.RemoveHs(ref_mol)
    rmsd = MA.GetBestRMS(gen_mol, ref_mol)
    return rmsd


def set_rdmol_positions(rdkit_mol, pos):
    rdkit_mol = Chem.RemoveHs(rdkit_mol)
    assert rdkit_mol.GetConformer(0).GetPositions().shape[0] == pos.shape[0]
    mol = copy.deepcopy(rdkit_mol)
    for i in range(pos.shape[0]):
        mol.GetConformer(0).SetAtomPosition(i, pos[i].tolist())
    return mol


def print_results(cov, mat):
    print("COV_mean: ", np.mean(cov), ";COV_median: ", np.median(cov))
    print("MAT_mean: ", np.mean(mat), ";MAT_median: ", np.median(mat))


def single_process(content):
    ref_mols, gen_mols, use_ff, threshold = content
    cov, mat = get_rmsd_min(ref_mols, gen_mols, use_ff, threshold)
    return cov, mat


def process(content):
    try:
        return single_process(content)
    except:
        return None


def cal_metrics(predict_path, data_path, use_ff=False, threshold=0.5, nthreads=40):
    ref_mols_list, gen_mols_list = data_pre(predict_path, data_path, normalize=True)
    print("cal_metrics...")
    cov_list, mat_list = [], []
    content_list = []
    for ref_mols, gen_mols in zip(ref_mols_list, gen_mols_list):
        content_list.append((ref_mols, gen_mols, use_ff, threshold))
    with Pool(nthreads) as pool:
        for inner_output in tqdm(pool.imap(process, content_list)):
            if inner_output is None:
                continue
            cov, mat = inner_output
            cov_list.append(cov)
            mat_list.append(mat)
    print_results(cov_list, mat_list)
    return True


def main():
    parser = argparse.ArgumentParser(
        description="generate initial rdkit test data and cal metrics"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="cal_metrics",
        choices=["gen_data", "cal_metrics"],
    )
    parser.add_argument(
        "--reference-file",
        type=str,
        default="./conformation_generation/qm9/test_data_200.pkl",
        help="Location of the reference set",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./conformation_generation/qm9",
        help="Location of the generated data",
    )
    parser.add_argument("--nthreads", type=int, default=40, help="num of threads")
    parser.add_argument(
        "--predict-file",
        type=str,
        default="./infer_confgen/save_confgen_test.out.pkl",
        help="Location of the prediction file",
    )
    parser.add_argument(
        "--threshold",
        default=0.5,
        type=float,
        help="threshold for cal metrics, qm9: 0.5; drugs: 1.25",
    )
    args = parser.parse_args()

    if args.mode == "gen_data":
        # generate test data
        output_dir = args.output_dir
        name = "test"
        data = pd.read_pickle(args.reference_file)
        content_list = (
            pd.DataFrame(data).groupby("smi")["mol"].apply(list).reset_index().values
        )
        print(content_list[0])
        write_lmdb(content_list, output_dir, name, nthreads=args.nthreads)

    ### Uni-Mol predicting... ###

    elif args.mode == "cal_metrics":
        # cal metrics
        predict_file = args.predict_file
        data_path = args.reference_file
        use_ff = False
        threshold = args.threshold
        cal_metrics(predict_file, data_path, use_ff, threshold, args.nthreads)


if __name__ == "__main__":
    main()
