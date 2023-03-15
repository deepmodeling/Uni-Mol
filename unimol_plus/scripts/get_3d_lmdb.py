import gzip
import os, sys
import pickle
from tqdm import tqdm
from multiprocessing import Pool
import lmdb

# '2022.09.3'
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdMolAlign import GetBestAlignmentTransform
import numpy as np
import torch

split_key = sys.argv[1]

split = torch.load("split_dict.pt")
valid_index = split[split_key]

lines = gzip.open("data.csv.gz", "r").readlines()

target = []
smiles = []

for i in range(1, len(lines)):
    try:
        s = lines[i].decode().split(",")
        smiles.append(s[1])
        target.append(float(s[2]))
    except:
        target.append(None)

del lines

if split_key == "train":
    label_env = lmdb.open(
        "label_3D.lmdb",
        subdir=False,
        readonly=False,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=1,
        map_size=int(100e9),
    )

    with label_env.begin() as txn:
        train_keys = list(txn.cursor().iternext(values=False))
else:
    train_keys = valid_index


def get_info(src_mol, perm=None):
    atoms = np.array([x.GetSymbol() for x in src_mol.GetAtoms()])
    pos = src_mol.GetConformer().GetPositions()
    if perm is not None:
        new_atoms = []
        new_pos = np.zeros_like(pos)
        for i in range(len(atoms)):
            j = perm[i]
            new_atoms.append(atoms[j])
            new_pos[i, :] = pos[j, :]
        return np.array(new_atoms), new_pos
    else:
        return atoms, pos


def align_to(src_mol, ref_mol):
    t = GetBestAlignmentTransform(src_mol, ref_mol)
    perm = {x[1]: x[0] for x in t[2]}
    R = t[1][:3, :3].T
    T = t[1][:3, 3].T

    ref_atoms, ref_pos = get_info(ref_mol)
    src_atoms, src_pos = get_info(src_mol, perm)
    assert np.all(ref_atoms == src_atoms)
    src_pos = src_pos @ R + T

    def cal_rmsd(true_atom_pos, pred_atom_pos, eps: float = 1e-6):
        sd = np.square(true_atom_pos - pred_atom_pos).sum(axis=-1)
        msd = np.mean(sd)
        return np.sqrt(msd + eps)

    cur_rmsd = cal_rmsd(src_pos, ref_pos)
    assert np.abs(cur_rmsd - t[0]) < 1e-2
    return ref_atoms, src_pos, ref_pos


def rdkit_mmff(mol):
    try:
        AllChem.MMFFOptimizeMolecule(mol)
        new_mol = rdkit_remove_hs(mol)
        pos = new_mol.GetConformer().GetPositions()
        return new_mol
    except:
        return rdkit_remove_hs(mol)


def read_smiles(smile):
    try:
        mol = Chem.MolFromSmiles(smile)
    except:
        print("warning: cannot sanitize smiles: ", smile)
        mol = Chem.MolFromSmiles(smile, sanitize=False)
    mol = Chem.AddHs(mol)
    return mol


def read_mol_block(mol_block, removeHs=True):
    try:
        mol = Chem.MolFromMolBlock(mol_block, removeHs=removeHs)
    except:
        print("warning: cannot sanitize : ", mol_block)
        mol = Chem.MolFromMolBlock(mol_block, sanitize=False, removeHs=removeHs)
    return mol


def rdkit_remove_hs(mol):
    try:
        return Chem.RemoveHs(mol)
    except:
        return Chem.RemoveHs(mol, sanitize=False)


def rdkit_2d_gen(smile):
    m = read_smiles(smile)
    AllChem.Compute2DCoords(m)
    m = rdkit_mmff(m)
    pos = m.GetConformer().GetPositions()
    return m


def rdkit_3d_gen(smile, seed):
    mol = read_smiles(smile)
    AllChem.EmbedMolecule(mol, randomSeed=seed, maxAttempts=1000)
    mol = rdkit_mmff(mol)
    pos = mol.GetConformer().GetPositions()
    return mol


def mols_gen(smiles, index, seed=-1, num_confs=8, num_obabel_confs=0, label_mol=None):
    si = 0
    ref_mol = None
    for i in range(5):
        try:
            ref_mol = rdkit_3d_gen(smiles, seed + i)
            if label_mol is not None:
                _, label_pos, _ = align_to(label_mol, ref_mol)
            ref_rdkit = True
            ref_2d = False
            break
        except:
            ref_mol = None
    si = i
    if ref_mol is None:
        try:
            ref_mol = rdkit_2d_gen(smiles)
            if label_mol is not None:
                _, label_pos, _ = align_to(label_mol, ref_mol)
            ref_rdkit = False
            ref_2d = True
        except:
            return None, None, None, None, False

    atoms, init_pos = get_info(ref_mol)
    init_pos_list = [init_pos]

    if label_mol is None:
        label_pos = init_pos

    if ref_2d:
        return ref_mol, atoms, init_pos_list, label_pos, False

    max_try = num_confs * 10
    for i in range(max_try):
        try:
            cur_mol = rdkit_3d_gen(smiles, seed + i + 1 + si)
            _, cur_pos, _ = align_to(cur_mol, ref_mol)
            init_pos_list.append(cur_pos)
        except:
            pass
        if len(init_pos_list) >= num_confs:
            break
    return ref_mol, atoms, init_pos_list, label_pos, True


def get_by_key(env, key):
    data = env.begin().get(key)
    if data is None:
        return data
    else:
        try:
            return pickle.loads(gzip.decompress(data))
        except:
            return None


# allowable multiple choice node and edge features
allowable_features = {
    "possible_atomic_num_list": list(range(1, 119)) + ["misc"],
    "possible_chirality_list": [
        "CHI_UNSPECIFIED",
        "CHI_TETRAHEDRAL_CW",
        "CHI_TETRAHEDRAL_CCW",
        "CHI_TRIGONALBIPYRAMIDAL",
        "CHI_OCTAHEDRAL",
        "CHI_SQUAREPLANAR",
        "CHI_OTHER",
    ],
    "possible_degree_list": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, "misc"],
    "possible_formal_charge_list": [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, "misc"],
    "possible_numH_list": [0, 1, 2, 3, 4, 5, 6, 7, 8, "misc"],
    "possible_number_radical_e_list": [0, 1, 2, 3, 4, "misc"],
    "possible_hybridization_list": ["SP", "SP2", "SP3", "SP3D", "SP3D2", "misc"],
    "possible_is_aromatic_list": [False, True],
    "possible_is_in_ring_list": [False, True],
    "possible_bond_type_list": ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC", "misc"],
    "possible_bond_stereo_list": [
        "STEREONONE",
        "STEREOZ",
        "STEREOE",
        "STEREOCIS",
        "STEREOTRANS",
        "STEREOANY",
    ],
    "possible_is_conjugated_list": [False, True],
}


def safe_index(l, e):
    """
    Return index of element e in list l. If e is not present, return the last index
    """
    try:
        return l.index(e)
    except:
        return len(l) - 1


def atom_to_feature_vector(atom):
    """
    Converts rdkit atom object to feature list of indices
    :param mol: rdkit atom object
    :return: list
    """
    atom_feature = [
        safe_index(allowable_features["possible_atomic_num_list"], atom.GetAtomicNum()),
        allowable_features["possible_chirality_list"].index(str(atom.GetChiralTag())),
        safe_index(allowable_features["possible_degree_list"], atom.GetTotalDegree()),
        safe_index(
            allowable_features["possible_formal_charge_list"], atom.GetFormalCharge()
        ),
        safe_index(allowable_features["possible_numH_list"], atom.GetTotalNumHs()),
        safe_index(
            allowable_features["possible_number_radical_e_list"],
            atom.GetNumRadicalElectrons(),
        ),
        safe_index(
            allowable_features["possible_hybridization_list"],
            str(atom.GetHybridization()),
        ),
        allowable_features["possible_is_aromatic_list"].index(atom.GetIsAromatic()),
        allowable_features["possible_is_in_ring_list"].index(atom.IsInRing()),
    ]
    return atom_feature


def bond_to_feature_vector(bond):
    """
    Converts rdkit bond object to feature list of indices
    :param mol: rdkit bond object
    :return: list
    """
    bond_feature = [
        safe_index(
            allowable_features["possible_bond_type_list"], str(bond.GetBondType())
        ),
        allowable_features["possible_bond_stereo_list"].index(str(bond.GetStereo())),
        allowable_features["possible_is_conjugated_list"].index(bond.GetIsConjugated()),
    ]
    return bond_feature


def get_graph(mol):
    """
    Converts SMILES string to graph Data object
    :input: SMILES string (str)
    :return: graph object
    """
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_features_list.append(atom_to_feature_vector(atom))
    x = np.array(atom_features_list, dtype=np.int32)
    # bonds
    num_bond_features = 3  # bond type, bond stereo, is_conjugated
    if len(mol.GetBonds()) > 0:  # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = bond_to_feature_vector(bond)
            # add edges in both directions
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)
        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = np.array(edges_list, dtype=np.int32).T
        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = np.array(edge_features_list, dtype=np.int32)

    else:  # mol has no bonds
        edge_index = np.empty((2, 0), dtype=np.int32)
        edge_attr = np.empty((0, num_bond_features), dtype=np.int32)
    return x, edge_index, edge_attr


def process_one(key):
    if split_key == "train":
        index = int.from_bytes(key, "big")
        label_str = get_by_key(label_env, key)
        label_mol = read_mol_block(label_str)
    else:
        index = int(key)
        key = index.to_bytes(4, byteorder="big")
        label_mol = None
    ori_smi = smiles[index]
    seed = int(index % 1000 + 1)
    ref_mol, atoms, init_pos_list, label_pos, is_3d = mols_gen(
        ori_smi, index, seed=seed, label_mol=label_mol
    )

    if label_pos is None or len(atoms) <= 0:
        print(index, ori_smi)
        return key, None, False
    node_attr, edge_index, edge_attr = get_graph(ref_mol)
    return (
        key,
        gzip.compress(
            pickle.dumps(
                {
                    "atoms": atoms,
                    "input_pos": init_pos_list,
                    "label_pos": label_pos,
                    "target": target[index],
                    "smi": ori_smi,
                    "node_attr": node_attr,
                    "edge_index": edge_index,
                    "edge_attr": edge_attr,
                }
            )
        ),
        is_3d,
    )


os.system(f"rm -f {split_key}.lmdb")

env_new = lmdb.open(
    f"{split_key}.lmdb",
    subdir=False,
    readonly=False,
    lock=False,
    readahead=False,
    meminit=False,
    max_readers=1,
    map_size=int(100e9),
)
txn_write = env_new.begin(write=True)
i = 0
error_cnt = 0
is_3d_cnt = 0
with Pool(112) as pool:
    for ret in tqdm(
        pool.imap_unordered(process_one, train_keys), total=len(train_keys)
    ):
        key, val, is_3d = ret
        if val is not None:
            txn_write.put(key, val)
        else:
            error_cnt += 1
        if is_3d:
            is_3d_cnt += 1
        # use `int.from_bytes(key, "big")` to decode from bytes
        i += 1
        if i % 10000 == 0:
            txn_write.commit()
            txn_write = env_new.begin(write=True)


txn_write.commit()
env_new.close()
print(error_cnt, is_3d_cnt)
