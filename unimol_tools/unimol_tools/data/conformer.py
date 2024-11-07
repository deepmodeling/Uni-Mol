# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function

import os
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
import warnings
from scipy.spatial import distance_matrix
RDLogger.DisableLog('rdApp.*') 
warnings.filterwarnings(action='ignore')
from .dictionary import Dictionary
from multiprocessing import Pool
from tqdm import tqdm
import torch
from numba import njit

from ..utils import logger
from ..config import MODEL_CONFIG
from ..weights import weight_download, WEIGHT_DIR

# https://github.com/snap-stanford/ogb/blob/master/ogb/utils/features.py
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

class ConformerGen(object):
    '''
    This class designed to generate conformers for molecules represented as SMILES strings using provided parameters and configurations. The `transform` method uses multiprocessing to speed up the conformer generation process.
    '''
    def __init__(self, **params):
        """
        Initializes the neural network model based on the provided model name and parameters.

        :param model_name: (str) The name of the model to initialize.
        :param params: Additional parameters for model configuration.

        :return: An instance of the specified neural network model.
        :raises ValueError: If the model name is not recognized.
        """
        self._init_features(**params)

    def _init_features(self, **params):
        """
        Initializes the features of the ConformerGen object based on provided parameters.

        :param params: Arbitrary keyword arguments for feature configuration.
                       These can include the random seed, maximum number of atoms, data type,
                       generation method, generation mode, and whether to remove hydrogens.
        """
        self.seed = params.get('seed', 42)
        self.max_atoms = params.get('max_atoms', 256)
        self.data_type = params.get('data_type', 'molecule')
        self.method = params.get('method', 'rdkit_random')
        self.mode = params.get('mode', 'fast')
        self.remove_hs = params.get('remove_hs', False)
        if self.data_type == 'molecule':
            name = "no_h" if self.remove_hs else "all_h" 
            name = self.data_type + '_' + name
            self.dict_name = MODEL_CONFIG['dict'][name]
        else:
            self.dict_name = MODEL_CONFIG['dict'][self.data_type]
        if not os.path.exists(os.path.join(WEIGHT_DIR, self.dict_name)):
            weight_download(self.dict_name, WEIGHT_DIR)
        self.dictionary = Dictionary.load(os.path.join(WEIGHT_DIR, self.dict_name))
        self.dictionary.add_symbol("[MASK]", is_special=True)

    def single_process(self, smiles):
        """
        Processes a single SMILES string to generate conformers using the specified method.

        :param smiles: (str) The SMILES string representing the molecule.
        :return: A unimolecular data representation (dictionary) of the molecule.
        :raises ValueError: If the conformer generation method is unrecognized.
        """
        if self.method == 'rdkit_random':
            atoms, coordinates = inner_smi2coords(smiles, seed=self.seed, mode=self.mode, remove_hs=self.remove_hs)
            return coords2unimol(atoms, coordinates, self.dictionary, self.max_atoms, remove_hs=self.remove_hs)
        else:
            raise ValueError('Unknown conformer generation method: {}'.format(self.method))
        
    def transform_raw(self, atoms_list, coordinates_list):

        inputs = []
        for atoms, coordinates in zip(atoms_list, coordinates_list):
            inputs.append(coords2unimol(atoms, coordinates, self.dictionary, self.max_atoms, remove_hs=self.remove_hs))
        return inputs

    def transform(self, smiles_list):
        pool = Pool()
        logger.info('Start generating conformers...')
        inputs = [item for item in tqdm(pool.imap(self.single_process, smiles_list))]
        pool.close()
        failed_cnt = np.mean([(item['src_coord']==0.0).all() for item in inputs])
        logger.info('Succeeded in generating conformers for {:.2f}% of molecules.'.format((1-failed_cnt)*100))
        failed_3d_cnt = np.mean([(item['src_coord'][:,2]==0.0).all() for item in inputs])
        logger.info('Succeeded in generating 3d conformers for {:.2f}% of molecules.'.format((1-failed_3d_cnt)*100))
        return inputs


def inner_smi2coords(smi, seed=42, mode='fast', remove_hs=True, return_mol=False):
    '''
    This function is responsible for converting a SMILES (Simplified Molecular Input Line Entry System) string into 3D coordinates for each atom in the molecule. It also allows for the generation of 2D coordinates if 3D conformation generation fails, and optionally removes hydrogen atoms and their coordinates from the resulting data.

    :param smi: (str) The SMILES representation of the molecule.
    :param seed: (int, optional) The random seed for conformation generation. Defaults to 42.
    :param mode: (str, optional) The mode of conformation generation, 'fast' for quick generation, 'heavy' for more attempts. Defaults to 'fast'.
    :param remove_hs: (bool, optional) Whether to remove hydrogen atoms from the final coordinates. Defaults to True.

    :return: A tuple containing the list of atom symbols and their corresponding 3D coordinates.
    :raises AssertionError: If no atoms are present in the molecule or if the coordinates do not align with the atom count.
    '''
    mol = Chem.MolFromSmiles(smi)
    mol = AllChem.AddHs(mol)
    atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
    assert len(atoms)>0, 'No atoms in molecule: {}'.format(smi)
    try:
        # will random generate conformer with seed equal to -1. else fixed random seed.
        res = AllChem.EmbedMolecule(mol, randomSeed=seed)
        if res == 0:
            try:
                # some conformer can not use MMFF optimize
                AllChem.MMFFOptimizeMolecule(mol)
                coordinates = mol.GetConformer().GetPositions().astype(np.float32)
            except:
                coordinates = mol.GetConformer().GetPositions().astype(np.float32)
        ## for fast test... ignore this ###
        elif res == -1 and mode == 'heavy':
            AllChem.EmbedMolecule(mol, maxAttempts=5000, randomSeed=seed)
            try:
                # some conformer can not use MMFF optimize
                AllChem.MMFFOptimizeMolecule(mol)
                coordinates = mol.GetConformer().GetPositions().astype(np.float32)
            except:
                AllChem.Compute2DCoords(mol)
                coordinates_2d = mol.GetConformer().GetPositions().astype(np.float32)
                coordinates = coordinates_2d
        else:
            AllChem.Compute2DCoords(mol)
            coordinates_2d = mol.GetConformer().GetPositions().astype(np.float32)
            coordinates = coordinates_2d
    except:
        print("Failed to generate conformer, replace with zeros.")
        coordinates = np.zeros((len(atoms),3))

    if return_mol:
        return mol # for unimolv2
    
    assert len(atoms) == len(coordinates), "coordinates shape is not align with {}".format(smi)
    if remove_hs:
        idx = [i for i, atom in enumerate(atoms) if atom != 'H']
        atoms_no_h = [atom for atom in atoms if atom != 'H']
        coordinates_no_h = coordinates[idx]
        assert len(atoms_no_h) == len(coordinates_no_h), "coordinates shape is not align with {}".format(smi)
        return atoms_no_h, coordinates_no_h
    else:
        return atoms, coordinates

def inner_coords(atoms, coordinates, remove_hs=True):
    """
    Processes a list of atoms and their corresponding coordinates to remove hydrogen atoms if specified.
    This function takes a list of atom symbols and their corresponding coordinates and optionally removes hydrogen atoms from the output. It includes assertions to ensure the integrity of the data and uses numpy for efficient processing of the coordinates. 

    :param atoms: (list) A list of atom symbols (e.g., ['C', 'H', 'O']).
    :param coordinates: (list of tuples or list of lists) Coordinates corresponding to each atom in the `atoms` list.
    :param remove_hs: (bool, optional) A flag to indicate whether hydrogen atoms should be removed from the output.
                      Defaults to True.
    
    :return: A tuple containing two elements; the filtered list of atom symbols and their corresponding coordinates.
             If `remove_hs` is False, the original lists are returned.
    
    :raises AssertionError: If the length of `atoms` list does not match the length of `coordinates` list.
    """
    assert len(atoms) == len(coordinates), "coordinates shape is not align atoms"
    coordinates = np.array(coordinates).astype(np.float32)
    if remove_hs:
        idx = [i for i, atom in enumerate(atoms) if atom != 'H']
        atoms_no_h = [atom for atom in atoms if atom != 'H']
        coordinates_no_h = coordinates[idx]
        assert len(atoms_no_h) == len(coordinates_no_h), "coordinates shape is not align with atoms"
        return atoms_no_h, coordinates_no_h
    else:
        return atoms, coordinates

def coords2unimol(atoms, coordinates, dictionary, max_atoms=256, remove_hs=True, **params):
    """
    Converts atom symbols and coordinates into a unified molecular representation.

    :param atoms: (list) List of atom symbols.
    :param coordinates: (ndarray) Array of atomic coordinates.
    :param dictionary: (Dictionary) An object that maps atom symbols to unique integers.
    :param max_atoms: (int) The maximum number of atoms to consider for the molecule.
    :param remove_hs: (bool) Whether to remove hydrogen atoms from the representation.
    :param params: Additional parameters.

    :return: A dictionary containing the molecular representation with tokens, distances, coordinates, and edge types.
    """
    atoms, coordinates = inner_coords(atoms, coordinates, remove_hs=remove_hs)
    atoms = np.array(atoms)
    coordinates = np.array(coordinates).astype(np.float32)
    # cropping atoms and coordinates
    if len(atoms) > max_atoms:
        idx = np.random.choice(len(atoms), max_atoms, replace=False)
        atoms = atoms[idx]
        coordinates = coordinates[idx]
    # tokens padding
    src_tokens = np.array([dictionary.bos()] + [dictionary.index(atom) for atom in atoms] + [dictionary.eos()])
    src_distance = np.zeros((len(src_tokens), len(src_tokens)))
    # coordinates normalize & padding
    src_coord = coordinates - coordinates.mean(axis=0)
    src_coord = np.concatenate([np.zeros((1,3)), src_coord, np.zeros((1,3))], axis=0)
    # distance matrix
    src_distance = distance_matrix(src_coord, src_coord)
    # edge type
    src_edge_type = src_tokens.reshape(-1, 1) * len(dictionary) + src_tokens.reshape(1, -1)

    return {
            'src_tokens': src_tokens.astype(int), 
            'src_distance': src_distance.astype(np.float32), 
            'src_coord': src_coord.astype(np.float32), 
            'src_edge_type': src_edge_type.astype(int),
            }

class UniMolV2Feature(object):
    '''
    This class is responsible for generating features for molecules represented as SMILES strings. It uses the ConformerGen class to generate conformers for the molecules and converts the resulting atom symbols and coordinates into a unified molecular representation.
    '''
    def __init__(self, **params):
        """
        Initializes the neural network model based on the provided model name and parameters.

        :param model_name: (str) The name of the model to initialize.
        :param params: Additional parameters for model configuration.

        :return: An instance of the specified neural network model.
        :raises ValueError: If the model name is not recognized.
        """
        self._init_features(**params)

    def _init_features(self, **params):
        """
        Initializes the features of the UniMolV2Feature object based on provided parameters.

        :param params: Arbitrary keyword arguments for feature configuration.
                       These can include the random seed, maximum number of atoms, data type,
                       generation method, generation mode, and whether to remove hydrogens.
        """
        self.seed = params.get('seed', 42)
        self.max_atoms = params.get('max_atoms', 128)
        self.data_type = params.get('data_type', 'molecule')
        self.method = params.get('method', 'rdkit_random')
        self.mode = params.get('mode', 'fast')
        self.remove_hs = params.get('remove_hs', True)

    def single_process(self, smiles):
        """
        Processes a single SMILES string to generate conformers using the specified method.

        :param smiles: (str) The SMILES string representing the molecule.
        :return: A unimolecular data representation (dictionary) of the molecule.
        :raises ValueError: If the conformer generation method is unrecognized.
        """
        if self.method == 'rdkit_random':
            mol = inner_smi2coords(smiles, seed=self.seed, mode=self.mode, remove_hs=self.remove_hs, return_mol=True)
            return mol2unimolv2(mol, self.max_atoms, remove_hs=self.remove_hs)
        else:
            raise ValueError('Unknown conformer generation method: {}'.format(self.method))
        
    def transform_raw(self, atoms_list, coordinates_list):
            
            inputs = []
            for atoms, coordinates in zip(atoms_list, coordinates_list):
                mol = create_mol_from_atoms_and_coords(atoms, coordinates)
                inputs.append(mol2unimolv2(mol, self.max_atoms, remove_hs=self.remove_hs))
            return inputs
    
    def transform(self, smiles_list):
        pool = Pool()
        logger.info('Start generating conformers...')
        inputs = [item for item in tqdm(pool.imap(self.single_process, smiles_list))]
        pool.close()
        # failed_cnt = np.mean([(item['src_coord']==0.0).all() for item in inputs])
        # logger.info('Succeeded in generating conformers for {:.2f}% of molecules.'.format((1-failed_cnt)*100))
        # failed_3d_cnt = np.mean([(item['src_coord'][:,2]==0.0).all() for item in inputs])
        # logger.info('Succeeded in generating 3d conformers for {:.2f}% of molecules.'.format((1-failed_3d_cnt)*100))
        return inputs

def create_mol_from_atoms_and_coords(atoms, coordinates):
    """
    Creates an RDKit molecule object from a list of atom symbols and their corresponding coordinates.

    :param atoms: (list) Atom symbols for the molecule.
    :param coordinates: (list) Atomic coordinates for the molecule.
    :return: RDKit molecule object.
    """
    mol = Chem.RWMol()
    atom_indices = []

    for atom in atoms:
        atom_idx = mol.AddAtom(Chem.Atom(atom))
        atom_indices.append(atom_idx)

    conf = Chem.Conformer(len(atoms))
    for i, coord in enumerate(coordinates):
        conf.SetAtomPosition(i, coord)

    mol.AddConformer(conf)
    Chem.SanitizeMol(mol)
    return mol

def mol2unimolv2(mol, max_atoms=128, remove_hs=True, **params):
    """
    Converts atom symbols and coordinates into a unified molecular representation.

    :param mol: (rdkit.Chem.Mol) The molecule object containing atom symbols and coordinates.
    :param max_atoms: (int) The maximum number of atoms to consider for the molecule.
    :param remove_hs: (bool) Whether to remove hydrogen atoms from the representation.
    :param params: Additional parameters.

    :return: A batched data containing the molecular representation.
    """
    
    mol = AllChem.AddHs(mol, addCoords=True)
    atoms_h = np.array([atom.GetSymbol() for atom in mol.GetAtoms()])
    nH_idx = [i for i, atom in enumerate(atoms_h) if atom != 'H']
    atoms = atoms_h[nH_idx]
    coordinates_h = mol.GetConformer().GetPositions().astype(np.float32)
    coordinates = coordinates_h[nH_idx]

    # cropping atoms and coordinates
    if len(atoms) > max_atoms:
        idx = np.random.choice(len(atoms), max_atoms, replace=False)
        atoms = atoms[idx]
        coordinates = coordinates[idx]
    # tokens padding
    src_tokens = torch.tensor([AllChem.GetPeriodicTable().GetAtomicNumber(item) for item in atoms])
    src_pos = torch.tensor(coordinates)
    # change AllChem.RemoveHs to AllChem.RemoveAllHs
    mol = AllChem.RemoveAllHs(mol)
    node_attr, edge_index, edge_attr = get_graph(mol)
    feat = get_graph_features(edge_attr, edge_index, node_attr, drop_feat=0)
    feat['src_tokens'] = src_tokens
    feat['src_pos'] = src_pos
    return feat

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

def get_graph_features(edge_attr, edge_index, node_attr, drop_feat):
    # atom_feat_sizes = [128] + [16 for _ in range(8)]
    atom_feat_sizes = [16 for _ in range(8)]
    edge_feat_sizes = [16, 16, 16]
    edge_attr, edge_index, x = edge_attr, edge_index, node_attr
    N = x.shape[0]

    # atom feature here
    atom_feat = convert_to_single_emb(x[:, 1:], atom_feat_sizes)

    # node adj matrix [N, N] bool
    adj = np.zeros([N, N], dtype=np.int32)
    adj[edge_index[0, :], edge_index[1, :]] = 1
    degree = adj.sum(axis=-1)

    # edge feature here
    if len(edge_attr.shape) == 1:
        edge_attr = edge_attr[:, None]
    edge_feat = np.zeros([N, N, edge_attr.shape[-1]], dtype=np.int32)
    edge_feat[edge_index[0, :], edge_index[1, :]] = (
        convert_to_single_emb(edge_attr, edge_feat_sizes) + 1
    )
    shortest_path_result = floyd_warshall(adj)
    # max distance is 509
    if drop_feat:
        atom_feat[...] = 1
        edge_feat[...] = 1
        degree[...] = 1
        shortest_path_result[...] = 511
    else:
        atom_feat = atom_feat + 2
        edge_feat = edge_feat + 2
        degree = degree + 2
        shortest_path_result = shortest_path_result + 1

    # combine, plus 1 for padding
    feat = {}
    feat["atom_feat"] = torch.from_numpy(atom_feat).long()
    feat["atom_mask"] = torch.ones(N).long()
    feat["edge_feat"] = torch.from_numpy(edge_feat).long()
    feat["shortest_path"] = torch.from_numpy((shortest_path_result)).long()
    feat["degree"] = torch.from_numpy(degree).long().view(-1)
    # pair-type
    atoms = feat["atom_feat"][..., 0]
    pair_type = torch.cat(
        [
            atoms.view(-1, 1, 1).expand(-1, N, -1),
            atoms.view(1, -1, 1).expand(N, -1, -1),
        ],
        dim=-1,
    )
    feat["pair_type"] = convert_to_single_emb(pair_type, [128, 128])
    feat["attn_bias"] = torch.zeros((N + 1, N + 1), dtype=torch.float32)
    return feat

def convert_to_single_emb(x, sizes):
    assert x.shape[-1] == len(sizes)
    offset = 1
    for i in range(len(sizes)):
        assert (x[..., i] < sizes[i]).all()
        x[..., i] = x[..., i] + offset
        offset += sizes[i]
    return x


@njit
def floyd_warshall(M):
    (nrows, ncols) = M.shape
    assert nrows == ncols
    n = nrows
    # set unreachable nodes distance to 510
    for i in range(n):
        for j in range(n):
            if M[i, j] == 0:
                M[i, j] = 510

    for i in range(n):
        M[i, i] = 0

    # floyed algo
    for k in range(n):
        for i in range(n):
            for j in range(n):
                cost_ikkj = M[i, k] + M[k, j]
                if M[i, j] > cost_ikkj:
                    M[i, j] = cost_ikkj

    for i in range(n):
        for j in range(n):
            if M[i, j] >= 510:
                M[i, j] = 510
    return M