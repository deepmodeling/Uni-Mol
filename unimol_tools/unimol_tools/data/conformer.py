# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
import warnings
from scipy.spatial import distance_matrix
RDLogger.DisableLog('rdApp.*') 
warnings.filterwarnings(action='ignore')
from unicore.data import Dictionary
from multiprocessing import Pool
from tqdm import tqdm
import pathlib
from ..utils import logger
from ..config import MODEL_CONFIG

WEIGHT_DIR = os.path.join(pathlib.Path(__file__).resolve().parents[1], 'weights')

class ConformerGen(object):
    def __init__(self, **params):
        self._init_features(**params)

    def _init_features(self, **params):
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
        self.dictionary = Dictionary.load(os.path.join(WEIGHT_DIR, self.dict_name))
        self.dictionary.add_symbol("[MASK]", is_special=True)

    def single_process(self, smiles):
        if self.method == 'rdkit_random':
            atoms, coordinates = inner_smi2coords(smiles, seed=self.seed, mode=self.mode, remove_hs=self.remove_hs)
            return coords2unimol(atoms, coordinates, self.dictionary, self.max_atoms)
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
        logger.info('Failed to generate conformers for {:.2f}% of molecules.'.format(failed_cnt*100))
        failed_3d_cnt = np.mean([(item['src_coord'][:,2]==0.0).all() for item in inputs])
        logger.info('Failed to generate 3d conformers for {:.2f}% of molecules.'.format(failed_3d_cnt*100))
        return inputs

def inner_smi2coords(smi, seed=42, mode='fast', remove_hs=True):
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

def coords2unimol(atoms, coordinates, dictionary, max_atoms=256, **params):
    atoms = np.array(atoms)
    coordinates = np.array(coordinates).astype(np.float32)
    ### cropping atoms and coordinates
    if len(atoms)>max_atoms:
        idx = np.random.choice(len(atoms), max_atoms, replace=False)
        atoms = atoms[idx]
        coordinates = coordinates[idx]
    ### tokens padding
    src_tokens = np.array([dictionary.bos()] + [dictionary.index(atom) for atom in atoms] + [dictionary.eos()])
    src_distance = np.zeros((len(src_tokens), len(src_tokens)))
    ### coordinates normalize & padding
    src_coord = coordinates - coordinates.mean(axis=0)
    src_coord = np.concatenate([np.zeros((1,3)), src_coord, np.zeros((1,3))], axis=0)
    ### distance matrix
    src_distance = distance_matrix(src_coord, src_coord)
    ### edge type 
    src_edge_type = src_tokens.reshape(-1, 1) * len(dictionary) + src_tokens.reshape(1, -1)

    return {
            'src_tokens': src_tokens.astype(int), 
            'src_distance': src_distance.astype(np.float32), 
            'src_coord': src_coord.astype(np.float32), 
            'src_edge_type': src_edge_type.astype(int),
            }

def coords2unimol_mof(atoms, coordinates, dictionary, max_atoms=256):
    atoms = np.array(atoms)
    coordinates = np.array(coordinates).astype(np.float32)
    ### cropping atoms and coordinates
    if len(atoms)>max_atoms:
        idx = np.random.choice(len(atoms), max_atoms, replace=False)
        atoms = atoms[idx]
        coordinates = coordinates[idx]
    ### tokens padding
    src_tokens = np.array([dictionary.bos()] + [dictionary.index(atom) for atom in atoms] + [dictionary.eos()])
    src_distance = np.zeros((len(src_tokens), len(src_tokens)))
    ### coordinates normalize & padding
    src_coord = coordinates - coordinates.mean(axis=0)
    src_coord = np.concatenate([np.zeros((1,3)), src_coord, np.zeros((1,3))], axis=0)
    ### distance matrix
    # src_distance = distance_matrix(src_coord, src_coord)
    src_distance = np.zeros((len(src_tokens), len(src_tokens)))
    src_distance[1:-1,1:-1] = distance_matrix(src_coord[1:-1], src_coord[1:-1])
                            
    ### edge type 
    src_edge_type = src_tokens.reshape(-1, 1) * len(dictionary) + src_tokens.reshape(1, -1)

    return {
            'src_tokens': src_tokens.astype(int), 
            'src_distance': src_distance.astype(np.float32), 
            'src_coord': src_coord.astype(np.float32), 
            'src_edge_type': src_edge_type.astype(int),
            }