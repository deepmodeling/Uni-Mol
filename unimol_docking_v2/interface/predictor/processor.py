# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import numpy as np
import lmdb
import pickle
import copy
import numpy as np
import pandas as pd
import json
from tqdm import tqdm
from multiprocessing import Pool
from typing import List
from sklearn.cluster import KMeans
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdMolAlign import AlignMolConformers
from biopandas.pdb import PandasPdb


class Processor:
    def __init__(self, 
        mode:str='single', 
        nthreads:int=8, 
        conf_size:int=10, 
        cluster:bool=False, 
        main_atoms:List[str]=["N", "CA", "C", "O", "H"], 
        allow_pocket_atoms:List[str]=[['C', 'H', 'N', 'O', 'S']],
        use_current_ligand_conf:bool=False
    ):
        self.mode = mode
        self.nthreads = nthreads
        self.conf_size = conf_size
        self.cluster = cluster
        self.main_atoms = main_atoms
        self.allow_pocket_atoms = allow_pocket_atoms
        if self.mode in ['batch_one2one', 'batch_one2many']:
            self.lmdb_name = 'batch_data'
        self.use_current_ligand_conf = use_current_ligand_conf

    def preprocess(self, input_protein:str, input_ligand, input_docking_grid:str, output_ligand_name:str, out_lmdb_dir:str):
        seed = 42 
        if self.mode=='single':
            supp = Chem.SDMolSupplier(input_ligand)
            mol = [mol for mol in supp if mol][0]
            ori_smiles = Chem.MolToSmiles(mol)
            smiles_list = [ori_smiles]
            input_protein = [input_protein]
            input_ligand = [input_ligand]
            input_docking_grid = [input_docking_grid]
        elif self.mode in ['batch_one2one', 'batch_one2many']:
            if self.mode == 'batch_one2many':
                input_protein = [input_protein] * len(input_ligand)
            smiles_list = []
            for i in range(len(input_ligand)):
                supp = Chem.SDMolSupplier(input_ligand[i])
                mol = [mol for mol in supp if mol][0]
                ori_smiles = Chem.MolToSmiles(mol)
                smiles_list.append(ori_smiles)
        lmdb_name = self.write_lmdb(output_ligand_name, smiles_list, input_protein, input_ligand, input_docking_grid, seed=seed, result_dir=out_lmdb_dir)
        return lmdb_name

    def single_conf_gen(self, tgt_mol, num_confs=1000, seed=42, removeHs=True):
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
        if removeHs:
            mol = Chem.RemoveHs(mol)
        return mol

    def single_conf_gen_no_MMFF(self, tgt_mol, num_confs=1000, seed=42, removeHs=True):
        mol = copy.deepcopy(tgt_mol)
        mol = Chem.AddHs(mol)
        allconformers = AllChem.EmbedMultipleConfs(
            mol, numConfs=num_confs, randomSeed=seed, clearConfs=True
        )
        if removeHs:
            mol = Chem.RemoveHs(mol)
        return mol

    def clustering_coords(self, mol, M=1000, N=100, seed=42, cluster=False, removeHs=True, gen_mode='mmff'):
        rdkit_coords_list = []
        if not cluster:
            M = N
        if gen_mode == 'mmff':
            rdkit_mol = self.single_conf_gen(mol, num_confs=M, seed=seed, removeHs=removeHs)
        elif gen_mode == 'no_mmff':
            rdkit_mol = self.single_conf_gen_no_MMFF(mol, num_confs=M, seed=seed, removeHs=removeHs)
        noHsIds = [
            rdkit_mol.GetAtoms()[i].GetIdx()
            for i in range(len(rdkit_mol.GetAtoms()))
            if rdkit_mol.GetAtoms()[i].GetAtomicNum() != 1
        ]
        ### exclude hydrogens for aligning
        AlignMolConformers(rdkit_mol, atomIds=noHsIds)
        sz = len(rdkit_mol.GetConformers())
        for i in range(sz):
            _coords = rdkit_mol.GetConformers()[i].GetPositions().astype(np.float32)
            rdkit_coords_list.append(_coords)

        ### exclude hydrogens for clustering, pick closest to centroid:
        if cluster:
            # (num_confs, num_atoms, 3)
            rdkit_coords = np.array(rdkit_coords_list)[:, noHsIds]
            # (num_confa, num_atoms, 3) -> (num_confs, num_atoms*3)
            rdkit_coords_flatten = rdkit_coords.reshape(sz, -1)
            kmeans = KMeans(n_clusters=N, random_state=seed).fit(rdkit_coords_flatten)
            # (num_clusters, num_atoms, 3)
            center_coords = kmeans.cluster_centers_.reshape(N, -1, 3)
            # (num_cluster, num_confs)
            cdist = ((center_coords[:, None] - rdkit_coords[None, :])**2).sum(axis=(-1, -2))
            # (num_confs,)
            argmin = np.argmin(cdist, axis=-1)
            coords_list = [rdkit_coords_list[i] for i in argmin]
        else:
            coords_list = rdkit_coords_list

        return coords_list

    def find_residues_in_pocket(self, pocket: dict, pdf):
        """
        Given a pocket config and a residue df, 
        return a list of residues that are in the pocket
        """
        def _get_vertex(pocket: dict, axis: str) -> tuple:
            """
            Return the minimum and maximum values of the given axis

            Args:
            pocket (dict): pocket config
            axis (str): ["x", "y", "z"]

            Returns:
            A tuple of floats.
            """
            return (
                pocket["center_{}".format(axis)] \
                    - pocket["size_{}".format(axis)] / 2,
                pocket["center_{}".format(axis)] \
                    + pocket["size_{}".format(axis)] / 2
                )
        min_x, max_x = _get_vertex(pocket, "x")
        min_y, max_y = _get_vertex(pocket, "y")
        min_z, max_z = _get_vertex(pocket, "z")
        min_array = np.array([min_x, min_y, min_z]).reshape(1,3)
        max_array = np.array([max_x, max_y, max_z]).reshape(1,3)
        patoms, pcoords, residues = [], np.empty((0,3)), []
        for i in range(len(pdf)):
            atom_info = pdf.iloc[i]
            _rescoor = np.array(atom_info[['x_coord','y_coord','z_coord']].values).reshape(-1,3)
            mapping = (_rescoor > min_array) & (_rescoor < max_array)
            if (mapping.sum(-1) == 3).sum() > 0:
                patoms += [atom_info['atom_name']]
                pcoords = np.concatenate((pcoords, _rescoor), axis=0)
                residues += [str(atom_info['chain_id'])+str(atom_info['residue_number'])]
        return patoms, pcoords, residues

    def extract_pocket(self, input_protein, input_docking_grid):
        try:
            pmol = PandasPdb().read_pdb(input_protein)
        except:
            with open('failed_pocket.txt', 'a') as f:
                f.write(' '.join(input_protein)+'\n')
            return None
        with open(input_docking_grid, "r") as file:
            box_dict = json.load(file)

        pdf = pmol.df['ATOM']
        patoms, pcoords, residues = self.find_residues_in_pocket(box_dict, pdf)
        def _filter_pocketatoms(atom):
            if atom[:2] in ['Cd','Cs', 'Cn', 'Ce', 'Cm', 'Cf', 'Cl', 'Ca', 'Cr', 'Co', 'Cu', 'Nh', 'Nd', 'Np', 'No', 'Ne', 'Na', 'Ni', \
                'Nb', 'Os', 'Og', 'Hf', 'Hg', 'Hs', 'Ho', 'He', 'Sr', 'Sn', 'Sb', 'Sg', 'Sm', 'Si', 'Sc', 'Se']:
                return None
            if atom[0] >= '0' and atom[0] <= '9':
                return _filter_pocketatoms(atom[1:])
            if atom[0] in ['Z','M','P','D','F','K','I','B']:
                return None
            if atom[0] in self.allow_pocket_atoms:
                return atom
            return atom

        atoms, index, residues_tmp = [], [], []
        for i,a in enumerate(patoms):
            output = _filter_pocketatoms(a)
            if output is not None:
                index.append(True)
                atoms.append(output)
                residues_tmp.append(residues[i])
            else:
                index.append(False)
        coordinates = pcoords[index].astype(np.float32)
        residues = residues_tmp
        patoms = atoms
        pcoords = [coordinates]
        side = [0 if a in self.main_atoms else 1 for a in patoms]
        return patoms, pcoords, residues, side, box_dict

    def parser(self, content):
        smiles, input_protein, input_ligand, input_docking_grid, seed = content
        patoms, pcoords, residues, side, config = self.extract_pocket(input_protein, input_docking_grid)
        # get ground truth conformation and generate ligand conformation
        supp = Chem.SDMolSupplier(input_ligand)
        mol = [mol for mol in supp if mol][0]
        if self.use_current_ligand_conf:
            return pickle.dumps(
                {
                    "atoms": [atom.GetSymbol() for atom in mol.GetAtoms()],
                    "coordinates": [mol.GetConformer().GetPositions().astype(np.float32)],
                    "mol_list": [mol],
                    "pocket_atoms": patoms,
                    "pocket_coordinates": pcoords,
                    "side": side,
                    "residue": residues,
                    "config": config,
                    "holo_coordinates": [mol.GetConformer().GetPositions().astype(np.float32)],
                    "holo_mol": mol,
                    "holo_pocket_coordinates": pcoords,
                    "smi": smiles,
                    "pocket": input_protein,
                },
                protocol=-1,
            )
        mol = Chem.AddHs(mol)
        smiles = Chem.MolToSmiles(mol)
        latoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
        holo_coordinates = [mol.GetConformer().GetPositions().astype(np.float32)]
        holo_mol = mol
        N = self.conf_size
        M = self.conf_size * 10
        mol_list = [mol] * N
        try:
            coordinate_list = self.clustering_coords(mol, M=M, N=N, seed=seed, cluster = self.cluster, removeHs=False, gen_mode='mmff') 
        except:
            try:
                coordinate_list = self.clustering_coords(mol, M=M, N=N, seed=seed, cluster = self.cluster, removeHs=False, gen_mode='no_mmff') 
            except:
                print(f'Failed to generate conformers with RDKit: {input_ligand}, skipped!')  
                return None
        return pickle.dumps(
            {
                "atoms": latoms,
                "coordinates": coordinate_list,
                "mol_list": mol_list,
                "pocket_atoms": patoms,
                "pocket_coordinates": pcoords,
                "side": side,
                "residue": residues,
                "config": config,
                "holo_coordinates": holo_coordinates,
                "holo_mol": holo_mol,
                "holo_pocket_coordinates": pcoords,
                "smi": smiles,
                "pocket": input_protein,
            },
            protocol=-1,
        )

    def write_lmdb(self, output_ligand_name, smiles_list, input_protein, input_ligand, input_docking_grid, seed=42, result_dir="./results"):
        os.makedirs(result_dir, exist_ok=True)
        if self.mode == 'single':
            outputfilename = os.path.join(result_dir, output_ligand_name + ".lmdb")
        elif self.mode in ['batch_one2one', 'batch_one2many']:
            outputfilename = os.path.join(result_dir, self.lmdb_name + ".lmdb")
            output_ligand_name = self.lmdb_name
        try:
            os.remove(outputfilename)
        except:
            pass
        env_new = lmdb.open(
            outputfilename,
            subdir=False,
            readonly=False,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=1,
            map_size=int(10e9),
        )
        txn_write = env_new.begin(write=True)
        print("Start preprocessing data...")
        print(f'Number of ligands: {len(smiles_list)}')
        seed = [seed] * len(input_ligand)
        content_list = zip(smiles_list, input_protein, input_ligand, input_docking_grid, seed)
        with Pool(self.nthreads) as pool:
            i = 0
            failed_num = 0
            for inner_output in tqdm(pool.imap(self.parser, content_list)):
                if inner_output is not None:
                    txn_write.put(f"{i}".encode("ascii"), inner_output)
                    i+=1
                elif inner_output is None: 
                    failed_num += 1
            txn_write.commit()
            env_new.close()
        print(f'Total num: {len(smiles_list)}, Success: {i}, Failed: {failed_num}')
        print("Done!")
        return output_ligand_name

    def load_lmdb_data(self, lmdb_path, key):
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

    def postprocess_data_pre(self, predict_file, lmdb_file):
        mol_list = self.load_lmdb_data(lmdb_file, "mol_list")
        mol_list = [Chem.RemoveHs(mol) for items in mol_list for mol in items]
        predict = pd.read_pickle(predict_file)
        smi_list, pocket_list, coords_predict_list, holo_coords_list, holo_center_coords_list, prmsd_score_list = [],[],[],[],[],[]
        for batch in predict:
            sz = batch['atoms'].size(0)
            for i in range(sz):
                smi_list.append(batch['smi_name'][i])
                pocket_list.append(batch['pocket_name'][i])
                prmsd_score_list.append(batch['prmsd_score'][i].numpy().astype(np.float32))
                
                token_mask = batch['atoms'][i]>2

                holo_coordinates = batch['holo_coordinates'][i]
                holo_coordinates = holo_coordinates[token_mask,:]
                holo_coordinates = holo_coordinates.numpy().astype(np.float32)

                coord_predict = batch['coord_predict'][i]
                coord_predict = coord_predict[token_mask,:]
                coord_predict = coord_predict.numpy().astype(np.float32)

                holo_center_coordinates = batch["holo_center_coordinates"][i][:3]
                holo_center_coordinates.numpy().astype(np.float32)

                holo_center_coords_list.append(holo_center_coordinates)        
                coords_predict_list.append(coord_predict)
                holo_coords_list.append(holo_coordinates)

        return mol_list, smi_list, coords_predict_list, holo_coords_list, holo_center_coords_list, prmsd_score_list

    def set_coord(self, mol, coords):
        for i in range(coords.shape[0]):
            mol.GetConformer(0).SetAtomPosition(i, coords[i].tolist())
        return mol

    def add_coord(self, mol, xyz):
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
    
    def get_sdf(self, mol_list, smi_list, coords_predict_list, holo_center_coords_list, prmsd_score_list, output_ligand_name, output_ligand_dir, tta_times=10):
        print("Start converting model predictions into sdf files...")
        output_ligand_list = []
        if self.mode == 'single':
            output_ligand_name = [output_ligand_name]
        for i in tqdm(range(len(smi_list)//tta_times)):
            coords_predict_tta = coords_predict_list[i*tta_times:(i+1)*tta_times]
            prmsd_score_tta = prmsd_score_list[i*tta_times:(i+1)*tta_times]
            mol_list_tta = mol_list[i*tta_times:(i+1)*tta_times]
            holo_center_coords_tta = holo_center_coords_list[i*tta_times:(i+1)*tta_times]
            idx = np.argmin(prmsd_score_tta)
            bst_predict_coords = coords_predict_tta[idx]
            mol = mol_list_tta[idx]
            mol = self.set_coord(mol, bst_predict_coords)
            holo_center_coords = holo_center_coords_tta[idx]
            mol = self.add_coord(mol, holo_center_coords.numpy())
            os.makedirs(output_ligand_dir, exist_ok=True)
            outputfilename = os.path.join(output_ligand_dir, str(output_ligand_name[i]) + '.sdf')
            try:
                os.remove(outputfilename)
            except:
                pass
            Chem.MolToMolFile(mol, outputfilename)
            output_ligand_list.append(outputfilename)
        print("Done!")
        if self.mode == 'single':
            return output_ligand_list[0]
        elif self.mode in ['batch_one2one', 'batch_one2many']:
            return output_ligand_list
    
    def single_clash_fix(self, input_content):
        input_ligand, output_ligand, label_ligand, pocket_mol = input_content
        script_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "unimol", "scripts", "6tsr.py")
        cmd = "python {} --input-ligand {} --output-ligand {} --label-ligand {} --pocket-mol {} --num-6t-trials 5".format(
            script_path, input_ligand, output_ligand, label_ligand, pocket_mol
        )
        os.system(cmd)
        return True

    def clash_fix(self, predicted_ligand, input_protein, input_ligand):
        if self.mode=='batch_one2many':
            input_protein = [input_protein] * len(input_ligand)
        elif self.mode == 'single':
            input_ligand = [input_ligand]
            input_protein = [input_protein]
            predicted_ligand = [predicted_ligand]
        input_content = zip(predicted_ligand, predicted_ligand, input_ligand, input_protein)

        with Pool(self.nthreads) as pool:
            for inner_output in tqdm(
                pool.imap(self.single_clash_fix, input_content), total=len(input_ligand) if type(input_ligand) is list else 1
            ):
                if not inner_output:
                    print("fail to clash fix")
        return predicted_ligand

    @classmethod
    def build_processors(
        cls, 
        mode='single', 
        nthreads = 8, 
        conf_size = 10, 
        cluster=False,
        use_current_ligand_conf:bool=False
    ):
        return cls(
            mode, 
            nthreads, 
            conf_size=conf_size, 
            cluster=cluster, 
            use_current_ligand_conf=use_current_ligand_conf
        )