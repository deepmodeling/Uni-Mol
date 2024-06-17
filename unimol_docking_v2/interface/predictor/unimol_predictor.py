# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from .processor import Processor


class UnimolPredictor:
    def __init__(self, model_dir, mode='single', nthreads=8, conf_size=10, cluster=False, use_current_ligand_conf=False, steric_clash_fix=False):
        self.model_dir = model_dir
        self.mode = mode
        self.nthreads = nthreads
        self.use_current_ligand_conf = use_current_ligand_conf
        self.cluster = cluster
        self.steric_clash_fix = steric_clash_fix
        if self.use_current_ligand_conf:
            self.conf_size = 1
        else:
            self.conf_size = conf_size

    def preprocess(self, input_protein, input_ligand, input_docking_grid, output_ligand_name, output_ligand_dir):
        # process the input pocket.pdb and ligand.sdf, store in LMDB.
        preprocessor = Processor.build_processors(
            self.mode, self.nthreads, conf_size=self.conf_size, cluster=self.cluster,
            use_current_ligand_conf=self.use_current_ligand_conf)
        processed_data = preprocessor.preprocess(input_protein, input_ligand, input_docking_grid, output_ligand_name, output_ligand_dir)

        # return lmdb path
        return processed_data

    def predict(self, input_protein:str, 
                input_ligand:str, 
                input_docking_grid:str, 
                output_ligand_name:str, 
                output_ligand_dir:str, 
                batch_size:int):
        lmdb_name = self.preprocess(input_protein, input_ligand, input_docking_grid, output_ligand_name, output_ligand_dir)
        
        pkt_data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "example_data", "dict_pkt.txt")
        mol_data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "example_data", "dict_mol.txt")
        script_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "unimol", "infer.py")
        user_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "unimol")
        # inference
        cmd = f' cp {pkt_data_path} {os.path.abspath(output_ligand_dir)} \n\
                 cp {mol_data_path} {os.path.abspath(output_ligand_dir)} \n\
            CUDA_VISIBLE_DEVICES="0" python {script_path} --user-dir {user_dir} {os.path.abspath(output_ligand_dir)} --valid-subset {lmdb_name} \
            --results-path {os.path.abspath(output_ligand_dir)} \
            --num-workers 8 --ddp-backend=c10d --batch-size {batch_size} \
            --task docking_pose_v2 --loss docking_pose_v2 --arch docking_pose_v2 \
            --conf-size {self.conf_size} \
            --dist-threshold 8.0 --recycling 4 \
            --path {self.model_dir}  \
            --fp16 --fp16-init-scale 4 --fp16-scale-window 256 \
            --log-interval 50 --log-format simple --required-batch-size-multiple 1'
        os.system(cmd)
        
        # return results path and lmdb path
        pkl_file = os.path.join(os.path.abspath(output_ligand_dir), lmdb_name + '.pkl')
        lmdb_file = os.path.join(os.path.abspath(output_ligand_dir), lmdb_name+'.lmdb')
        return pkl_file, lmdb_file

    def postprocess(self, output_pkl, output_lmdb, output_ligand_name, output_ligand_dir, input_ligand, input_protein):
        # output the inference results to SDF file
        postprocessor = Processor.build_processors(self.mode, conf_size=self.conf_size)
        mol_list, smi_list, coords_predict_list, holo_coords_list, holo_center_coords_list, prmsd_score_list = postprocessor.postprocess_data_pre(output_pkl, output_lmdb)
        output_ligand_sdf = postprocessor.get_sdf(mol_list, smi_list, coords_predict_list, holo_center_coords_list, prmsd_score_list, output_ligand_name, output_ligand_dir, tta_times=self.conf_size)
        if self.steric_clash_fix:
            output_ligand_sdf = postprocessor.clash_fix(output_ligand_sdf, input_protein, input_ligand)
        return output_ligand_sdf

    def predict_sdf(self, input_protein:str, 
                    input_ligand:str, input_docking_grid:str, 
                    output_ligand_name:str, output_ligand_dir:str, 
                    batch_size:int = 4):
        output_pkl, output_lmdb = self.predict(input_protein, 
                                               input_ligand, 
                                               input_docking_grid, 
                                               output_ligand_name, 
                                               output_ligand_dir, 
                                               batch_size)
        output_sdf = self.postprocess(output_pkl, 
                                      output_lmdb, 
                                      output_ligand_name, 
                                      output_ligand_dir,
                                      input_ligand,
                                      input_protein)

        # return sdf path
        return input_protein, input_ligand, input_docking_grid, output_sdf

    @classmethod
    def build_predictors(cls, model_dir, mode = 'single', 
                         nthreads = 8, conf_size =10, 
                         cluster=False, use_current_ligand_conf=False, steric_clash_fix=False):
        return cls(model_dir, mode, nthreads, conf_size, 
                   cluster, use_current_ligand_conf=use_current_ligand_conf, steric_clash_fix=steric_clash_fix)    
