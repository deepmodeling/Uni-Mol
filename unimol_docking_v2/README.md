Uni-Mol Docking V2
===================================================================
[![arXiv](https://img.shields.io/badge/arXiv-2405.11769-00ff00.svg)](https://arxiv.org/abs/2405.11769) ![Static Badge](https://img.shields.io/badge/Bohrium_Apps-Uni--Mol_Docking_V2-blue?link=https%3A%2F%2Fbohrium.dp.tech%2Fapps%2Funimoldockingv2)

<p align="center"><img src="figure/bohrium_app.gif" width=60%></p>
<p align="center"><b>Uni-Mol Docking V2 Bohrium App</b></p>

We update Uni-Mol Docking to Uni-Mol Docking V2, which demonstrates a remarkable improvement in performance, accurately predicting the binding poses of 77+% of ligands in the PoseBusters benchmark with an RMSD value of less than 2.0 Å, and 75+\% passing all quality checks. This represents a significant increase from the 62% achieved by the previous Uni-Mol Docking model. Notably, our Uni-Mol Docking approach generates chemically accurate predictions, circumventing issues such as chirality inversions and steric
clashes that have plagued previous ML models.

Service of Uni-Mol Docking V2 is avaiable at https://bohrium.dp.tech/apps/unimoldockingv2

Dependencies
------------
 - [Uni-Core](https://github.com/dptech-corp/Uni-Core), check its [Installation Documentation](https://github.com/dptech-corp/Uni-Core#installation).
 - rdkit==2022.9.3, install via `pip install rdkit-pypi==2022.9.3 -i https://pypi.tuna.tsinghua.edu.cn/simple/ --trusted-host pypi.tuna.tsinghua.edu.cn`
  - biopandas==0.4.1, install via `pip install biopandas tqdm scikit-learn`

Data
----------------------------------
| Data                     | File Size  | Update Date | Download Link                                                                                                             | 
|--------------------------|------------| ----------- |---------------------------------------------------------------------------------------------------------------------------|
| Raw training data       | 4.95GB   | May 14 2024 |https://zenodo.org/records/11191555     |
| Posebusters and Astex   | 8.2MB   | Nov 16 2023 |https://github.com/deepmodeling/Uni-Mol/files/13352676/eval_sets.zip     |


Note that we use the `Posebusters V1` (428 datapoints, released in August 2023). For the latest version, please refer to [Posebusters repo](https://github.com/maabuu/posebusters).


Model weights
----------------------------------

| Model                     | File Size  |Update Date | Download Link                                                | 
|--------------------------|------------| ------------|--------------------------------------------------------------|
| unimol docking v2       | 464MB   | May 17 2024 |https://www.dropbox.com/scl/fi/sfhrtx1tjprce18wbvmdr/unimol_docking_v2_240517.pt?rlkey=5zg7bh150kcinalrqdhzmyyoo&st=n6j0nt6c&dl=0                |


Results
----------------------------------
|< 2.0 Å RMSD(% )      | PoseBusters (N=428) | Astex (N=85) | 
|--------|----|----|   
| DeepDock | 17.8 | 34.12 |
| DiffDock        |  37.9 |71.76  |
| UMol |   45| - | 
| Vina      |  52.3  | 57.65 | 
| Uni-Mol Docking     |  62.4 | 82.35 | 
| AlphaFold latest     |  73.6 | - |
| **Uni-Mol Docking V2**   |  **77.6** | **95.29**|

To reproduce the Posebusters results, we provide a notebook `interface/posebuster_demo` that includes the pipeline from data processing, model inference to metric calculation.

Training
----------------------------------

In the training script, `data_path`, `save_dir`, `finetune_mol_model`, and `finetune_pocket_model` need to be specified. 

The pretrained molecular and pocket model weights can be obtained from [Uni-Mol repo]((https://github.com/maabuu/posebusters)). We use the no_h version weights for molecule.

```
bash train.sh
```

Inference
----------------------------------

We add an interface for model inference in `interface/demo.py`.

About inputs and outpus:

- `--input-protein`: PDB file, abusolute path or raletive path, in batch_one2one mode, list of paths

- `--input-ligand`: SDF file, abusolute path or raletive path; in batch mode, list of paths

- `--input-docking-grid`: JSON file, include center coordinate and box size, abusolute path or raletive path; in batch mode, list of paths

- `--output-ligand-name`: str, the output SDF file name; in batch mode, list of names

- `--output-ligand-dir`: str, abusolute path or raletive path

In batch mode, you can save `input_protein`, `input_ligand`, `input_docking_grid`, and `output_ligand_name` to a CSV file and use `--input-batch-file` to input it.

Other parameters used:

-  `--steric-clash-fix`: The predicted SDF file will be corrected for chemical detail and clash relaxation.

- `--mode`: optional values are `single`, `batch_one2one` and `batch_one2many`. 
  - `single` represents one protein and one ligand as input. 
  - `batch_one2one` represents a batch of proteins and a batch of ligands, where the relationship is one-to-one. 
  - `batch_one2many` represents one protein and a batch of ligands, where the relationship is one-to-many.

Demo:

```
cd interface
bash demo.sh  # demo_batch_one2one.sh for batch mode
```
Or refer to this notebook `interface/posebuster_demo`.


Citation
--------

Please kindly cite this paper if you use the data/code/model.
```
@article{alcaide2024uni,
  title={Uni-Mol Docking V2: Towards Realistic and Accurate Binding Pose Prediction},
  author={Alcaide, Eric and Gao, Zhifeng and Ke, Guolin and Li, Yaqi and Zhang, Linfeng and Zheng, Hang and Zhou, Gengmo},
  journal={arXiv preprint arXiv:2405.11769},
  year={2024}
}
```

License
-------

This project is licensed under the terms of the MIT license. See [LICENSE](https://github.com/deepmodeling/Uni-Mol/blob/main/LICENSE) for additional details.
