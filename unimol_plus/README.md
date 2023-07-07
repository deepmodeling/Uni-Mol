Highly Accurate Quantum Chemical Property Prediction with Uni-Mol+
==================================================================
[![arXiv](https://img.shields.io/badge/arXiv-2303.16982-00ff00.svg)](https://arxiv.org/abs/2303.16982) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/highly-accurate-quantum-chemical-property/graph-regression-on-pcqm4mv2-lsc)](https://paperswithcode.com/sota/graph-regression-on-pcqm4mv2-lsc?p=highly-accurate-quantum-chemical-property) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/highly-accurate-quantum-chemical-property/initial-structure-to-relaxed-energy-is2re)](https://paperswithcode.com/sota/initial-structure-to-relaxed-energy-is2re?p=highly-accurate-quantum-chemical-property)


<p align="center"><img src="figure/overview.png" width=80%></p>
<p align="center"><b>Schematic illustration of the Uni-Mol+ framework</b></p>

Uni-Mol+ is a model for quantum chemical property prediction. Firstly, given a 2D molecular graph, Uni-Mol+ generates an initial 3D conformation from inexpensive methods such as RDKit. Then, the initial conformation is iteratively optimized to its equilibrium conformation, and the optimized conformation is further used to predict the QC properties.

In the [PCQM4MV2](https://ogb.stanford.edu/docs/lsc/leaderboards/#pcqm4mv2) bencmark, Uni-Mol+ outperforms previous SOTA methods by a large margin.

| Model Settings   | # Layers   | # Param.    | Validation MAE   | Model Checkpoint | 
|------------------|------------| ----------- |------------------|------------------|
| Uni-Mol+         |     12     |   52.4 M    | 0.0696           | [link](https://github.com/dptech-corp/Uni-Mol/releases/download/v0.2/unimol_plus_pcq_base.pt)          |
| Uni-Mol+ Large   |     18     |   77 M      | 0.0693           | [link](https://github.com/dptech-corp/Uni-Mol/releases/download/v0.2/unimol_plus_pcq_large.pt)         |
| Uni-Mol+ Small   |      6     |   27.7 M    | 0.0714           | [link](https://github.com/dptech-corp/Uni-Mol/releases/download/v0.2/unimol_plus_pcq_small.pt)         |


In the [OC20](https://opencatalystproject.org/leaderboard.html) IS2RE  bencmark, Uni-Mol+ outperforms previous SOTA methods by a large margin.

| Model Settings   | # Layers   | # Param.    | Validation Mean MAE  | Test Mean MAE  | Model Checkpoint | 
|------------------|------------| ----------- |----------------------|----------------|------------------|
| Uni-Mol+         |     12     |  48.6 M     | 0.4088               | 0.4143         | [link](https://github.com/dptech-corp/Uni-Mol/releases/download/v0.2/unimol_plus_oc20_base.pt)          |


Dependencies
------------
 - [Uni-Core](https://github.com/dptech-corp/Uni-Core) with pytorch > 2.0.0, check its [Installation Documentation](https://github.com/dptech-corp/Uni-Core#installation).
 - rdkit==2022.09.3, install via `pip install rdkit==2022.09.3`


Data Preparation
----------------

First, download the data:

```bash
cd scripts
bash download.sh
```

Second, covert the 3D SDF (training set only) to lmdb file:

```bash
python get_label3d_lmdb.py
```

Finally, generate the training, validation and test datasets:

```bash
python get_3d_lmdb.py train
python get_3d_lmdb.py valid
python get_3d_lmdb.py test-dev
python get_3d_lmdb.py test-challenge
```

Inference
---------
```bash
export data_path="your_data_path"
export results_path="your_result_path"
export weight_path="your_ckp_path"
export arch="unimol_plus_large" # or "unimol_plus_base" if you use 12-layer model
bash inference.sh test-dev # or other splits
```

Training
--------
```bash
data_path="your_data_path"
save_dir="your_save_path"
lr=2e-4
batch_size=128 # per gpu batch size 128, we default use 8 GPUs
export arch="unimol_plus_large" # or "unimol_plus_base" if you use 12-layer model
bash train_pcq.sh $data_path $save_dir $lr $batch_size
```

Citation
--------

Please kindly cite this paper if you use the data/code/model.
```
@misc{lu2023highly,
      title={Highly Accurate Quantum Chemical Property Prediction with Uni-Mol+}, 
      author={Shuqi Lu and Zhifeng Gao and Di He and Linfeng Zhang and Guolin Ke},
      year={2023},
      eprint={2303.16982},
      archivePrefix={arXiv},
      primaryClass={physics.chem-ph}
}
```

License
-------

This project is licensed under the terms of the MIT license. See [LICENSE](https://github.com/dptech-corp/Uni-Mol/blob/main/LICENSE) for additional details.
