Uni-Mol: A Universal 3D Molecular Representation Learning Framework 
===================================================================

[[ChemRxiv](https://chemrxiv.org/engage/chemrxiv/article-details/6294500fcd6c1c16be204e28)]

Authors: Gengmo Zhou, Zhifeng Gao, Qiankun Ding, Hang Zheng, Hongteng Xu, Zhewei Wei, Linfeng Zhang, Guolin Ke 

Uni-Mol is a universal 3D molecular pretraining framework that significantly enlarges the representation ability and application scope in drug design. 

<img src="figure/overview.png" width=80%> 
<p align="center"><b>Schematic illustration of the Uni-Mol framework</b></p>

Uni-Mol is composed of two models: a molecular pretraining model trained by 209M molecular 3D conformations; a pocket pretraining model trained by 3M candidate protein pocket data. The two models are used independently for separate tasks, and are combined when used in protein-ligand binding tasks. Uni-Mol outperforms SOTA in 14/15 molecular property prediction tasks. Moreover, Uni-Mol achieves superior performance in 3D spatial tasks, including protein-ligand binding pose prediction, molecular conformation generation, etc. 


News
----

**Jun 10 2022**: The 3D conformation data used in Uni-Mol is released.


Uni-Mol's 3D conformation data 
------------------------------

For the details of datasets, please refer to Appendix A and B in our [paper](https://chemrxiv.org/engage/chemrxiv/article-details/6294500fcd6c1c16be204e28).

There are total 6 datasets:


| Data                     | File Size  |  Download Link                                                                    | 
|--------------------------|------------|-----------------------------------------------------------------------------------|
| molecular pretrain       | 114.76GB   | https://unimol.dp.tech/data/pretrain/ligands.tar.gz                               |
| pocket pretrain          | 8.585GB    | https://unimol.dp.tech/data/pretrain/pockets.tar.gz                               |
| molecular property       | 5.412GB    | https://unimol.dp.tech/data/finetune/molecular_property_prediction.tar.gz         |
| molecular conformation   | 558.941MB  | https://unimol.dp.tech/data/finetune/conformation_generation.tar.gz               |
| pocket property          | 455.236MB  | https://unimol.dp.tech/data/finetune/pocket_property_prediction.tar.gz            |
| protein-ligand binding   | 201.492MB  | https://unimol.dp.tech/data/finetune/protein_ligand_binding_pose_prediction.tar.gz|


We use [LMDB](https://lmdb.readthedocs.io) to store data, you can use the following code snippets to read from the LMDB file.

```python
import lmdb
import numpy as np
import os
import pickle

def read_lmdb(lmdb_path):
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
    keys = list(txn.cursor().iternext(values=False))
    for idx in keys:
        datapoint_pickled = txn.get(idx)
        data = pickle.loads(datapoint_pickled)
```
We use pickle protocol 5, so Python >= 3.8 is recommended.


Code & Model (WIP)
------------------
We will release the source code and the pretrained models soon.


Citation
--------

Please kindly cite this paper if you use the data/code/model.
```
@article{zhou2022uni,
  title={Uni-Mol: A Universal 3D Molecular Representation Learning Framework},
  author={Zhou, Gengmo and Gao, Zhifeng and Ding, Qiankun and Zheng, Hang and Xu, Hongteng and Wei, Zhewei and Zhang, Linfeng and Ke, Guolin},
  journal={ChemRxiv},
  publisher={Cambridge Open Engage},
  DOI={10.26434/chemrxiv-2022-jjm0j-v2},
  year={2022}
}
```

License
-------

This project is licensed under the terms of the MIT license. See [LICENSE](https://github.com/dptech-corp/Uni-Mol/blob/main/LICENSE) for additional details.