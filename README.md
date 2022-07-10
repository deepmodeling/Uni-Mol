Uni-Mol: A Universal 3D Molecular Representation Learning Framework 
===================================================================

[[ChemRxiv](https://chemrxiv.org/engage/chemrxiv/article-details/6294500fcd6c1c16be204e28)]

Authors: Gengmo Zhou, Zhifeng Gao, Qiankun Ding, Hang Zheng, Hongteng Xu, Zhewei Wei, Linfeng Zhang, Guolin Ke 

Uni-Mol is a universal 3D molecular pretraining framework that significantly enlarges the representation ability and application scope in drug design. 

<p align="center"><img src="figure/overview.png" width=80%></p>
<p align="center"><b>Schematic illustration of the Uni-Mol framework</b></p>

Uni-Mol is composed of two models: a molecular pretraining model trained by 209M molecular 3D conformations; a pocket pretraining model trained by 3M candidate protein pocket data. The two models are used independently for separate tasks, and are combined when used in protein-ligand binding tasks. Uni-Mol outperforms SOTA in 14/15 molecular property prediction tasks. Moreover, Uni-Mol achieves superior performance in 3D spatial tasks, including protein-ligand binding pose prediction, molecular conformation generation, etc. 


News
----

**Jul 10 2022**: Pretraining codes are released.
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
| molecular conformation   | 558.941MB  | TBA |
| pocket property          | 455.236MB  | https://unimol.dp.tech/data/finetune/pocket_property_prediction.tar.gz            |
| protein-ligand binding   | 201.492MB  | TBA |


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


Dependencies
------------
 - [Uni-Core](https://github.com/dptech-corp/Uni-Core), install via `pip install git+git://github.com/dptech-crop/Uni-Core.git@stable#egg=Uni-Core`
 - rdkit==2021.09.5, install via `conda install -y -c conda-forge rdkit==2021.09.5`
 - PyTDC, install via `pip install PyTDC`

As Uni-Core need compile CUDA kernels in installation, we also provide a docker image to save your effects. To use the GPU within docker you need to install nvidia-docker2 first. Use the following command to pull the docker image:  

```bash
docker pull dptechnology/unimol:pytorch1.11.0-cuda11.3
```

Molecular Pretraining
---------------------

```bash
data_path=./example_data/molecule/ # replace to your data path
save_dir=./save # replace to your save path
n_gpu=8
MASTER_PORT=10086
lr=1e-4
wd=1e-4
batch_size=16
update_freq=1
masked_token_loss=1
masked_coord_loss=5
masked_dist_loss=10
x_norm_loss=0.01
delta_pair_repr_norm_loss=0.01
mask_prob=0.15
only_polar=-1
noise_type='uniform'
noise=1.0
seed=1
warmup_steps=10000
max_steps=1000000

export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1
python -m torch.distributed.launch --nproc_per_node=$n_gpu --master_port=$MASTER_PORT $(which unicore-train) $data_path  --user-dir ./unimol --train-subset train --valid-subset valid \
       --num-workers 8 --ddp-backend=c10d \
       --task unimol --loss unimol --arch unimol_base  \
       --optimizer adam --adam-betas '(0.9, 0.99)' --adam-eps 1e-6 --clip-norm 1.0 --weight-decay $wd \
       --lr-scheduler polynomial_decay --lr $lr --warmup-updates $warmup_steps --total-num-update $max_steps \
       --update-freq $update_freq --seed $seed \
       --fp16 --fp16-init-scale 4 --fp16-scale-window 256 --tensorboard-logdir $save_dir/tsb \
       --max-update $max_steps --log-interval 10 --log-format simple \
       --save-interval-updates 10000 --validate-interval-updates 10000 --keep-interval-updates 10 --no-epoch-checkpoints  \
       --masked-token-loss $masked_token_loss --masked-coord-loss $masked_coord_loss --masked-dist-loss $masked_dist_loss \
       --x-norm-loss $x_norm_loss --delta-pair-repr-norm-loss $delta_pair_repr_norm_loss \
       --mask-prob $mask_prob --noise-type $noise_type --noise $noise --batch-size $batch_size \
       --save-dir $save_dir  --only-polar $only_polar

```
The above setting is for 8 V100 GPUs, and the batch size is 128 (`n_gpu * batch_size * update_freq`). You may need to change `batch_size` or `update_freq` according to your environment. 

Pocket Pretraining
------------------

```bash
data_path=./example_data/pocket/ # replace to your data path
save_dir=./save # replace to your save path
n_gpu=8
MASTER_PORT=10086
dict_name='dict_coarse.txt'
lr=1e-4
wd=1e-4
batch_size=16
update_freq=1
masked_token_loss=1
masked_coord_loss=1
masked_dist_loss=1
mask_prob=0.15
noise_type='uniform'
noise=1.0
seed=1
warmup_steps=10000
max_steps=1000000

export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1
python -m torch.distributed.launch --nproc_per_node=$n_gpu --master_port=$MASTER_PORT $(which unicore-train) $data_path  --user-dir ./unimol --train-subset train --valid-subset valid \
       --num-workers 8 --ddp-backend=c10d \
       --task unimol_pocket --loss unimol --arch unimol_base  \
       --optimizer adam --adam-betas '(0.9, 0.99)' --adam-eps 1e-6 --clip-norm 1.0 --weight-decay $wd \
       --lr-scheduler polynomial_decay --lr $lr --warmup-updates $warmup_steps --total-num-update $max_steps \
       --update-freq $update_freq --seed $seed \
       --dict-name $dict_name \
       --fp16 --fp16-init-scale 4 --fp16-scale-window 256 --tensorboard-logdir $save_dir/tsb \
       --max-update $max_steps --log-interval 10 --log-format simple \
       --save-interval-updates 10000 --validate-interval-updates 10000 --keep-interval-updates 10 \
       --masked-token-loss $masked_token_loss --masked-coord-loss $masked_coord_loss --masked-dist-loss $masked_dist_loss \
       --mask-prob $mask_prob --noise-type $noise_type --noise $noise --batch-size $batch_size \
       --save-dir $save_dir

```
The above setting is for 8 V100 GPUs, and the batch size is 128 (`n_gpu * batch_size * update_freq`). You may need to change `batch_size` or `update_freq` according to your environment. 


Code & Model Release (WIP)
--------------------------

| Data                     | Code               |  Model Checkpoint  | 
|--------------------------|--------------------|--------------------|
| molecular pretrain       | :heavy_check_mark: |                    |
| pocket pretrain          | :heavy_check_mark: |                    |
| molecular property       | :heavy_check_mark: |                    |
| molecular conformation   |                    |                    |
| pocket property          |                    |                    |
| protein-ligand binding   |                    |                    |


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
