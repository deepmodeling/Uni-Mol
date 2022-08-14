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


| Data                     | File Size  | Update Date | Download Link                                                                    | 
|--------------------------|------------| ----------- |-----------------------------------------------------------------------------------|
| molecular pretrain       | 114.76GB   | Jun 10 2022 |https://unimol.dp.tech/data/pretrain/ligands.tar.gz                               |
| pocket pretrain          | 8.585GB    | Jun 10 2022 |https://unimol.dp.tech/data/pretrain/pockets.tar.gz                               |
| molecular property       | 3.506GB    | Jul 10 2022 |https://unimol.dp.tech/data/finetune/molecular_property_prediction.tar.gz         |
| molecular conformation   | 8.331GB    | Jul 19 2022 |https://unimol.dp.tech/data/finetune/conformation_generation.tar.gz               |
| pocket property          | 455.239MB  | Jul 19 2022 |https://unimol.dp.tech/data/finetune/pocket_property_prediction.tar.gz            |
| protein-ligand binding   |            |             | TBA |


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
 - [Uni-Core](https://github.com/dptech-corp/Uni-Core), check its [Installation Documentation](https://github.com/dptech-corp/Uni-Core#installation).
 - rdkit==2021.09.5, install via `conda install -y -c conda-forge rdkit==2021.09.5`

To use GPUs within docker you need to [install nvidia-docker-2](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker) first. Use the following command to pull the docker image:

```bash
docker pull dptechnology/unimol:latest-pytorch1.11.0-cuda11.3
```

Molecular Pretraining
---------------------

```bash
data_path=./example_data/molecule/ # replace to your data path
save_dir=./save/ # replace to your save path
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
save_dir=./save/ # replace to your save path
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


Molecular Property Prediction
------------------

```bash
data_path='./molecular_property_prediction'  # replace to your data path
save_dir='./save_finetune'  # replace to your save path
n_gpu=4
MASTER_PORT=10086
dict_name='dict.txt'
weight_path='./weights/checkpoint.pt'  # replace to your ckpt path
task_name='qm9dft'  # molecular property prediction task name 
task_num=3
loss_func='finetune_smooth_mae'
lr=1e-4
batch_size=32
epoch=40
dropout=0
warmup=0.06
local_batch_size=32
only_polar=-1
conf_size=11
seed=0

if [ "$task_name" == "qm7dft" ] || [ "$task_name" == "qm8dft" ] || [ "$task_name" == "qm9dft" ]; then
	metric="valid_agg_mae"
elif [ "$task_name" == "esol" ] || [ "$task_name" == "freesolv" ] || [ "$task_name" == "lipo" ]; then
    metric="valid_agg_rmse"
else 
    metric="valid_agg_auc"
fi

export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1
update_freq=`expr $batch_size / $local_batch_size`
python -m torch.distributed.launch --nproc_per_node=$n_gpu --master_port=$MASTER_PORT $(which unicore-train) $data_path --task-name $task_name --user-dir ./unimol --train-subset train --valid-subset valid \
       --conf-size $conf_size \
       --num-workers 8 --ddp-backend=c10d \
       --dict-name $dict_name \
       --task mol_finetune --loss $loss_func --arch unimol_base  \
       --classification-head-name $task_name --num-classes $task_num \
       --optimizer adam --adam-betas '(0.9, 0.99)' --adam-eps 1e-6 --clip-norm 1.0 \
       --lr-scheduler polynomial_decay --lr $lr --warmup-ratio $warmup --max-epoch $epoch --batch-size $local_batch_size --pooler-dropout $dropout\
       --update-freq $update_freq --seed $seed \
       --fp16 --fp16-init-scale 4 --fp16-scale-window 256 \
       --log-interval 100 --log-format simple \
       --validate-interval 1 \
       --finetune-from-model $weight_path \
       --best-checkpoint-metric $metric --patience 20 \
       --save-dir $save_dir --only-polar $only_polar \
       --reg

# --reg, for regression task
# --maximize-best-checkpoint-metric, for classification task

```

To speed up finetune, we set `n_gpu=4` for QM9, MUV, PCBA and HIV, and `n_gpu=1` for others, and the batch size is `n_gpu * local_batch_size * update_freq`.
For regression task, we set `--reg`. 
For classification task, we set `--maximize-best-checkpoint-metric`.

Each task will be run by 3 different seeds. We choose the checkpoint with the best metric on validation set and report the mean and standard deviation of the three results on the test set.

For the selection of `task_num` and other hyperparameters, please refer to the following table:

- Classification

|Dataset      | BBBP | BACE | ClinTox | Tox21 | ToxCast | SIDER | HIV | PCBA | MUV |
|--------|----|----|----|----|----|-----|-----|----|-----|       
| task_num |  2 | 2 | 2 | 12 | 617 | 27 | 2 | 128 | 17 |
| lr         |  4e-4 | 1e-4 | 5e-5 | 1e-4 | 1e-4 | 5e-4 | 5e-5 | 1e-4 | 2e-5 |
| batch_size |  128 | 64 | 256 | 128 | 64 | 32 | 256 | 128 | 128 |
| epoch      |  40 | 60 | 100 | 80 | 80 | 80 | 5 | 20 | 40 |
| dropout    |  0 | 0.1 | 0.5 | 0.1 | 0.1 | 0 | 0.2 | 0.1 | 0 |
| warmup     |  0.06 | 0.06 | 0.1 | 0.06 | 0.06 | 0.1 | 0.1 | 0.06 | 0 |

For BBBP, BACE and HIV, we set `loss_func=finetune_cross_entropy`.
For ClinTox, Tox21, ToxCast, SIDER, HIV, PCBA and MUV, we set `loss_func=multi_task_BCE`.

- Regression

| Dataset | ESOL | FreeSolv | Lipo | QM7 | QM8 | QM9 |
|----- | ---- | ---- | ---- | ---- | --- | --- |
| task_num | 1 | 1 |  1 | 1  | 12 | 3 |
| lr         | 5e-4 | 1e-4 |  1e-4 | 3e-4  | 1e-4 | 1e-4 |
| batch_size | 256 | 64 |  32 | 32  | 32 | 128 |
| epoch      | 100 | 40 |  80 | 100  | 40 | 40 |
| dropout    | 0.2 | 0.1 |  0.1 | 0  | 0 | 0 |
| warmup     | 0.06 | 0.06 | 0.06 | 0.06  | 0.06 | 0.06 |


For ESOL, FreeSolv and Lipo, we set `loss_func=finetune_mse`.
For QM7, QM8 and QM9, we set `loss_func=finetune_smooth_mae`.

**NOTE**: You'd better align the `only_polar` parameter in pretraining and finetuning: `-1` for all hydrogen, `0` for no hydrogen, `1` for polar hydrogen.


Molecular conformation generation
------------------

1. Finetune Uni-Mol pretrained model on the training set of the conformation generation task: 

```bash
data_path='./conformation_generation'  # replace to your data path
save_dir='./save_confgen'  # replace to your save path
n_gpu=1
MASTER_PORT=10086
dict_name='dict.txt'
weight_path='./weights/checkpoint.pt'  # replace to your ckpt path
task_name='qm9'  # or 'drugs', conformation generation task name, as a part of complete data path
dist=8.0
recycles=4
coord_loss=1
distance_loss=1
beta=4.0
smooth=0.1
topN=20
lr=1e-4
batch_size=128
epoch=50
dropout=0.2
warmup=0.06
update_freq=1

export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1
python -m torch.distributed.launch --nproc_per_node=$n_gpu --master_port=$MASTER_PORT $(which unicore-train) $data_path --task-name $task_name --user-dir ./unimol --train-subset train --valid-subset valid \
       --num-workers 8 --ddp-backend=c10d \
       --task mol_confG --loss mol_confG --arch mol_confG  \
       --optimizer adam --adam-betas '(0.9, 0.99)' --adam-eps 1e-6 --clip-norm 1.0 \
       --lr-scheduler polynomial_decay --lr $lr --warmup-ratio $warmup --max-epoch $epoch --batch-size $batch_size \
       --update-freq $update_freq --seed 1 \
       --fp16 --fp16-init-scale 4 --fp16-scale-window 256 \
       --log-interval 100 --log-format simple --tensorboard-logdir $save_dir/tsb \
       --validate-interval 1 --keep-last-epochs 10 \
       --keep-interval-updates 10 --best-checkpoint-metric loss  --patience 50 --all-gather-list-size 102400 \
       --finetune-from-model $weight_path --save-dir $save_dir \
       --coord-loss $coord_loss --distance-loss $distance_loss --dist-threshold $dist \
       --num-recycles $recycles --beta $beta --smooth $smooth --topN $topN \
       --find-unused-parameters

```

2. Generate initial RDKit conformations for inference: 

- Run this command, 

```bash
python ./unimol/conf_gen_cal_metrics.py --mode gen_data --nthreads ${Num of threads} --reference-file ${Reference file dir} --output-dir ${Generated initial data dir}
```

3. Inference on the generated RDKit initial conformations:

```bash
data_path='./conformation_generation'  # replace to your data path
results_path='./infer_confgen'  # replace to your results path
weight_path='./save_confgen/checkpoint_best.pt'  # replace to your ckpt path
batch_size=128
task_name='qm9'  # or 'drugs', conformation generation task name 

python ./unimol/conf_gen_infer.py --user-dir ./unimol $data_path --valid-subset test \
       --results-path $results_path \
       --num-workers 8 --ddp-backend=c10d --batch-size $batch_size --task mol_confG \
       --model-overrides "{'task_name': '${task_name}'}" \
       --path $weight_path \
       --fp16 --fp16-init-scale 4 --fp16-scale-window 256 \
       --log-interval 50 --log-format simple 
```

4. Calculate metrics on the results of inference: 

- Run this command
```bash
python ./unimol/conf_gen_cal_metrics.py --mode cal_metrics --threshold ${Threshold for cal metrics} --nthreads ${Num of threads} --predict-file ${Your inference file dir} --reference-file ${Your reference file dir}
```


Pocket Property Prediction
------------------

```bash
data_path='./pocket_property_prediction'  # replace to your data path
save_dir='./save_finetune'  # replace to your save path
n_gpu=1
MASTER_PORT=10086
dict_name='dict_coarse.txt'
weight_path='./weights/checkpoint.pt'
task_name='drugabbility'  # or 'nrdld', pocket property prediction dataset folder name 
lr=3e-4
batch_size=32
epoch=20
dropout=0
warmup=0.1
local_batch_size=32
seed=1

if [ "$task_name" == "drugabbility" ]; then
       metric="valid_mse"
       loss_func='finetune_mse_pocket'
       task_num=1
else
       metric='loss'
       loss_func='finetune_cross_entropy_pocket'
       task_num=2
fi

export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1
update_freq=`expr $batch_size / $local_batch_size`
python -m torch.distributed.launch --nproc_per_node=$n_gpu --master_port=$MASTER_PORT $(which unicore-train) $data_path --task-name $task_name --user-dir ./unimol --train-subset train --valid-subset valid \
       --num-workers 8 --ddp-backend=c10d \
       --dict-name $dict_name \
       --task pocket_finetune --loss $loss_func --arch unimol_base  \
       --classification-head-name $task_name --num-classes $task_num \
       --optimizer adam --adam-betas '(0.9, 0.99)' --adam-eps 1e-6 --clip-norm 1.0 \
       --lr-scheduler polynomial_decay --lr $lr --warmup-ratio $warmup --max-epoch $epoch --batch-size $local_batch_size --pooler-dropout $dropout \
       --update-freq $update_freq --seed $seed \
       --fp16 --fp16-init-scale 4 --fp16-scale-window 256 \
       --log-interval 100 --log-format simple \
       --validate-interval 1 --finetune-from-model $weight_path \
       --best-checkpoint-metric $metric --patience 2000 \
       --save-dir $save_dir --remove-hydrogen 

# --maximize-best-checkpoint-metric, for classification task

```

The batch size is `n_gpu * local_batch_size * update_freq`.
For classification task, we set `--maximize-best-checkpoint-metric`.

We choose the checkpoint with the best metric on validation set or training set.


WIP
---

- [ ] code & data for protein-ligand binding
- [ ] model checkpoints



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
