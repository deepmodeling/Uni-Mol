# Quick start

Quick start for UniMol Tools.

## Molecule property prediction

To train a model, you need to provide training data containing molecules represented as SMILES strings and corresponding target values. Targets can be real numbers for regression or binary values (0s and 1s) for classification. Leave target values blank for instances where they are unknown.

The model can be trained either on a single target ("single tasking") or on multiple targets simultaneously ("multi-tasking").

The data file can be a **CSV file with a header row**. The CSV format should have `SMILES` as input, followed by `TARGET` as the label. Note that the label is named with the `TARGET` prefix when the task involves multilabel (regression/classification). For example:

| SMILES                                          | TARGET |
| ----------------------------------------------- | ------ |
| NCCCCC(NC(CCc1ccccc1)C(=O)O)C(=O)N2CCCC2C(=O)O  | 0      |
| COc1cc(CN2CCCCC2)cc2cc(C(=O)O)c(=O)oc12         | 1      |
| CCN(CC)C(C)CN1c2ccccc2Sc3c1cccc3                | 1      |
|...                                              | ...    |

custom dict can also as the input. The dict format should be like 

```python
{'atoms':[['C','C'],['C','H','O']], 'coordinates':[coordinates_1,coordinates_2]}
```
Here is an example to train a model and make a prediction. When using Unimol V2, set `model_name='unimolv2'`.
```python
from unimol_tools import MolTrain, MolPredict
clf = MolTrain(task='classification', 
                data_type='molecule', 
                epochs=10, 
                batch_size=16, 
                metrics='auc',
                model_name='unimolv1', # avaliable: unimolv1, unimolv2
                model_size='84m', # work when model_name is unimolv2. avaliable: 84m, 164m, 310m, 570m, 1.1B.
                )
pred = clf.fit(data = train_data)
# currently support data with smiles based csv/txt file

clf = MolPredict(load_model='../exp')
res = clf.predict(data = test_data)
```
## Uni-Mol molecule and atoms level representation

Uni-Mol representation can easily be achieved as follow.

```python
import numpy as np
from unimol_tools import UniMolRepr
# single smiles unimol representation
clf = UniMolRepr(data_type='molecule', 
                 remove_hs=False,
                 model_name='unimolv1', # avaliable: unimolv1, unimolv2
                 model_size='84m', # work when model_name is unimolv2. avaliable: 84m, 164m, 310m, 570m, 1.1B.
                 )
smiles = 'c1ccc(cc1)C2=NCC(=O)Nc3c2cc(cc3)[N+](=O)[O]'
smiles_list = [smiles]
unimol_repr = clf.get_repr(smiles_list, return_atomic_reprs=True)
# CLS token repr
print(np.array(unimol_repr['cls_repr']).shape)
# atomic level repr, align with rdkit mol.GetAtoms()
print(np.array(unimol_repr['atomic_reprs']).shape)
```
## Continue training (Re-train)

```python
clf = MolTrain(
    task='regression',
    data_type='molecule',
    epochs=10,
    batch_size=16,
    save_path='./model_dir',
    remove_hs=False,
    target_cols='TARGET',
    )
pred = clf.fit(data = train_data)
# After train a model, set load_model_dir='./model_dir' to continue training

clf2 = MolTrain(
    task='regression',
    data_type='molecule',
    epochs=10,
    batch_size=16,
    save_path='./retrain_model_dir',
    remove_hs=False,
    target_cols='TARGET',
    load_model_dir='./model_dir',
    )

pred2 = clf.fit(data = retrain_data)                
```

## Distributed Data Parallel (DDP) Training

Uni-Mol Tools now supports Distributed Data Parallel (DDP) training using PyTorch. DDP allows you to train models across multiple GPUs or nodes, significantly speeding up the training process.

### Parameters
- `use_ddp`: bool, default=True, whether to enable Distributed Data Parallel (DDP).
- `use_gpu`: str, default='all', specifies which GPUs to use. `'all'` means all available GPUs, while `'0,1,2'` means using GPUs 0, 1, and 2.

### Example Usage
To enable DDP, ensure your environment supports distributed training (e.g., PyTorch with distributed support). Set `use_ddp=True` and specify the GPUs using the `use_gpu` parameter when initializing the `MolTrain` class.

#### Example for Training

```python
from unimol_tools import MolTrain

# Initialize the training class with DDP enabled
if __name__ == '__main__':
    clf = MolTrain(
        task='regression',
        data_type='molecule',
        epochs=10,
        batch_size=16,
        save_path='./model_dir',
        remove_hs=False,
        target_cols='TARGET',
        use_ddp=True,
        use_gpu="all"
        )
    pred = clf.fit(data = train_data)
```

#### Example for Molecular Representation

```python
from unimol_tools import UniMolRepr

# Initialize the UniMolRepr class with DDP enabled
if __name__ == '__main__':
    repr_model = UniMolRepr(
        data_type='molecule',
        batch_size=32,
        remove_hs=False,
        model_name='unimolv2',
        model_size='84m',
        use_ddp=True,  # Enable Distributed Data Parallel
        use_gpu='0,1'  # Use GPU 0 and 1
    )

    unimol_repr = repr_model.get_repr(smiles_list, return_atomic_reprs=True)

    # CLS token representation
    print(unimol_repr['cls_repr'])
    # Atomic-level representation
    print(unimol_repr['atomic_reprs'])
```

- **Important:** When the number of SMILES strings is small, it is not recommended to use DDP for the `get_repr` method. Communication overhead between processes may outweigh the benefits of parallel computation, leading to slower performance. In such cases, consider disabling DDP by setting `use_ddp=False`.

### Why use `if __name__ == '__main__':`

In Python, when using multiprocessing (e.g., PyTorch's `DistributedDataParallel` or other libraries requiring multiple processes), it is recommended to use the `if __name__ == '__main__':` idiom. This is because, in a multiprocessing environment, child processes may re-import the main module. Without this idiom, the code in the main module could be executed multiple times, leading to unexpected behavior or errors.

#### Common Error

If you do not use `if __name__ == '__main__':`, you might encounter the following error:

```Python
RuntimeError: 
        An attempt has been made to start a new process before the
        current process has finished its bootstrapping phase.

        This probably means that you are not using fork to start your
        child processes and you have forgotten to use the proper idiom
        in the main module:

            if __name__ == '__main__':
                freeze_support()
                ...

        The "freeze_support()" line can be omitted if the program
        is not going to be frozen to produce an executable.
```

To avoid this error, ensure that all code requiring multiprocessing is enclosed within the if `__name__ == '__main__'`: block.

### Notes
- For multi-node training, the `MASTER_ADDR` and `MASTER_PORT` environment variables can be configured as below.

```bash
export MASTER_ADDR='localhost'
export MASTER_PORT='19198'
```