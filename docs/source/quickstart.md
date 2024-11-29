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
clf = MolTrain(task='regression',
                data_type='molecule',
                epochs=10,
                batch_size=16,
                save_path='./model_dir',
                remove_hs=False,
                target_cols='TARGET',
                )
pred = clf.fit(data = train_data)
# After train a model, set load_model_dir='./model_dir' to continue training

clf2 = MolTrain(task='regression',
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