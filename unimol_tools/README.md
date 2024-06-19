# unimol tools for various prediction and downstreams.

Documentation of Uni-Mol tools is available at https://unimol.readthedocs.io/en/latest/

## details can be found in bohrium notebook
* [unimol property predict](https://bohrium.dp.tech/notebook/298bcead4f614971bb62fbeef2e9db16)
* [unimol representation](https://bohrium.dp.tech/notebook/f39a7a8836134cca8e22c099dc9654f8)

## install
- pytorch is required, please install pytorch according to your environment. if you are using cuda, please install pytorch with cuda. More details can be found at https://pytorch.org/get-started/locally/
- currently, rdkit needs with numpy<2.0.0, please install rdkit with numpy<2.0.0.
```python
## dependencies installation
pip install -r requirements.txt
## clone repo
git clone https://github.com/dptech-corp/Uni-Mol.git
cd Uni-Mol/unimol_tools/unimol_tools

## download pretrained weights
wget https://github.com/dptech-corp/Uni-Mol/releases/download/v0.1/mol_pre_all_h_220816.pt
wget https://github.com/dptech-corp/Uni-Mol/releases/download/v0.1/mol_pre_no_h_220816.pt
wget https://github.com/dptech-corp/Uni-Mol/releases/download/v0.1/oled_pre_no_h_230101.pt

mkdir -p weights
mv *.pt weights/

## install
cd ..
python setup.py install
```

## News
- 2024-06-20: unimol_tools v0.1.0 released, we remove the dependency of Uni-Core. And we will publish to pypi soon.
- 2024-03-20: unimol_tools documents is available at https://unimol.readthedocs.io/en/latest/

## molecule property prediction
```python
from unimol_tools import MolTrain, MolPredict
clf = MolTrain(task='classification', 
                data_type='molecule', 
                epochs=10, 
                batch_size=16, 
                metrics='auc',
                )
pred = clf.fit(data = data)
# currently support data with smiles based csv/txt file, and
# custom dict of {'atoms':[['C','C],['C','H','O']], 'coordinates':[coordinates_1,coordinates_2]}

clf = MolPredict(load_model='../exp')
res = clf.predict(data = data)
```
## unimol molecule and atoms level representation
```python
import numpy as np
from unimol_tools import UniMolRepr
# single smiles unimol representation
clf = UniMolRepr(data_type='molecule', remove_hs=False)
smiles = 'c1ccc(cc1)C2=NCC(=O)Nc3c2cc(cc3)[N+](=O)[O]'
smiles_list = [smiles]
unimol_repr = clf.get_repr(smiles_list, return_atomic_reprs=True)
# CLS token repr
print(np.array(unimol_repr['cls_repr']).shape)
# atomic level repr, align with rdkit mol.GetAtoms()
print(np.array(unimol_repr['atomic_reprs']).shape)
```

Please kindly cite our papers if you use the data/code/model.
```
@inproceedings{
  zhou2023unimol,
  title={Uni-Mol: A Universal 3D Molecular Representation Learning Framework},
  author={Gengmo Zhou and Zhifeng Gao and Qiankun Ding and Hang Zheng and Hongteng Xu and Zhewei Wei and Linfeng Zhang and Guolin Ke},
  booktitle={The Eleventh International Conference on Learning Representations },
  year={2023},
  url={https://openreview.net/forum?id=6K2RM6wVqKu}
}
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

This project is licensed under the terms of the MIT license. See LICENSE for additional details.
