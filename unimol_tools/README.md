# unimol tools for various prediction and downstreams.

## details can be found in bohrium notebook
* [unimol_property_predict](https://bohrium.dp.tech/notebook/298bcead4f614971bb62fbeef2e9db16)
* [unimol representation](https://bohrium.dp.tech/notebook/f39a7a8836134cca8e22c099dc9654f8)
* [unimol docking](https://bohrium.dp.tech/notebook/80c6893e315641e6bd05567c9a6adbbb)
* [unimol for mof absorption prediction](https://bohrium.dp.tech/notebook/cca98b584a624753981dfd5f8bb79674)

## install
```python
## clone repo
git clone https://github.com/dptech-corp/Uni-Mol.git
cd Uni-Mol/unimol_tools/unimol_tools

## download pretrained weights
wget https://github.com/dptech-corp/Uni-Mol/releases/download/v0.1/mol_pre_all_h_220816.pt
wget https://github.com/dptech-corp/Uni-Mol/releases/download/v0.1/mol_pre_no_h_220816.pt
wget https://github.com/dptech-corp/Uni-Mol/releases/download/v0.1/pocket_pre_220816.pt
wget https://github.com/dptech-corp/Uni-Mol/releases/download/v0.1/mof_pre_no_h_CORE_MAP_20230505.pt
wget https://github.com/dptech-corp/Uni-Mol/releases/download/v0.1/mp_all_h_230313.pt
mkdir -p weights
mv *.pt weights/

## install
cd ..
pip install -r requirements.txt
python setup.py install
```
## finetune
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
## unimol repr
```python
import torch as th
from unimol_tools import UniMolRepr
clf = UniMolRepr(data_type='molecule')
smiles = ['CCO', 'CCC', 'CCCC']
reprs = clf.get_repr(smiles)
(
    # dict_keys(['cls_repr', 'atomic_reprs'])
    reprs.keys(),  
    # torch.Size([3, 512])
    th.tensor(reprs["cls_repr"]).shape,  
    # [torch.Size([9, 512]), torch.Size([11, 512]), torch.Size([14, 512])])
    [th.tensor(x).shape for x in reprs["atomic_reprs"]]  
) 
```

## unimol mof absorption prediction
```python
from unimol_tools import MOFPredictor
clf = MOFPredictor()
GAS2ID = {
        "UNK":0,
        "CH4":1, 
        "CO2":2, 
        "Ar":3, 
        "Kr":4, 
        "Xe":5, 
        "O2":6,
        "He":7, 
        "N2":8, 
        "H2":9,
    }
gas = 'CH4'
mof_name = 'la304204k_si_003_clean'
res = clf.predict_grid(cif_path=f'../examples/mof/{mof_name}.cif',
                        gas=gas,
                        temperature_list=[190,298],
                        pressure_bins=8)
print(res.head())
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
@article{wang2023metal,
  title={Metal-organic frameworks meet Uni-MOF: a revolutionary gas adsorption detector},
  author={Wang, Jingqi and Liu, Jiapeng and Wang, Hongshuai and Ke, Guolin and Zhang, Linfeng and Wu, Jianzhong and Gao, Zhifeng and Lu, Diannan},
  year={2023}
}
@article{gao2023uni,
  title={Uni-QSAR: an Auto-ML Tool for Molecular Property Prediction},
  author={Gao, Zhifeng and Ji, Xiaohong and Zhao, Guojiang and Wang, Hongshuai and Zheng, Hang and Ke, Guolin and Zhang, Linfeng},
  journal={arXiv preprint arXiv:2304.12239},
  year={2023}
}
```

License
-------

This project is licensed under the terms of the MIT license. See LICENSE for additional details.
