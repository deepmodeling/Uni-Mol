# Uni-Mol tools for various prediction and downstreams.

Documentation of Uni-Mol tools is available at https://unimol.readthedocs.io/en/latest/

## Details can be found in bohrium notebook
* [unimol property predict](https://bohrium.dp.tech/notebook/298bcead4f614971bb62fbeef2e9db16)
* [unimol representation](https://bohrium.dp.tech/notebook/f39a7a8836134cca8e22c099dc9654f8)

## Install
- pytorch is required, please install pytorch according to your environment. if you are using cuda, please install pytorch with cuda. More details can be found at https://pytorch.org/get-started/locally/
- currently, rdkit needs with numpy<2.0.0, please install rdkit with numpy<2.0.0.

### Option 1: Installing from PyPi (Recommended)

```bash
pip install unimol_tools
```

We recommend installing ```huggingface_hub``` so that the required unimol models can be automatically downloaded at runtime! It can be install by

```bash
pip install huggingface_hub
```

`huggingface_hub` allows you to easily download and manage models from the Hugging Face Hub, which is key for using UniMol models.

### Option 2: Installing from source

```python
## Dependencies installation
pip install -r requirements.txt

## Clone repository
git clone https://github.com/deepmodeling/Uni-Mol.git
cd Uni-Mol/unimol_tools

## Install
python setup.py install
```

### Models in Huggingface

The UniMol pretrained models can be found at [dptech/Uni-Mol-Models](https://huggingface.co/dptech/Uni-Mol-Models/tree/main).

If the download is slow, you can use other mirrors, such as:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

Setting the `HF_ENDPOINT` environment variable specifies the mirror address for the Hugging Face Hub to use when downloading models.

### Modify the default directory for weights

Setting the `UNIMOL_WEIGHT_DIR` environment variable specifies the directory for pre-trained weights if the weights have been downloaded from another source.

```bash
export UNIMOL_WEIGHT_DIR=/path/to/your/weights/dir/
```

## News
- 2024-07-23: User experience improvements: Add `UNIMOL_WEIGHT_DIR`.
- 2024-06-25: unimol_tools has been publish to pypi! Huggingface has been used to manage the pretrain models.
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
@article{gao2023uni,
  title={Uni-qsar: an auto-ml tool for molecular property prediction},
  author={Gao, Zhifeng and Ji, Xiaohong and Zhao, Guojiang and Wang, Hongshuai and Zheng, Hang and Ke, Guolin and Zhang, Linfeng},
  journal={arXiv preprint arXiv:2304.12239},
  year={2023}
}
```

License
-------

This project is licensed under the terms of the MIT license. See LICENSE for additional details.
