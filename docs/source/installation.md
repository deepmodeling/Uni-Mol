# Installation

## Overview

[Uni-Mol](https://github.com/dptech-corp/Uni-Mol/tree/main)
 now can be installed from source. 
```shell
## unicore and other dependencies installation
pip install -r requirements.txt
## clone repo
git clone https://github.com/dptech-corp/Uni-Mol.git
cd Uni-Mol/unimol_tools/unimol_tools

## download pretrained weights
wget https://github.com/dptech-corp/Uni-Mol/releases/download/v0.1/mol_pre_all_h_220816.pt
wget https://github.com/dptech-corp/Uni-Mol/releases/download/v0.1/mol_pre_no_h_220816.pt
wget https://github.com/dptech-corp/Uni-Mol/releases/download/v0.1/pocket_pre_220816.pt
wget https://github.com/dptech-corp/Uni-Mol/releases/download/v0.1/mof_pre_no_h_CORE_MAP_20230505.pt
wget https://github.com/dptech-corp/Uni-Mol/releases/download/v0.1/mp_all_h_230313.pt
wget https://github.com/dptech-corp/Uni-Mol/releases/download/v0.1/oled_pre_no_h_230101.pt

mkdir -p weights
mv *.pt weights/

## install
cd ..
python setup.py install
```

## Bohrium notebook

Uni-Mol images can be avaliable on the online notebook platform [Bohirum notebook](https://nb.bohrium.dp.tech/).
 