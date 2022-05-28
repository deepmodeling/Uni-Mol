# Uni-Mol: A Universal 3D Molecular Representation Learning Framework 
[[ChemRxiv](https://chemrxiv.org/engage/chemrxiv/article-details/628e5b4d5d948517f5ce6d72)]

Authors: Gengmo Zhou, Zhifeng Gao, Qiankun Ding, Hang Zheng, Hongteng Xu, Zhewei Wei, Linfeng Zhang, Guolin Ke 

Uni-Mol is a universal 3D molecular pretraining framework that significantly enlarges the representation ability and application scope in drug design. 

![](https://cdn.nlark.com/yuque/0/2022/png/22931975/1653727970657-ad3e03aa-d789-4a86-9ed0-06e830f07015.png#clientId=u4d41bdf6-f72d-4&crop=0&crop=0&crop=1&crop=1&from=paste&id=u0b4eb0cc&margin=%5Bobject%20Object%5D&originHeight=1112&originWidth=2015&originalType=url&ratio=1&rotation=0&showTitle=false&status=done&style=none&taskId=uab32a03a-1cd6-4733-8907-69f02c50851&title=)

<p align="center">**Schematic illustration of the Uni-Mol framework**</p>

Uni-Mol is composed of two models: a molecular pretraining model trained by 209M molecular 3D conformations; a pocket pretraining model trained by 3M candidate protein pocket data. The two models are used independently for separate tasks, and are combined when used in protein-ligand binding tasks. 


## Data release 
We will later release the data we used in the pretraining and downstream tasks. 
## Code release 
We will release the source code in this repo. 