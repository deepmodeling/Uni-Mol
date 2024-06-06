Official Repository for the Uni-Mol Series Methods
==================================================

Shortcuts
---------

- [Uni-Mol](./unimol/)
- [Uni-Mol+](./unimol_plus/)
- [Uni-Mol Tools](./unimol_tools/)
- [Uni-Mol Docking V2](./unimol_docking_v2/)


**Note**: if you want to install or run our codes, please `cd` to subfolders first.


Uni-Mol: A Universal 3D Molecular Representation Learning Framework
-------------------------------------------------------------------

[[Paper](https://openreview.net/forum?id=6K2RM6wVqKu)], [[Uni-Mol Docking Colab](https://colab.research.google.com/github/dptech-corp/Uni-Mol/blob/main/unimol/notebooks/unimol_binding_pose_demo.ipynb)]

Authors: Gengmo Zhou, Zhifeng Gao, Qiankun Ding, Hang Zheng, Hongteng Xu, Zhewei Wei, Linfeng Zhang, Guolin Ke 

<p align="center"><img src="unimol/figure/overview.png" width=60%></p>
<p align="center"><b>Schematic illustration of the Uni-Mol framework</b></p>

Uni-Mol is a universal 3D molecular pretraining framework that offers a significant expansion of representation capacity and application scope in drug design. The framework comprises two models: a molecular pretraining model that has been trained using 209M molecular 3D conformations, 
and a pocket pretraining model that has been trained using 3M candidate protein pocket data. These two models can be used independently for different tasks and are combined for protein-ligand binding tasks. Uni-Mol has demonstrated superior performance compared to the state-of-the-art (SOTA) in 14 out of 15 molecular property prediction tasks. Moreover, Uni-Mol has achieved exceptional accuracy in 3D spatial tasks, such as protein-ligand binding pose prediction and molecular conformation generation.

Check this [subfolder](./unimol/) for more detalis.


Highly Accurate Quantum Chemical Property Prediction with Uni-Mol+
-------------------------------------------------------------------
[![arXiv](https://img.shields.io/badge/arXiv-2303.16982-00ff00.svg)](https://arxiv.org/abs/2303.16982) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/highly-accurate-quantum-chemical-property/graph-regression-on-pcqm4mv2-lsc)](https://paperswithcode.com/sota/graph-regression-on-pcqm4mv2-lsc?p=highly-accurate-quantum-chemical-property) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/highly-accurate-quantum-chemical-property/initial-structure-to-relaxed-energy-is2re)](https://paperswithcode.com/sota/initial-structure-to-relaxed-energy-is2re?p=highly-accurate-quantum-chemical-property)

<p align="center"><img src="unimol_plus/figure/overview.png" width=80%></p>
<p align="center"><b>Schematic illustration of the Uni-Mol+ framework</b></p>

Uni-Mol+ is a model for quantum chemical property prediction. Firstly, given a 2D molecular graph, Uni-Mol+ generates an initial 3D conformation from inexpensive methods such as RDKit. Then, the initial conformation is iteratively optimized to its equilibrium conformation, and the optimized conformation is further used to predict the QC properties. In the PCQM4MV2 and OC20 bencmarks, Uni-Mol+ outperforms previous SOTA methods by a large margin.

Check this [subfolder](./unimol_plus/) for more detalis.

Uni-Mol tools for property prediction, representation and downstreams
--------------------------------------------------------------------
Uni-Mol tools is a easy-use wrappers for property prediction,representation and downstreams with Uni-Mol. It includes the following tools:
* molecular property prediction with Uni-Mol.
* molecular representation with Uni-Mol.
* other downstreams with Uni-Mol.

Check this [subfolder](./unimol_tools/) for more detalis.

Documentation of Uni-Mol tools is available at https://unimol.readthedocs.io/en/latest/

Uni-Mol Docking V2: Towards realistic and accurate binding pose prediction
--------------------------------------------------------------------
[![arXiv](https://img.shields.io/badge/arXiv-2405.11769-00ff00.svg)](https://arxiv.org/abs/2405.11769) ![Static Badge](https://img.shields.io/badge/Bohrium_Apps-Uni--Mol_Docking_V2-blue?link=https%3A%2F%2Fbohrium.dp.tech%2Fapps%2Funimoldockingv2)


<p align="center"><img src="unimol_docking_v2/figure/bohrium_app.png" width=60%></p>
<p align="center"><b>Uni-Mol Docking V2 Bohrium App Interface</b></p>

We update Uni-Mol Docking to Uni-Mol Docking V2, which demonstrates a remarkable improvement in performance, accurately predicting the binding poses of 77+% of ligands in the PoseBusters benchmark with an RMSD value of less than 2.0 Å, and 75+\% passing all quality checks. This represents a significant increase from the 62% achieved by the previous Uni-Mol Docking model. Notably, our Uni-Mol Docking approach generates chemically accurate predictions, circumventing issues such as chirality inversions and steric
clashes that have plagued previous ML models.

Check this [subfolder](./unimol_docking_v2/) for more detalis.

Service of Uni-Mol Docking V2 is avaiable at https://bohrium.dp.tech/apps/unimoldockingv2

News
----
**May 20 2024**: We release Uni-Mol Docking V2, including data, model and weight.

**Jul 7 2023**: We update a new version of Uni-Mol+, including the model setting for OC20 and a better performance on PCQM4MV2. 

**Jun 9 2023**: We release Uni-Mol tools for property prediction, representation and downstreams.

**Mar 16 2023**: We release Uni-Mol+, a model for quantum chemical property prediction.

**Jan 21 2023**: Uni-Mol is accepted by ICLR 2023.

**Oct 12 2022**: Provide a demo to get Uni-Mol molecular representation.

**Sep 20 2022**: Provide Uni-Mol based IFD scoring function baseline for [AIAC 2022 Competition Prediction of protein binding ability of drug molecules](http://www.aiinnovation.com.cn/#/aiaeDetail?id=560). 

**Sep 9 2022**: Provide Uni-Mol binding pose prediction (docking) demo on Colab.

**Sep 8 2022**: 

- The code and data for protein-ligand binding pose prediction are released. 
- Finetuned model weights of molecular conformation generation and protein-ligand binding pose prediction are released. 
- [Paper update](https://chemrxiv.org/engage/chemrxiv/article-details/6318b529bada388485bc8361).

**Aug 17 2022**: Pretrained models are released.

**Jul 10 2022**: Pretraining codes are released.

**Jun 10 2022**: The 3D conformation data used in Uni-Mol is released.


Contact Us
--------
**1. Github issue** (recommended)   
you can open an issue on GitHub.

**2. WeChat**   
We welcome everyone to join our Uni-Mol user **WeChat group**. Scan the QR code below to join.


<img src = "https://bohrium.oss-cn-zhangjiakou.aliyuncs.com/article/16664/34ccee67813f42468568978e61f2f639/3b6b5467-1d17-4c37-a234-150129ac8e22.png" width = 15%>


**3. E-mail**   

If you wish to have in-depth contact with the Uni-Mol development team, please send an email to unimol@dp.tech.



Citation
--------

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

This project is licensed under the terms of the MIT license. See [LICENSE](https://github.com/dptech-corp/Uni-Mol/blob/main/LICENSE) for additional details.
