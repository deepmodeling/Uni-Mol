.. UniMol documentation master file, created by
   sphinx-quickstart on Wed Nov 29 03:53:18 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Uni-Mol' documentation!
==========================================

Uni-Mol is the first universal large-scale three-dimensional Molecular Representation Learning (MRL) framework developed by the DP Technology. It expands the application scope and representation capabilities of MRL. 

This framework consists of two models, one trained on billions of molecular three-dimensional conformations and the other on millions of protein pocket data. 

It has shown excellent performance in various molecular property prediction tasks, especially in 3D-related tasks, where it demonstrates significant performance. In addition to drug design, Uni-Mol can also predict the properties of materials, such as the gas adsorption performance of MOF materials and the optical properties of OLED molecules. 

.. Important::

   The project Uni-Mol is licensed under `MIT LICENSE <https://github.com/deepmodeling/Uni-Mol/blob/main/LICENSE>`_.
   If you use Uni-Mol in your research, please kindly cite the following works:

   - Gengmo Zhou, Zhifeng Gao, Qiankun Ding, Hang Zheng, Hongteng Xu, Zhewei Wei, Linfeng Zhang, Guolin Ke. "Uni-Mol: A Universal 3D Molecular Representation Learning Framework." The Eleventh International Conference on Learning Representations, 2023. `https://openreview.net/forum?id=6K2RM6wVqKu <https://openreview.net/forum?id=6K2RM6wVqKu>`_.
   - Shuqi Lu, Zhifeng Gao, Di He, Linfeng Zhang, Guolin Ke. "Data-driven quantum chemical property prediction leveraging 3D conformations with Uni-Mol+." Nature Communications, 2024. `https://www.nature.com/articles/s41467-024-51321-w <https://www.nature.com/articles/s41467-024-51321-w>`_.


Uni-Mol tools is a easy-use wrappers for property prediction,representation and downstreams with Uni-Mol. It includes the following tools:

* molecular property prediction with Uni-Mol.
* molecular representation with Uni-Mol.
* other downstreams with Uni-Mol.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started:

   requirements
   installation

.. toctree::
   :maxdepth: 2
   :caption: Tutorials:

   quickstart
   school
   examples

.. toctree::
   :maxdepth: 2
   :caption: Uni-Mol tools:

   train
   data
   models
   task
   utils
   weight
   features


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
