Uni-Mol Docking V2
===================================================================


Dependencies
------------
 - [Uni-Core](https://github.com/dptech-corp/Uni-Core), check its [Installation Documentation](https://github.com/dptech-corp/Uni-Core#installation).
 - rdkit==2022.9.3, install via `pip install rdkit-pypi==2022.9.3 -i https://pypi.tuna.tsinghua.edu.cn/simple/ --trusted-host pypi.tuna.tsinghua.edu.cn`
  - biopandas==0.4.1, install via `pip install biopandas`


Model weights
----------------------------------

| Model                     | File Size  |Update Date | Download Link                                                | 
|--------------------------|------------| ------------|--------------------------------------------------------------|
| unimol docking v2       | 464MB   | May 17 2024 |https://www.dropbox.com/scl/fi/sfhrtx1tjprce18wbvmdr/unimol_docking_v2_240517.pt?rlkey=5zg7bh150kcinalrqdhzmyyoo&st=n6j0nt6c&dl=0                |


Results
----------------------------------
|< 2.0 Å RMSD(% )      | PoseBusters (N=428) | Astex (N=85) | 
|--------|----|----|   
| DeepDock | 17.8 | 34.12 |
| DiffDock        |  37.9 |71.76  |
| UMol |   45| - | 
| Vina      |  52.3  | 57.65 | 
| Uni-Mol Docking     |  58.9 | 82.35 | 
| AlphaFold latest     |  73.6 | - |
| **Uni-Mol Docking V2**   |  **77.6** | **95.29**|
