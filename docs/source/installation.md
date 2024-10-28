# Installation

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

`huggingface_hub` allows you to easily download and manage models from the Hugging Face Hub, which is key for using Uni-Mol models.

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

The Uni-Mol pretrained models can be found at [dptech/Uni-Mol-Models](https://huggingface.co/dptech/Uni-Mol-Models/tree/main).

If the download is slow, you can use other mirrors, such as:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

Setting the `HF_ENDPOINT` environment variable specifies the mirror address for the Hugging Face Hub to use when downloading models.

## Bohrium notebook

Uni-Mol images can be avaliable on the online notebook platform [Bohirum notebook](https://nb.bohrium.dp.tech/).
 