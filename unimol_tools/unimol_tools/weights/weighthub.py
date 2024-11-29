import os

from ..utils import logger

try:
    from huggingface_hub import snapshot_download
except:
    huggingface_hub_installed = False
    def snapshot_download(*args, **kwargs):
        raise ImportError('huggingface_hub is not installed. If weights are not avaliable, please install it by running: pip install huggingface_hub. Otherwise, please download the weights manually from https://huggingface.co/dptech/Uni-Mol-Models')

WEIGHT_DIR = os.environ.get('UNIMOL_WEIGHT_DIR', os.path.dirname(os.path.abspath(__file__)))

if 'UNIMOL_WEIGHT_DIR' in os.environ:
    logger.warning(f'Using custom weight directory from UNIMOL_WEIGHT_DIR: {WEIGHT_DIR}')
else:
    logger.info(f'Weights will be downloaded to default directory: {WEIGHT_DIR}')

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com" # use mirror to download weights

def weight_download(pretrain, save_path, local_dir_use_symlinks=True):
    """
    Downloads the specified pretrained model weights.

    :param pretrain: (str), The name of the pretrained model to download.
    :param save_path: (str), The directory where the weights should be saved.
    :param local_dir_use_symlinks: (bool, optional), Whether to use symlinks for the local directory. Defaults to True.
    """
    if os.path.exists(os.path.join(save_path, pretrain)):
        logger.info(f'{pretrain} exists in {save_path}')
        return
    
    logger.info(f'Downloading {pretrain}')
    snapshot_download(
        repo_id="dptech/Uni-Mol-Models",
        local_dir=save_path,
        allow_patterns=pretrain,
        local_dir_use_symlinks=local_dir_use_symlinks,
        #max_workers=8
    )

def weight_download_v2(pretrain, save_path, local_dir_use_symlinks=True):
    """
    Downloads the specified pretrained model weights.

    :param pretrain: (str), The name of the pretrained model to download.
    :param save_path: (str), The directory where the weights should be saved.
    :param local_dir_use_symlinks: (bool, optional), Whether to use symlinks for the local directory. Defaults to True.
    """
    if os.path.exists(os.path.join(save_path, pretrain)):
        logger.info(f'{pretrain} exists in {save_path}')
        return
    
    logger.info(f'Downloading {pretrain}')
    snapshot_download(
        repo_id="dptech/Uni-Mol2",
        local_dir=save_path,
        allow_patterns=pretrain,
        local_dir_use_symlinks=local_dir_use_symlinks,
        #max_workers=8
    )

# Download all the weights when this script is run
def download_all_weights(local_dir_use_symlinks=False):
    """
    Downloads all available pretrained model weights to the WEIGHT_DIR.
    
    :param local_dir_use_symlinks: (bool, optional), Whether to use symlinks for the local directory. Defaults to False.
    """
    logger.info(f'Downloading all weights to {WEIGHT_DIR}')
    snapshot_download(
        repo_id="dptech/Uni-Mol-Models",
        local_dir=WEIGHT_DIR,
        allow_patterns='*',
        local_dir_use_symlinks=local_dir_use_symlinks,
        #max_workers=8
    )

if '__main__' == __name__:
    download_all_weights()