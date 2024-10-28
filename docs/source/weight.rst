.. _weights:

Weights
=======

We recommend installing ``huggingface_hub`` so that the required Uni-Mol models can be automatically downloaded at runtime! It can be installed by:

.. code-block:: bash

    pip install huggingface_hub

``huggingface_hub`` allows you to easily download and manage models from the Hugging Face Hub, which is key for using Uni-Mol models.

Models in Huggingface
---------------------

The Uni-Mol pretrained models can be found at `dptech/Uni-Mol-Models <https://huggingface.co/dptech/Uni-Mol-Models/tree/main>`_.

If the download is slow, you can use other mirrors, such as:

.. code-block:: bash

    export HF_ENDPOINT=https://hf-mirror.com

Setting the ``HF_ENDPOINT`` environment variable specifies the mirror address for the Hugging Face Hub to use when downloading models.

`unimol_tools.weights.weight_hub.py <https://github.com/deepmodeling/Uni-Mol/blob/main/unimol_tools/unimol_tools/weights/weighthub.py>`_ control the logger.

.. automodule:: unimol_tools.weights.weighthub
   :members: