# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from hashlib import md5

def pad_1d_tokens(
    values,
    pad_idx,
    left_pad=False,
    pad_to_length=None,
    pad_to_multiple=1,
    ):
    """
    padding one dimension tokens inputs.

    Args:
        values (list): A list of 1d tensors.
        pad_idx (int): The padding index.
        left_pad (bool, optional): Whether to left pad the tensors. Defaults to False.
        pad_to_length (int, optional): The desired length of the padded tensors. Defaults to None.
        pad_to_multiple (int, optional): The multiple to pad the tensors to. Defaults to 1.

    Returns:
        torch.Tensor: A padded 1d tensor.

    """
    size = max(v.size(0) for v in values)
    size = size if pad_to_length is None else max(size, pad_to_length)
    if pad_to_multiple != 1 and size % pad_to_multiple != 0:
        size = int(((size - 0.1) // pad_to_multiple + 1) * pad_to_multiple)
    res = values[0].new(len(values), size).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v) :] if left_pad else res[i][: len(v)])
    return res


def pad_2d(
    values,
    pad_idx,
    left_pad=False,
    pad_to_length=None,
    pad_to_multiple=1,
    ):
    """
    padding two dimension tensor inputs.

    Args:
        values (list): A list of 2d tensors.
        pad_idx (int): The padding index.
        left_pad (bool, optional): Whether to pad on the left side. Defaults to False.
        pad_to_length (int, optional): The length to pad the tensors to. If None, the maximum length in the list is used. Defaults to None.
        pad_to_multiple (int, optional): The multiple to pad the tensors to. Defaults to 1.

    Returns:
        torch.Tensor: A padded 2d tensor.
    """
    size = max(v.size(0) for v in values)
    size = size if pad_to_length is None else max(size, pad_to_length)
    if pad_to_multiple != 1 and size % pad_to_multiple != 0:
        size = int(((size - 0.1) // pad_to_multiple + 1) * pad_to_multiple)
    res = values[0].new(len(values), size, size).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v) :, size - len(v) :] if left_pad else res[i][: len(v), : len(v)])
    return res


def pad_coords(
    values,
    pad_idx,
    left_pad=False,
    pad_to_length=None,
    pad_to_multiple=1,
    ):
    """
    padding two dimension tensor coords which the third dimension is 3.

    Args:
        values (list): A list of 1d tensors.
        pad_idx (int): The value used for padding.
        left_pad (bool, optional): Whether to pad on the left side. Defaults to False.
        pad_to_length (int, optional): The desired length of the padded tensor. Defaults to None.
        pad_to_multiple (int, optional): The multiple to pad the tensor to. Defaults to 1.

    Returns:
        torch.Tensor: A padded 2d coordinate tensor.
    """
    size = max(v.size(0) for v in values)
    size = size if pad_to_length is None else max(size, pad_to_length)
    if pad_to_multiple != 1 and size % pad_to_multiple != 0:
        size = int(((size - 0.1) // pad_to_multiple + 1) * pad_to_multiple)
    res = values[0].new(len(values), size, 3).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v) :, :] if left_pad else res[i][: len(v),:])
    return res