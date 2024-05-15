# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from functools import lru_cache
from unicore.data import BaseWrapperDataset


class PrependAndAppend2DDataset(BaseWrapperDataset):
    def __init__(self, dataset, token=None):
        super().__init__(dataset)
        self.token = token

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        item = self.dataset[idx]
        if self.token is not None:
            h, w = item.size(-2), item.size(-1)
            new_item = torch.full((h + 2, w + 2), self.token).type_as(item)
            new_item[1:-1, 1:-1] = item
            return new_item
        return item
