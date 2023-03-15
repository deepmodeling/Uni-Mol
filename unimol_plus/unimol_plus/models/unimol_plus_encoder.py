import imp
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from unicore.modules import LayerNorm

from .layers import (
    UnimolPlusEncoderLayer,
    Dropout,
)


class UnimolPLusEncoder(nn.Module):
    def __init__(
        self,
        num_encoder_layers: int = 6,
        embedding_dim: int = 768,
        pair_dim: int = 64,
        pair_hidden_dim: int = 32,
        ffn_embedding_dim: int = 3072,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        activation_fn: str = "gelu",
        droppath_prob: float = 0.0,
        pair_dropout: float = 0.25,
    ) -> None:

        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_head = num_attention_heads
        self.layer_norm = LayerNorm(embedding_dim)
        self.pair_layer_norm = LayerNorm(pair_dim)
        self.layers = nn.ModuleList([])

        if droppath_prob > 0:
            droppath_probs = [
                x.item() for x in torch.linspace(0, droppath_prob, num_encoder_layers)
            ]
        else:
            droppath_probs = None

        self.layers.extend(
            [
                UnimolPlusEncoderLayer(
                    embedding_dim=embedding_dim,
                    pair_dim=pair_dim,
                    pair_hidden_dim=pair_hidden_dim,
                    ffn_embedding_dim=ffn_embedding_dim,
                    num_attention_heads=num_attention_heads,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                    activation_dropout=activation_dropout,
                    activation_fn=activation_fn,
                    droppath_prob=droppath_probs[i]
                    if droppath_probs is not None
                    else 0,
                    pair_dropout=pair_dropout,
                )
                for i in range(num_encoder_layers)
            ]
        )

    def forward(
        self,
        x,
        pair,
        atom_mask,
        pair_mask,
        attn_mask=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        x = self.layer_norm(x)
        pair = self.pair_layer_norm(pair)
        op_mask = atom_mask.unsqueeze(-1)
        op_mask = op_mask * (op_mask.size(-2) ** -0.5)
        eps = 1e-3
        op_norm = 1.0 / (eps + torch.einsum("...bc,...dc->...bdc", op_mask, op_mask))
        for layer in self.layers:
            x, pair = layer(
                x,
                pair,
                pair_mask=pair_mask,
                self_attn_mask=attn_mask,
                op_mask=op_mask,
                op_norm=op_norm,
            )
        return x, pair
