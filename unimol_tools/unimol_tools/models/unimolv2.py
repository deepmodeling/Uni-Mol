# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from .transformersv2 import TransformerEncoderWithPairV2
from ..utils import pad_1d_tokens, pad_2d, pad_coords
import argparse
import pathlib
import os

from .transformersv2 import AtomFeature, EdgeFeature, SE3InvariantKernel, MovementPredictionHead
from ..utils import logger
from ..config import MODEL_CONFIG_V2
from ..data import Dictionary
from ..weights import weight_download_v2, WEIGHT_DIR

BACKBONE = {
    'transformer': TransformerEncoderWithPairV2,
}

class UniMolV2Model(nn.Module):
    """
    UniMolModel is a specialized model for molecular, protein, crystal, or MOF (Metal-Organic Frameworks) data. 
    It dynamically configures its architecture based on the type of data it is intended to work with. The model
    supports multiple data types and incorporates various architecture configurations and pretrained weights.

    Attributes:
        - output_dim: The dimension of the output layer.
        - data_type: The type of data the model is designed to handle.
        - remove_hs: Flag to indicate whether hydrogen atoms are removed in molecular data.
        - pretrain_path: Path to the pretrained model weights.
        - dictionary: The dictionary object used for tokenization and encoding.
        - mask_idx: Index of the mask token in the dictionary.
        - padding_idx: Index of the padding token in the dictionary.
        - embed_tokens: Embedding layer for token embeddings.
        - encoder: Transformer encoder backbone of the model.
        - gbf_proj, gbf: Layers for Gaussian basis functions or numerical embeddings.
        - classification_head: The final classification head of the model.
    """
    def __init__(self, output_dim=2, model_size='84m', **params):
        """
        Initializes the UniMolModel with specified parameters and data type.

        :param output_dim: (int) The number of output dimensions (classes).
        :param data_type: (str) The type of data (e.g., 'molecule', 'protein').
        :param params: Additional parameters for model configuration.
        """
        super().__init__()

        self.args = molecule_architecture(model_size=model_size)
        self.output_dim = output_dim
        self.model_size = model_size
        self.remove_hs = params.get('remove_hs', False)

        name = model_size
        if not os.path.exists(os.path.join(WEIGHT_DIR, MODEL_CONFIG_V2['weight'][name])):
            weight_download_v2(MODEL_CONFIG_V2['weight'][name], WEIGHT_DIR)

        self.pretrain_path = os.path.join(WEIGHT_DIR, MODEL_CONFIG_V2['weight'][name])

        self.token_num = 128
        self.padding_idx = 0
        self.mask_idx = 127
        self.embed_tokens = nn.Embedding(
            self.token_num, self.args.encoder_embed_dim, self.padding_idx
        )

        self.encoder = BACKBONE[self.args.backbone](
            num_encoder_layers = self.args.num_encoder_layers,
            embedding_dim = self.args.encoder_embed_dim,

            pair_dim = self.args.pair_embed_dim,
            pair_hidden_dim = self.args.pair_hidden_dim,

            ffn_embedding_dim = self.args.ffn_embedding_dim,
            num_attention_heads = self.args.num_attention_heads,
            dropout = self.args.dropout,
            attention_dropout = self.args.attention_dropout,
            activation_dropout = self.args.activation_dropout,
            activation_fn = self.args.activation_fn,
            droppath_prob = self.args.droppath_prob,
            pair_dropout = self.args.pair_dropout,
        )

        num_atom = 512
        num_degree = 128
        num_edge = 64
        num_pair = 512
        num_spatial = 512

        K = 128
        n_edge_type = 1
        
        self.atom_feature = AtomFeature(
            num_atom=num_atom,
            num_degree=num_degree,
            hidden_dim=self.args.encoder_embed_dim,
        )

        self.edge_feature = EdgeFeature(
            pair_dim=self.args.pair_embed_dim,
            num_edge=num_edge,
            num_spatial=num_spatial,
        )


        self.se3_invariant_kernel = SE3InvariantKernel(
            pair_dim=self.args.pair_embed_dim,
            num_pair=num_pair,
            num_kernel=K,
            std_width=self.args.gaussian_std_width,
            start=self.args.gaussian_mean_start,
            stop=self.args.gaussian_mean_stop,
        )

        self.movement_pred_head = MovementPredictionHead(
            self.args.encoder_embed_dim, self.args.pair_embed_dim, self.args.encoder_attention_heads
        )

        self.classification_heads = nn.ModuleDict()
        self.dtype = torch.float32

        self.classification_head = ClassificationHead(
            input_dim=self.args.encoder_embed_dim,
            inner_dim=self.args.encoder_embed_dim,
            num_classes=self.output_dim,
            activation_fn=self.args.pooler_activation_fn,
            pooler_dropout=self.args.pooler_dropout,
        )
        self.load_pretrained_weights(path=self.pretrain_path)

    def load_pretrained_weights(self, path):
        """
        Loads pretrained weights into the model.

        :param path: (str) Path to the pretrained weight file.
        """
        if path is not None:
            logger.info("Loading pretrained weights from {}".format(path))
            state_dict = torch.load(path, map_location=lambda storage, loc: storage)
            self.load_state_dict(state_dict['model'], strict=False)

    @classmethod
    def build_model(cls, args):
        """
        Class method to build a new instance of the UniMolModel.

        :param args: Arguments for model configuration.
        :return: An instance of UniMolModel.
        """
        return cls(args)
#'atom_feat', 'atom_mask', 'edge_feat', 'shortest_path', 'degree', 'pair_type', 'attn_bias', 'src_tokens'
    def forward(
        self,
        atom_feat,
        atom_mask,
        edge_feat,
        shortest_path,
        degree,
        pair_type,
        attn_bias,
        src_tokens,
        src_pos,
        return_repr=False,
        return_atomic_reprs=False,
        **kwargs
    ):
        

        pos = src_pos

        n_mol, n_atom = atom_feat.shape[:2]
        token_feat = self.embed_tokens(src_tokens)
        x = self.atom_feature({'atom_feat': atom_feat, 'degree': degree}, token_feat)

        dtype = self.dtype

        x = x.type(dtype)

        attn_mask = attn_bias.clone()
        attn_bias = torch.zeros_like(attn_mask)
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.args.encoder_attention_heads, 1, 1)
        attn_bias = attn_bias.unsqueeze(-1).repeat(1, 1, 1, self.args.pair_embed_dim)
        attn_bias = self.edge_feature({'shortest_path':shortest_path, 'edge_feat': edge_feat}, attn_bias)
        attn_mask = attn_mask.type(self.dtype)

        atom_mask_cls = torch.cat(
            [
                torch.ones(n_mol, 1, device=atom_mask.device, dtype=atom_mask.dtype),
                atom_mask,
            ],
            dim=1,
        ).type(self.dtype)

        pair_mask = atom_mask_cls.unsqueeze(-1) * atom_mask_cls.unsqueeze(-2)

        def one_block(x, pos, return_x=False):
            delta_pos = pos.unsqueeze(1) - pos.unsqueeze(2)
            dist = delta_pos.norm(dim=-1)
            attn_bias_3d = self.se3_invariant_kernel(dist.detach(), pair_type)
            new_attn_bias = attn_bias.clone()
            new_attn_bias[:, 1:, 1:, :] = new_attn_bias[:, 1:, 1:, :] + attn_bias_3d
            new_attn_bias = new_attn_bias.type(dtype)
            x, pair = self.encoder(
                x,
                new_attn_bias,
                atom_mask=atom_mask_cls,
                pair_mask=pair_mask,
                attn_mask=attn_mask,
            )
            node_output = self.movement_pred_head(
                x[:, 1:, :],
                pair[:, 1:, 1:, :],
                attn_mask[:, :, 1:, 1:],
                delta_pos.detach(),
            )
            if return_x:
                return x, pair, pos + node_output
            else:
                return pos + node_output

        x, pair, pos = one_block(x, pos, return_x=True)
        cls_repr = x[:, 0, :]  # CLS token repr
        all_repr = x[:, :, :]  # all token repr
        
        if return_repr:
            filtered_tensors = []
            filtered_coords = []

            for tokens, coord in zip(src_tokens, src_pos):
                filtered_tensor = tokens[(tokens != 0) & (tokens != 1) & (tokens != 2)] # filter out BOS(0), EOS(1), PAD(2)
                filtered_coord = coord[(tokens != 0) & (tokens != 1) & (tokens != 2)]
                filtered_tensors.append(filtered_tensor)
                filtered_coords.append(filtered_coord)

            lengths = [len(filtered_tensor) for filtered_tensor in filtered_tensors] # Compute the lengths of the filtered tensors
            if return_atomic_reprs:
                cls_atomic_reprs = [] 
                atomic_symbols = []
                for i in range(len(all_repr)):
                    atomic_reprs = x[i, 1:lengths[i]+1, :]
                    atomic_symbol = filtered_tensors[i]
                    atomic_symbols.append(atomic_symbol)
                    cls_atomic_reprs.append(atomic_reprs)
                return {
                    'cls_repr': cls_repr, 
                    'atomic_symbol': atomic_symbols, 
                    'atomic_coords': filtered_coords, 
                    'atomic_reprs': cls_atomic_reprs
                    }        
            else:
                return {'cls_repr': cls_repr}  
        
        logits = self.classification_head(cls_repr)
        return logits


    def register_classification_head(
        self, name, num_classes=None, inner_dim=None, **kwargs
    ):
        """Register a classification head."""
        if name in self.classification_heads:
            prev_num_classes = self.classification_heads[name].out_proj.out_features
            prev_inner_dim = self.classification_heads[name].dense.out_features
            if num_classes != prev_num_classes or inner_dim != prev_inner_dim:
                logger.warning(
                    're-registering head "{}" with num_classes {} (prev: {}) '
                    "and inner_dim {} (prev: {})".format(
                        name, num_classes, prev_num_classes, inner_dim, prev_inner_dim
                    )
                )
        self.classification_heads[name] = ClassificationHead(
            input_dim=self.args.encoder_embed_dim,
            inner_dim=inner_dim or self.args.encoder_embed_dim,
            num_classes=num_classes,
            activation_fn=self.args.pooler_activation_fn,
            pooler_dropout=self.args.pooler_dropout,
        )

    def set_num_updates(self, num_updates):
        """State from trainer to pass along to model at every update."""
        self._num_updates = num_updates

    def get_num_updates(self):
        return self._num_updates

    def batch_collate_fn(self, samples):
        """
        Custom collate function for batch processing non-MOF data.

        :param samples: A list of sample data.

        :return: A tuple containing a batch dictionary and labels.
        """
        batch = {}
        for k in samples[0][0].keys():
            if k == 'atom_feat':
                v = pad_coords([s[0][k] for s in samples], pad_idx=self.padding_idx, dim=8)
            elif k == 'atom_mask':
                v = pad_1d_tokens([s[0][k] for s in samples], pad_idx=self.padding_idx)
            elif k == 'edge_feat':
                v = pad_2d([s[0][k] for s in samples], pad_idx=self.padding_idx, dim=3)
            elif k == 'shortest_path':
                v = pad_2d([s[0][k] for s in samples], pad_idx=self.padding_idx)
            elif k == 'degree':
                v = pad_1d_tokens([s[0][k] for s in samples], pad_idx=self.padding_idx)
            elif k == 'pair_type':
                v = pad_2d([s[0][k] for s in samples], pad_idx=self.padding_idx, dim=2)
            elif k == 'attn_bias':
                v = pad_2d([s[0][k] for s in samples], pad_idx=self.padding_idx)
            elif k == 'src_tokens':
                v = pad_1d_tokens([s[0][k] for s in samples], pad_idx=self.padding_idx)
            elif k == 'src_pos':
                v = pad_coords([s[0][k] for s in samples], pad_idx=self.padding_idx)
            batch[k] = v
        try:
            label = torch.tensor([s[1] for s in samples])
        except:
            label = None
        return batch, label

class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
        self,
        input_dim,
        inner_dim,
        num_classes,
        activation_fn,
        pooler_dropout,
    ):
        """
        Initialize the classification head.

        :param input_dim: Dimension of input features.
        :param inner_dim: Dimension of the inner layer.
        :param num_classes: Number of classes for classification.
        :param activation_fn: Activation function name.
        :param pooler_dropout: Dropout rate for the pooling layer.
        """
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.activation_fn = get_activation_fn(activation_fn)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, features, **kwargs):
        """
        Forward pass for the classification head.

        :param features: Input features for classification.

        :return: Output from the classification head.
        """
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class NonLinearHead(nn.Module):
    """
    A neural network module used for simple classification tasks. It consists of a two-layered linear network 
    with a nonlinear activation function in between.

    Attributes:
        - linear1: The first linear layer.
        - linear2: The second linear layer that outputs to the desired dimensions.
        - activation_fn: The nonlinear activation function.
    """
    def __init__(
        self,
        input_dim,
        out_dim,
        activation_fn,
        hidden=None,
    ):
        """
        Initializes the NonLinearHead module.

        :param input_dim: Dimension of the input features.
        :param out_dim: Dimension of the output.
        :param activation_fn: The activation function to use.
        :param hidden: Dimension of the hidden layer; defaults to the same as input_dim if not provided.
        """
        super().__init__()
        hidden = input_dim if not hidden else hidden
        self.linear1 = nn.Linear(input_dim, hidden)
        self.linear2 = nn.Linear(hidden, out_dim)
        self.activation_fn = get_activation_fn(activation_fn)

    def forward(self, x):
        """
        Forward pass of the NonLinearHead.

        :param x: Input tensor to the module.

        :return: Tensor after passing through the network.
        """
        x = self.linear1(x)
        x = self.activation_fn(x)
        x = self.linear2(x)
        return x

@torch.jit.script
def gaussian(x, mean, std):
    """
    Gaussian function implemented for PyTorch tensors.

    :param x: The input tensor.
    :param mean: The mean for the Gaussian function.
    :param std: The standard deviation for the Gaussian function.

    :return: The output tensor after applying the Gaussian function.
    """
    pi = 3.14159
    a = (2 * pi) ** 0.5
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)

def get_activation_fn(activation):
    """ Returns the activation function corresponding to `activation` """

    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    elif activation == "tanh":
        return torch.tanh
    elif activation == "linear":
        return lambda x: x
    else:
        raise RuntimeError("--activation-fn {} not supported".format(activation))

class GaussianLayer(nn.Module):
    """
    A neural network module implementing a Gaussian layer, useful in graph neural networks.

    Attributes:
        - K: Number of Gaussian kernels.
        - means, stds: Embeddings for the means and standard deviations of the Gaussian kernels.
        - mul, bias: Embeddings for scaling and bias parameters.
    """
    def __init__(self, K=128, edge_types=1024):
        """
        Initializes the GaussianLayer module.

        :param K: Number of Gaussian kernels.
        :param edge_types: Number of different edge types to consider.

        :return: An instance of the configured Gaussian kernel and edge types.
        """
        super().__init__()
        self.K = K
        self.means = nn.Embedding(1, K)
        self.stds = nn.Embedding(1, K)
        self.mul = nn.Embedding(edge_types, 1)
        self.bias = nn.Embedding(edge_types, 1)
        nn.init.uniform_(self.means.weight, 0, 3)
        nn.init.uniform_(self.stds.weight, 0, 3)
        nn.init.constant_(self.bias.weight, 0)
        nn.init.constant_(self.mul.weight, 1)

    def forward(self, x, edge_type):
        """
        Forward pass of the GaussianLayer.

        :param x: Input tensor representing distances or other features.
        :param edge_type: Tensor indicating types of edges in the graph.

        :return: Tensor transformed by the Gaussian layer.
        """
        mul = self.mul(edge_type).type_as(x)
        bias = self.bias(edge_type).type_as(x)
        x = mul * x.unsqueeze(-1) + bias
        x = x.expand(-1, -1, -1, self.K)
        mean = self.means.weight.float().view(-1)
        std = self.stds.weight.float().view(-1).abs() + 1e-5
        return gaussian(x.float(), mean, std).type_as(self.means.weight)
    
class NumericalEmbed(nn.Module):
    """
    Numerical embedding module, typically used for embedding edge features in graph neural networks.

    Attributes:
        - K: Output dimension for embeddings.
        - mul, bias, w_edge: Embeddings for transformation parameters.
        - proj: Projection layer to transform inputs.
        - ln: Layer normalization.
    """
    def __init__(self, K=128, edge_types=1024, activation_fn='gelu'):
        """
        Initializes the NonLinearHead.

        :param input_dim: The input dimension of the first layer.
        :param out_dim: The output dimension of the second layer.
        :param activation_fn: The activation function to use.
        :param hidden: The dimension of the hidden layer; defaults to input_dim if not specified.
        """
        super().__init__()
        self.K = K 
        self.mul = nn.Embedding(edge_types, 1)
        self.bias = nn.Embedding(edge_types, 1)
        self.w_edge = nn.Embedding(edge_types, K)

        self.proj = NonLinearHead(1, K, activation_fn, hidden=2*K)
        self.ln = nn.LayerNorm(K)

        nn.init.constant_(self.bias.weight, 0)
        nn.init.constant_(self.mul.weight, 1)
        nn.init.kaiming_normal_(self.w_edge.weight)


    def forward(self, x, edge_type):    # edge_type, atoms
        """
        Forward pass of the NonLinearHead.

        :param x: Input tensor to the classification head.

        :return: The output tensor after passing through the layers.
        """
        mul = self.mul(edge_type).type_as(x)
        bias = self.bias(edge_type).type_as(x)
        w_edge = self.w_edge(edge_type).type_as(x)
        edge_emb = w_edge * torch.sigmoid(mul * x.unsqueeze(-1) + bias)
        
        edge_proj = x.unsqueeze(-1).type_as(self.mul.weight)
        edge_proj = self.proj(edge_proj)
        edge_proj = self.ln(edge_proj)

        h = edge_proj + edge_emb
        h = h.type_as(self.mul.weight)
        return h

def molecule_architecture(model_size='84m'):
    args = argparse.ArgumentParser()
    if model_size == '84m':  
        args.num_encoder_layers = getattr(args, "num_encoder_layers", 12)
        args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 768)
        args.num_attention_heads = getattr(args, "num_attention_heads", 48)
        args.ffn_embedding_dim = getattr(args, "ffn_embedding_dim", 768)
        args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 48)
    elif model_size == '164m':
        args.num_encoder_layers = getattr(args, "num_encoder_layers", 24)
        args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 768)
        args.num_attention_heads = getattr(args, "num_attention_heads", 48)
        args.ffn_embedding_dim = getattr(args, "ffn_embedding_dim", 768)
        args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 48)
    elif model_size == '310m':
        args.num_encoder_layers = getattr(args, "num_encoder_layers", 32)
        args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
        args.num_attention_heads = getattr(args, "num_attention_heads", 64)
        args.ffn_embedding_dim = getattr(args, "ffn_embedding_dim", 1024)
        args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 64)
    elif model_size == '570m':
        args.num_encoder_layers = getattr(args, "num_encoder_layers", 32)
        args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1536)
        args.num_attention_heads = getattr(args, "num_attention_heads", 96)
        args.ffn_embedding_dim = getattr(args, "ffn_embedding_dim", 1536)
        args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 96)
    elif model_size == '1.1B':
        args.num_encoder_layers = getattr(args, "num_encoder_layers", 64)
        args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1536)
        args.num_attention_heads = getattr(args, "num_attention_heads", 96)
        args.ffn_embedding_dim = getattr(args, "ffn_embedding_dim", 1536)
        args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 96)
    else:
        raise ValueError('Current not support data type: {}'.format(model_size))
    args.pair_embed_dim = getattr(args, "pair_embed_dim", 512)
    args.pair_hidden_dim = getattr(args, "pair_hidden_dim", 64)
    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.droppath_prob = getattr(args, "droppath_prob", 0.0)
    args.pair_dropout = getattr(args, "pair_dropout", 0.25)
    args.backbone = getattr(args, "backbone", "transformer")
    args.gaussian_std_width = getattr(args, "gaussian_std_width", 1.0)
    args.gaussian_mean_start = getattr(args, "gaussian_mean_start", 0.0)
    args.gaussian_mean_stop = getattr(args, "gaussian_mean_stop", 9.0)
    args.pooler_dropout = getattr(args, "pooler_dropout", 0.0)
    args.pooler_activation_fn = getattr(args, "pooler_activation_fn", "tanh")
    return args

