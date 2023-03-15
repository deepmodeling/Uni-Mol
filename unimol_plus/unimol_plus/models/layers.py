import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from unicore import utils
from unicore.modules import softmax_dropout, LayerNorm

from unicore.utils import (
    permute_final_dims,
)

from torch import Tensor
from typing import Callable, Optional


class Dropout(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, x, inplace: bool = False):
        if self.p > 0 and self.training:
            return F.dropout(x, p=self.p, training=True, inplace=inplace)
        else:
            return x


class Linear(nn.Linear):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        bias: bool = True,
        init: str = "default",
    ):
        super(Linear, self).__init__(d_in, d_out, bias=bias)

        self.use_bias = bias

        if self.use_bias:
            with torch.no_grad():
                self.bias.fill_(0)

        if init == "default":
            self._trunc_normal_init(1.0)
        elif init == "relu":
            self._trunc_normal_init(2.0)
        elif init == "glorot":
            self._glorot_uniform_init()
        elif init == "gating":
            self._zero_init(self.use_bias)
        elif init == "normal":
            self._normal_init()
        elif init == "final":
            self._zero_init(False)
        else:
            raise ValueError("Invalid init method.")

    def _trunc_normal_init(self, scale=1.0):
        # Constant from scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
        TRUNCATED_NORMAL_STDDEV_FACTOR = 0.87962566103423978
        _, fan_in = self.weight.shape
        scale = scale / max(1, fan_in)
        std = (scale**0.5) / TRUNCATED_NORMAL_STDDEV_FACTOR
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std)

    def _glorot_uniform_init(self):
        nn.init.xavier_uniform_(self.weight, gain=1)

    def _zero_init(self, use_bias=True):
        with torch.no_grad():
            self.weight.fill_(0.0)
            if use_bias:
                with torch.no_grad():
                    self.bias.fill_(1.0)

    def _normal_init(self):
        torch.nn.init.kaiming_normal_(self.weight, nonlinearity="linear")


class Embedding(nn.Embedding):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int = None,
    ):
        super(Embedding, self).__init__(
            num_embeddings, embedding_dim, padding_idx=padding_idx
        )
        self._normal_init()

        if padding_idx is not None:
            self.weight.data[self.padding_idx].zero_()

    def _normal_init(self, std=0.02):
        nn.init.normal_(self.weight, mean=0.0, std=std)


class Transition(nn.Module):
    def __init__(self, d_in, n, dropout=0.0):

        super(Transition, self).__init__()

        self.d_in = d_in
        self.n = n

        self.linear_1 = Linear(self.d_in, self.n * self.d_in, init="relu")
        self.act = nn.GELU()
        self.linear_2 = Linear(self.n * self.d_in, d_in, init="final")
        self.dropout = dropout

    def _transition(self, x):
        x = self.linear_1(x)
        x = self.act(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.linear_2(x)
        return x

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        x = self._transition(x=x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        q_dim: int,
        k_dim: int,
        v_dim: int,
        pair_dim: int,
        head_dim: int,
        num_heads: int,
        gating: bool = False,
        dropout: float = 0.0,
    ):
        super(Attention, self).__init__()

        self.num_heads = num_heads
        total_dim = head_dim * self.num_heads
        self.gating = gating
        self.linear_q = Linear(q_dim, total_dim, bias=False, init="glorot")
        self.linear_k = Linear(k_dim, total_dim, bias=False, init="glorot")
        self.linear_v = Linear(v_dim, total_dim, bias=False, init="glorot")
        self.linear_o = Linear(total_dim, q_dim, init="final")
        self.linear_g = None
        if self.gating:
            self.linear_g = Linear(q_dim, total_dim, init="gating")
        # precompute the 1/sqrt(head_dim)
        self.norm = head_dim**-0.5
        self.dropout = dropout
        self.linear_bias = Linear(pair_dim, num_heads)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        pair: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        g = None
        if self.linear_g is not None:
            # gating, use raw query input
            g = self.linear_g(q)

        q = self.linear_q(q)
        q *= self.norm
        k = self.linear_k(k)
        v = self.linear_v(v)

        q = q.view(q.shape[:-1] + (self.num_heads, -1)).transpose(-2, -3).contiguous()
        k = k.view(k.shape[:-1] + (self.num_heads, -1)).transpose(-2, -3).contiguous()
        v = v.view(v.shape[:-1] + (self.num_heads, -1)).transpose(-2, -3)

        attn = torch.matmul(q, k.transpose(-1, -2))
        del q, k
        bias = self.linear_bias(pair).permute(0, 3, 1, 2).contiguous()
        attn = softmax_dropout(attn, self.dropout, self.training, mask=mask, bias=bias)
        o = torch.matmul(attn, v)
        del attn, v

        o = o.transpose(-2, -3).contiguous()
        o = o.view(*o.shape[:-2], -1)

        if g is not None:
            o = torch.sigmoid(g) * o

        # merge heads
        o = self.linear_o(o)
        return o


class OuterProduct(nn.Module):
    def __init__(self, d_atom, d_pair, d_hid=32):
        super(OuterProduct, self).__init__()

        self.d_atom = d_atom
        self.d_pair = d_pair
        self.d_hid = d_hid

        self.linear_in = nn.Linear(d_atom, d_hid * 2)
        self.linear_out = nn.Linear(d_hid**2, d_pair)
        self.act = nn.GELU()

    def _opm(self, a, b):
        bsz, n, d = a.shape
        # outer = torch.einsum("...bc,...de->...bdce", a, b)
        a = a.view(bsz, n, 1, d, 1)
        b = b.view(bsz, 1, n, 1, d)
        outer = a * b
        outer = outer.view(outer.shape[:-2] + (-1,))
        outer = self.linear_out(outer)
        return outer

    def forward(
        self,
        m: torch.Tensor,
        op_mask: Optional[torch.Tensor] = None,
        op_norm: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        ab = self.linear_in(m)
        ab = ab * op_mask
        a, b = ab.chunk(2, dim=-1)
        z = self._opm(a, b)
        z *= op_norm
        return z


class AtomFeature(nn.Module):
    """
    Compute atom features for each atom in the molecule.
    """

    def __init__(
        self,
        num_atom,
        num_degree,
        hidden_dim,
    ):
        super(AtomFeature, self).__init__()
        self.atom_encoder = Embedding(num_atom, hidden_dim, padding_idx=0)
        self.degree_encoder = Embedding(num_degree, hidden_dim, padding_idx=0)
        self.vnode_encoder = Embedding(1, hidden_dim)

    def forward(self, batched_data):
        x, degree = (
            batched_data["atom_feat"],
            batched_data["degree"],
        )
        n_graph, n_node = x.size()[:2]

        node_feature = self.atom_encoder(x).sum(dim=-2)  # [n_graph, n_node, n_hidden]
        dtype = node_feature.dtype
        degree_feature = self.degree_encoder(degree)
        node_feature = node_feature + degree_feature

        graph_token_feature = self.vnode_encoder.weight.unsqueeze(0).repeat(
            n_graph, 1, 1
        )

        graph_node_feature = torch.cat([graph_token_feature, node_feature], dim=1)
        return graph_node_feature.type(dtype)


class EdgeFeature(nn.Module):
    """
    Compute attention bias for each head.
    """

    def __init__(
        self,
        pair_dim,
        num_edge,
        num_spatial,
    ):
        super(EdgeFeature, self).__init__()
        self.pair_dim = pair_dim

        self.edge_encoder = Embedding(num_edge, pair_dim, padding_idx=0)
        self.shorest_path_encoder = Embedding(num_spatial, pair_dim, padding_idx=0)
        self.vnode_virtual_distance = Embedding(1, pair_dim)

    def forward(self, batched_data, graph_attn_bias):
        shortest_path = batched_data["shortest_path"]
        edge_input = batched_data["edge_feat"]

        graph_attn_bias[:, 1:, 1:, :] = self.shorest_path_encoder(shortest_path)

        # reset spatial pos here
        t = self.vnode_virtual_distance.weight.view(1, 1, self.pair_dim)
        graph_attn_bias[:, 1:, 0, :] = t
        graph_attn_bias[:, 0, :, :] = t

        edge_input = self.edge_encoder(edge_input).mean(-2)
        graph_attn_bias[:, 1:, 1:, :] = graph_attn_bias[:, 1:, 1:, :] + edge_input
        return graph_attn_bias


class SE3InvariantKernel(nn.Module):
    """
    Compute 3D attention bias according to the position information for each head.
    """

    def __init__(
        self,
        pair_dim,
        num_pair,
        num_kernel,
        std_width=1.0,
        start=0.0,
        stop=9.0,
    ):
        super(SE3InvariantKernel, self).__init__()
        self.num_kernel = num_kernel

        self.gaussian = GaussianKernel(
            self.num_kernel,
            num_pair,
            std_width=std_width,
            start=start,
            stop=stop,
        )
        self.out_proj = NonLinear(self.num_kernel, pair_dim)

    def forward(self, dist, node_type_edge):
        edge_feature = self.gaussian(
            dist,
            node_type_edge.long(),
        )
        edge_feature = self.out_proj(edge_feature)

        return edge_feature


@torch.jit.script
def gaussian(x, mean, std):
    pi = 3.14159
    a = (2 * pi) ** 0.5
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)


class GaussianKernel(nn.Module):
    def __init__(self, K=128, num_pair=512, std_width=1.0, start=0.0, stop=9.0):
        super().__init__()
        self.K = K
        std_width = std_width
        start = start
        stop = stop
        mean = torch.linspace(start, stop, K)
        self.std = (std_width * (mean[1] - mean[0])).item()
        self.register_buffer("mean", mean)
        self.mul = Embedding(num_pair, 1, padding_idx=0)
        self.bias = Embedding(num_pair, 1, padding_idx=0)
        nn.init.constant_(self.bias.weight, 0)
        nn.init.constant_(self.mul.weight, 1.0)

    def forward(self, x, atom_pair):
        mul = self.mul(atom_pair).abs().sum(dim=-2)
        bias = self.bias(atom_pair).sum(dim=-2)
        x = mul * x.unsqueeze(-1) + bias
        x = x.expand(-1, -1, -1, self.K)
        mean = self.mean.float().view(-1)
        return gaussian(x.float(), mean, self.std)


class NonLinear(nn.Module):
    def __init__(self, input, output_size, hidden=None):
        super(NonLinear, self).__init__()

        if hidden is None:
            hidden = input
        self.layer1 = Linear(input, hidden, init="relu")
        self.layer2 = Linear(hidden, output_size, init="final")

    def forward(self, x):
        x = self.layer1(x)
        x = F.gelu(x)
        x = self.layer2(x)
        return x

    def zero_init(self):
        nn.init.zeros_(self.layer2.weight)
        nn.init.zeros_(self.layer2.bias)


class EnergyHead(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
    ):
        super().__init__()
        self.layer_norm = LayerNorm(input_dim)
        self.linear_in = Linear(input_dim, input_dim, init="relu")

        self.linear_out = Linear(input_dim, output_dim, bias=True, init="final")

    def forward(self, x):
        x = x.type(self.linear_in.weight.dtype)
        x = F.gelu(self.layer_norm(self.linear_in(x)))
        x = self.linear_out(x)
        return x


class MovementPredictionHead(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        pair_dim: int,
        num_head: int,
    ):
        super().__init__()
        self.layer_norm = LayerNorm(embed_dim)
        self.embed_dim = embed_dim
        self.q_proj = Linear(embed_dim, embed_dim, bias=False, init="glorot")
        self.k_proj = Linear(embed_dim, embed_dim, bias=False, init="glorot")
        self.v_proj = Linear(embed_dim, embed_dim, bias=False, init="glorot")
        self.num_head = num_head
        self.scaling = (embed_dim // num_head) ** -0.5
        self.force_proj1 = Linear(embed_dim, 1, init="final")
        self.force_proj2 = Linear(embed_dim, 1, init="final")
        self.force_proj3 = Linear(embed_dim, 1, init="final")
        self.linear_bias = Linear(pair_dim, num_head)
        self.dropout = 0.1

    def zero_init(self):
        nn.init.zeros_(self.force_proj1.weight)
        nn.init.zeros_(self.force_proj1.bias)
        nn.init.zeros_(self.force_proj2.weight)
        nn.init.zeros_(self.force_proj2.bias)
        nn.init.zeros_(self.force_proj3.weight)
        nn.init.zeros_(self.force_proj3.bias)

    def forward(
        self,
        query: Tensor,
        pair: Tensor,
        attn_mask: Tensor,
        delta_pos: Tensor,
    ) -> Tensor:
        bsz, n_node, _ = query.size()
        query = self.layer_norm(query)
        q = (
            self.q_proj(query).view(bsz, n_node, self.num_head, -1).transpose(1, 2)
            * self.scaling
        )
        k = self.k_proj(query).view(bsz, n_node, self.num_head, -1).transpose(1, 2)
        v = self.v_proj(query).view(bsz, n_node, self.num_head, -1).transpose(1, 2)
        attn = q @ k.transpose(-1, -2)  # [bsz, head, n, n]
        bias = self.linear_bias(pair).permute(0, 3, 1, 2).contiguous()
        attn_probs = softmax_dropout(
            attn,
            self.dropout,
            self.training,
            mask=attn_mask.contiguous(),
            bias=bias.contiguous(),
        ).view(bsz, self.num_head, n_node, n_node)
        rot_attn_probs = attn_probs.unsqueeze(-1) * delta_pos.unsqueeze(1).type_as(
            attn_probs
        )  # [bsz, head, n, n, 3]
        rot_attn_probs = rot_attn_probs.permute(0, 1, 4, 2, 3)
        x = rot_attn_probs @ v.unsqueeze(2)  # [bsz, head , 3, n, d]
        x = x.permute(0, 3, 2, 1, 4).contiguous().view(bsz, n_node, 3, -1)
        f1 = self.force_proj1(x[:, :, 0, :]).view(bsz, n_node, 1)
        f2 = self.force_proj2(x[:, :, 1, :]).view(bsz, n_node, 1)
        f3 = self.force_proj3(x[:, :, 2, :]).view(bsz, n_node, 1)
        cur_force = torch.cat([f1, f2, f3], dim=-1).float()
        return cur_force


class DropPath(torch.nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (
            x.ndim - 1
        )  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output

    def extra_repr(self) -> str:
        return f"prob={self.drop_prob}"


class TriangleMultiplication(nn.Module):
    def __init__(self, d_pair, d_hid):
        super(TriangleMultiplication, self).__init__()

        self.linear_ab_p = Linear(d_pair, d_hid * 2)
        self.linear_ab_g = Linear(d_pair, d_hid * 2, init="gating")

        self.linear_g = Linear(d_pair, d_pair, init="gating")
        self.linear_z = Linear(d_hid, d_pair, init="final")

        self.layer_norm_out = LayerNorm(d_hid)

    def forward(
        self,
        z: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        mask = mask.unsqueeze(-1)
        mask = mask * (mask.shape[-2] ** -0.5)

        g = self.linear_g(z)
        if self.training:
            ab = self.linear_ab_p(z) * mask * torch.sigmoid(self.linear_ab_g(z))
        else:
            ab = self.linear_ab_p(z)
            ab *= mask
            ab *= torch.sigmoid(self.linear_ab_g(z))
        a, b = torch.chunk(ab, 2, dim=-1)
        del z, ab

        a1 = permute_final_dims(a, (2, 0, 1))
        b1 = b.transpose(-1, -3)
        x = torch.matmul(a1, b1)
        del a1, b1
        b2 = permute_final_dims(b, (2, 0, 1))
        a2 = a.transpose(-1, -3)
        x = x + torch.matmul(a2, b2)
        del a, b, a2, b2

        x = permute_final_dims(x, (1, 2, 0))

        x = self.layer_norm_out(x)
        x = self.linear_z(x)
        return g * x


class UnimolPlusEncoderLayer(nn.Module):
    """
    Implements a Transformer-M Encoder Layer.
    """

    def __init__(
        self,
        embedding_dim: int = 768,
        pair_dim: int = 64,
        pair_hidden_dim: int = 32,
        ffn_embedding_dim: int = 3072,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        activation_fn: str = "relu",
        droppath_prob: float = 0.0,
        pair_dropout: float = 0.25,
    ) -> None:
        super().__init__()

        # Initialize parameters
        self.embedding_dim = embedding_dim
        self.num_attention_heads = num_attention_heads
        self.attention_dropout = attention_dropout

        if droppath_prob > 0.0:
            self.dropout_module = DropPath(droppath_prob)
        else:
            self.dropout_module = Dropout(dropout)

        # Initialize blocks
        self.activation_fn = utils.get_activation_fn(activation_fn)
        head_dim = self.embedding_dim // self.num_attention_heads
        self.self_attn = Attention(
            self.embedding_dim,
            self.embedding_dim,
            self.embedding_dim,
            pair_dim=pair_dim,
            head_dim=head_dim,
            num_heads=self.num_attention_heads,
            gating=False,
            dropout=attention_dropout,
        )

        # layer norm associated with the self attention layer
        self.self_attn_layer_norm = LayerNorm(self.embedding_dim)

        self.ffn = Transition(
            self.embedding_dim,
            ffn_embedding_dim // self.embedding_dim,
            dropout=activation_dropout,
        )

        # layer norm associated with the position wise feed-forward NN
        self.final_layer_norm = LayerNorm(self.embedding_dim)

        self.opm = OuterProduct(self.embedding_dim, pair_dim, d_hid=pair_hidden_dim)
        self.pair_layer_norm_opm = LayerNorm(pair_dim)

        self.pair_layer_norm_ffn = LayerNorm(pair_dim)
        self.pair_ffn = Transition(
            pair_dim,
            1,
            dropout=activation_dropout,
        )

        self.pair_dropout = pair_dropout
        self.pair_layer_norm_trimul = LayerNorm(pair_dim)
        self.pair_tri_mul = TriangleMultiplication(pair_dim, pair_hidden_dim)

    def shared_dropout(self, x, shared_dim, dropout):
        shape = list(x.shape)
        shape[shared_dim] = 1
        with torch.no_grad():
            mask = x.new_ones(shape)
        return F.dropout(mask, p=dropout, training=self.training) * x

    def forward(
        self,
        x: torch.Tensor,
        pair: torch.Tensor,
        pair_mask: torch.Tensor,
        self_attn_mask: Optional[torch.Tensor] = None,
        op_mask: Optional[torch.Tensor] = None,
        op_norm: Optional[torch.Tensor] = None,
    ):
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer implementation.
        """
        residual = x
        x = self.self_attn(
            x,
            x,
            x,
            pair=pair,
            mask=self_attn_mask,
        )
        x = self.dropout_module(x)
        x = residual + x
        x = self.self_attn_layer_norm(x)

        residual = x
        x = self.ffn(x)
        x = self.dropout_module(x)

        x = residual + x
        x = self.final_layer_norm(x)

        # outer product
        pair = pair + self.dropout_module(self.opm(x, op_mask, op_norm))
        pair = self.pair_layer_norm_opm(pair)

        pair_update = self.shared_dropout(
            self.pair_tri_mul(pair, pair_mask), -3, self.pair_dropout
        )
        pair = pair + pair_update
        pair = self.pair_layer_norm_trimul(pair)

        # ffn
        pair = pair + self.dropout_module(self.pair_ffn(pair))
        pair = self.pair_layer_norm_ffn(pair)
        return x, pair
