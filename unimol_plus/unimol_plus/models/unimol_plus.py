import logging

import numpy as np
import torch
import torch.nn as nn
from unicore import utils
from unicore.models import (
    BaseUnicoreModel,
    register_model,
    register_model_architecture,
)

from .layers import (
    AtomFeature,
    EdgeFeature,
    SE3InvariantKernel,
    MovementPredictionHead,
    EnergyHead,
    Linear,
    Embedding,
)
from .unimol_plus_encoder import UnimolPLusEncoder

logger = logging.getLogger(__name__)


def init_params(module):
    def normal_(data):
        data.copy_(data.cpu().normal_(mean=0.0, std=0.02).to(data.device))

    if isinstance(module, nn.Linear) or isinstance(module, Linear):
        normal_(module.weight.data)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding) or isinstance(module, Embedding):
        normal_(module.weight.data)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()


@register_model("unimol_plus")
class UnimolPlusModel(BaseUnicoreModel):
    """
    Class for training a Masked Language Model. It also supports an
    additional sentence level prediction if the sent-loss argument is set.
    """

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # the model releated parameters
        parser.add_argument(
            "--num-3d-bias-kernel",
            type=int,
            metavar="D",
            help="number of kernel in 3D attention bias",
        )
        parser.add_argument(
            "--droppath-prob",
            type=float,
            metavar="D",
            help="stochastic path probability",
            default=0.0,
        )

        parser.add_argument(
            "--dropout", type=float, metavar="D", help="dropout probability"
        )
        parser.add_argument(
            "--attention-dropout",
            type=float,
            metavar="D",
            help="dropout probability for attention weights",
        )
        parser.add_argument(
            "--act-dropout",
            type=float,
            metavar="D",
            help="dropout probability for activation in FFN",
        )
        # Arguments related to input and output embeddings
        parser.add_argument(
            "--embed-dim",
            type=int,
            metavar="N",
            help="embedding dimension",
        )
        # Arguments related to hidden states and self-attention
        parser.add_argument(
            "--ffn-embed-dim",
            type=int,
            metavar="N",
            help="embedding dimension for FFN",
        )
        parser.add_argument(
            "--pair-embed-dim",
            type=int,
            metavar="N",
            help="embedding dimension for pair repr",
        )
        parser.add_argument(
            "--pair-hidden-dim",
            type=int,
            metavar="N",
        )
        parser.add_argument(
            "--pair-dropout",
            type=float,
            metavar="D",
            help="dropout probability for pair",
        )
        parser.add_argument(
            "--layers", type=int, metavar="N", help="num encoder layers"
        )
        parser.add_argument(
            "--attention-heads",
            type=int,
            metavar="N",
            help="num encoder attention heads",
        )
        parser.add_argument(
            "--activation-fn",
            choices=utils.get_available_activation_fns(),
            help="activation function to use",
        )
        parser.add_argument(
            "--num-block",
            type=int,
            help="number of iterations",
        )
        parser.add_argument(
            "--pos-step-size",
            type=float,
            help="step size for pos update",
        )
        parser.add_argument(
            "--gaussian-std-width",
            type=float,
        )
        parser.add_argument(
            "--gaussian-mean-start",
            type=float,
        )
        parser.add_argument(
            "--gaussian-mean-stop",
            type=float,
        )
        # training related parameters
        parser.add_argument(
            "--noise-scale",
            default=0.2,
            type=float,
            help="coordinate noise for masked atoms",
        )
        parser.add_argument(
            "--label-prob",
            default=0.7,
            type=float,
            help="the probability of using label conformer as input",
        )
        parser.add_argument(
            "--mid-prob",
            default=0.1,
            type=float,
            help="the probability of using label conformer as input",
        )
        parser.add_argument(
            "--mid-upper",
            default=0.6,
            type=float,
            help="the probability of using label conformer as input",
        )
        parser.add_argument(
            "--mid-lower",
            default=0.4,
            type=float,
            help="the probability of using label conformer as input",
        )
        parser.add_argument(
            "--pos-loss-weight",
            default=0.2,
            type=float,
            help="loss weight for pos",
        )

    def __init__(self, args):
        super().__init__()
        base_architecture(args)
        self.args = args
        self.molecule_encoder = UnimolPLusEncoder(
            num_encoder_layers=args.layers,
            embedding_dim=args.embed_dim,
            pair_dim=args.pair_embed_dim,
            pair_hidden_dim=args.pair_hidden_dim,
            ffn_embedding_dim=args.ffn_embed_dim,
            num_attention_heads=args.attention_heads,
            dropout=args.dropout,
            attention_dropout=args.attention_dropout,
            activation_dropout=args.act_dropout,
            activation_fn=args.activation_fn,
            droppath_prob=args.droppath_prob,
        )
        num_atom = 512
        num_degree = 128
        num_edge = 64
        num_pair = 512
        num_spatial = 512
        embedding_dim = args.embed_dim
        num_3d_bias_kernel = args.num_3d_bias_kernel
        self.atom_feature = AtomFeature(
            num_atom=num_atom,
            num_degree=num_degree,
            hidden_dim=embedding_dim,
        )

        self.edge_feature = EdgeFeature(
            pair_dim=args.pair_embed_dim,
            num_edge=num_edge,
            num_spatial=num_spatial,
        )

        self.se3_invariant_kernel = SE3InvariantKernel(
            pair_dim=args.pair_embed_dim,
            num_pair=num_pair,
            num_kernel=num_3d_bias_kernel,
            std_width=args.gaussian_std_width,
            start=args.gaussian_mean_start,
            stop=args.gaussian_mean_stop,
        )
        self.energy_head = EnergyHead(args.embed_dim, 1)
        self.movement_pred_head = MovementPredictionHead(
            args.embed_dim, args.pair_embed_dim, args.attention_heads
        )
        self.movement_pred_head.zero_init()
        self._num_updates = 0
        self.dtype = torch.float32

    def half(self):
        super().half()
        self.se3_invariant_kernel = self.se3_invariant_kernel.float()
        self.atom_feature = self.atom_feature.float()
        self.edge_feature = self.edge_feature.float()
        self.energy_head = self.energy_head.float()
        self.dtype = torch.half
        return self

    def bfloat16(self):
        super().bfloat16()
        self.se3_invariant_kernel = self.se3_invariant_kernel.float()
        self.atom_feature = self.atom_feature.float()
        self.edge_feature = self.edge_feature.float()
        self.energy_head = self.energy_head.float()
        self.dtype = torch.bfloat16
        return self

    def float(self):
        super().float()
        self.dtype = torch.float32
        return self

    def forward(self, batched_data):
        data_x = batched_data["atom_feat"]
        atom_mask = batched_data["atom_mask"]
        pair_type = batched_data["pair_type"]
        pos = batched_data["pos"]

        num_block = self.args.num_block

        n_mol, n_atom = data_x.shape[:2]
        x = self.atom_feature(batched_data)

        dtype = self.dtype

        x = x.type(dtype)

        attn_mask = batched_data["attn_bias"].clone()
        attn_bias = torch.zeros_like(attn_mask)
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.args.attention_heads, 1, 1)
        attn_bias = attn_bias.unsqueeze(-1).repeat(1, 1, 1, self.args.pair_embed_dim)
        attn_bias = self.edge_feature(batched_data, attn_bias)
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
            x, pair = self.molecule_encoder(
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
            node_output = node_output * self.args.pos_step_size
            if return_x:
                return x, pos + node_output
            else:
                return pos + node_output

        for _ in range(num_block - 1):
            pos = one_block(x, pos)
        x, pos = one_block(x, pos, return_x=True)
        pred_y = self.energy_head(x[:, 0, :]).view(-1)

        pred_pos = pos
        pred_dist = (pred_pos.unsqueeze(1) - pred_pos.unsqueeze(2)).norm(dim=-1)

        return (
            pred_y,
            pred_pos,
            pred_dist,
        )

    @classmethod
    def build_model(cls, args, task):
        return cls(args)

    def set_num_updates(self, num_updates):
        """State from trainer to pass along to model at every update."""
        self._num_updates = num_updates


@register_model_architecture("unimol_plus", "unimol_plus_base")
def base_architecture(args):
    args.embed_dim = getattr(args, "embed_dim", 768)
    args.pair_embed_dim = getattr(args, "pair_embed_dim", 256)
    args.pair_hidden_dim = getattr(args, "pair_hidden_dim", 32)
    args.layers = getattr(args, "layers", 12)
    args.attention_heads = getattr(args, "attention_heads", 48)
    args.ffn_embed_dim = getattr(args, "ffn_embed_dim", 768)
    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.droppath_prob = getattr(args, "droppath_prob", 0.1)
    args.pair_dropout = getattr(args, "pair_dropout", 0.25)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.act_dropout = getattr(args, "act_dropout", 0.1)
    args.dropout = getattr(args, "dropout", 0.0)
    args.num_3d_bias_kernel = getattr(args, "num_3d_bias_kernel", 128)
    args.num_block = getattr(args, "num_block", 2)
    args.pos_step_size = getattr(args, "pos_step_size", 0.01)
    args.gaussian_std_width = getattr(args, "gaussian_std_width", 1.0)
    args.gaussian_mean_start = getattr(args, "gaussian_mean_start", 0.0)
    args.gaussian_mean_stop = getattr(args, "gaussian_mean_stop", 9.0)


@register_model_architecture("unimol_plus", "unimol_plus_large")
def large_architecture(args):
    args.layers = getattr(args, "layers", 18)
    base_architecture(args)


@register_model_architecture("unimol_plus", "unimol_plus_small")
def small_architecture(args):
    args.layers = getattr(args, "layers", 6)
    base_architecture(args)
