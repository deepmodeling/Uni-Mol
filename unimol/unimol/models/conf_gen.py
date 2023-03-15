import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from unicore import utils
from unicore.models import BaseUnicoreModel, register_model, register_model_architecture
from unicore.data import Dictionary
from .unimol import UniMolModel
from unicore.modules import LayerNorm
from typing import Optional, Dict, Any, List
from .unimol import base_architecture

logger = logging.getLogger(__name__)


@register_model("mol_confG")
class UnimolConfGModel(BaseUnicoreModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument(
            "--distance-loss",
            type=float,
            default=1.0,
            help="weight for the distance loss",
        )
        parser.add_argument(
            "--coord-loss",
            type=float,
            default=1.0,
            help="weight for the coordinate loss",
        )

        parser.add_argument(
            "--num-recycles",
            type=int,
            default=1,
            help="number of cycles to use for coordinate prediction",
        )

    def __init__(self, args, mol_dictionary):
        super().__init__()
        unimol_confG_architecture(args)
        self.args = args
        self.unimol = UniMolModel(self.args, mol_dictionary)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        return cls(args, task.dictionary)

    def forward(
        self,
        src_tokens,
        src_distance,
        src_coord,
        src_edge_type,
        encoder_masked_tokens=None,
        **kwargs
    ):
        def get_dist_features(dist, et):
            n_node = dist.size(-1)
            gbf_feature = self.unimol.gbf(dist, et)
            gbf_result = self.unimol.gbf_proj(gbf_feature)
            graph_attn_bias = gbf_result
            graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
            graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
            return graph_attn_bias

        def fill_attn_mask(attn_mask, padding_mask, fill_val=float("-inf")):
            if attn_mask is not None and padding_mask is not None:
                # merge key_padding_mask and attn_mask
                attn_mask = attn_mask.view(x.size(0), -1, seq_len, seq_len)
                attn_mask.masked_fill_(
                    padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                    fill_val,
                )
                attn_mask = attn_mask.view(-1, seq_len, seq_len)
                padding_mask = None
            return attn_mask, padding_mask

        def single_encoder(
            emb: torch.Tensor,
            attn_mask: Optional[torch.Tensor] = None,
            padding_mask: Optional[torch.Tensor] = None,
        ):
            x = self.unimol.encoder.emb_layer_norm(emb)
            x = F.dropout(x, p=self.unimol.encoder.emb_dropout, training=self.training)

            if padding_mask is not None:
                x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))
            attn_mask, padding_mask = fill_attn_mask(
                attn_mask, padding_mask, fill_val=float("-inf")
            )

            for i in range(len(self.unimol.encoder.layers)):
                x, attn_mask, _ = self.unimol.encoder.layers[i](
                    x, padding_mask=padding_mask, attn_bias=attn_mask, return_attn=True
                )

            return x, attn_mask

        padding_mask = src_tokens.eq(self.unimol.padding_idx)
        input_padding_mask = padding_mask
        x = self.unimol.embed_tokens(src_tokens)
        attn_mask = get_dist_features(src_distance, src_edge_type)
        input_attn_mask = attn_mask
        bsz = x.size(0)
        seq_len = x.size(1)

        for _ in range(self.args.num_recycles):
            x, attn_mask = single_encoder(
                x, padding_mask=padding_mask, attn_mask=attn_mask
            )

        if self.unimol.encoder.final_layer_norm is not None:
            x = self.unimol.encoder.final_layer_norm(x)

        delta_pair_repr = attn_mask - input_attn_mask
        delta_pair_repr, _ = fill_attn_mask(delta_pair_repr, input_padding_mask, 0)
        attn_mask, _ = fill_attn_mask(attn_mask, input_padding_mask, 0)
        attn_mask = (
            attn_mask.view(bsz, -1, seq_len, seq_len).permute(0, 2, 3, 1).contiguous()
        )
        delta_pair_repr = (
            delta_pair_repr.view(bsz, -1, seq_len, seq_len)
            .permute(0, 2, 3, 1)
            .contiguous()
        )

        distance_predict, coords_predict = None, None

        if self.args.coord_loss > 0 or True:
            if padding_mask is not None:
                atom_num = (torch.sum(~padding_mask, dim=1) - 1).view(-1, 1, 1, 1)
            else:
                atom_num = src_coord.shape[1] - 1
            delta_pos = src_coord.unsqueeze(1) - src_coord.unsqueeze(2)
            attn_probs = self.unimol.pair2coord_proj(delta_pair_repr)
            coords_update = delta_pos / atom_num * attn_probs
            coords_update = torch.sum(coords_update, dim=2)
            coords_predict = src_coord + coords_update

        if self.args.distance_loss > 0 or True:
            distance_predict = self.unimol.dist_head(attn_mask)

        return [distance_predict, coords_predict]


@register_model_architecture("mol_confG", "mol_confG")
def unimol_confG_architecture(args):
    def base_architecture(args):
        args.encoder_layers = getattr(args, "encoder_layers", 15)
        args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
        args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
        args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 64)
        args.dropout = getattr(args, "dropout", 0.1)
        args.emb_dropout = getattr(args, "emb_dropout", 0.1)
        args.attention_dropout = getattr(args, "attention_dropout", 0.1)
        args.activation_dropout = getattr(args, "activation_dropout", 0.0)
        args.pooler_dropout = getattr(args, "pooler_dropout", 0.0)
        args.max_seq_len = getattr(args, "max_seq_len", 512)
        args.activation_fn = getattr(args, "activation_fn", "gelu")
        args.pooler_activation_fn = getattr(args, "pooler_activation_fn", "tanh")
        args.post_ln = getattr(args, "post_ln", False)
        args.masked_coord_loss = getattr(args, "masked_coord_loss", 1.0)
        args.masked_dist_loss = getattr(args, "masked_dist_loss", 1.0)

    base_architecture(args)
