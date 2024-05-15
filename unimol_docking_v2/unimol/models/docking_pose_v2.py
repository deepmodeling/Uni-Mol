# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import argparse
import torch
import torch.nn.functional as F
from unicore.models import BaseUnicoreModel, register_model, register_model_architecture
from unicore.data import data_utils
from .unimol import UniMolModel, base_architecture, NonLinearHead, DistanceHead, GaussianLayer
from .transformer_encoder_with_pair import TransformerEncoderWithPair
import numpy as np

logger = logging.getLogger(__name__)


@register_model("docking_pose_v2")
class DockingPoseV2Model(BaseUnicoreModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument(
            "--mol-pooler-dropout",
            type=float,
            metavar="D",
            help="dropout probability in the masked_lm pooler layers",
        )
        parser.add_argument(
            "--pocket-pooler-dropout",
            type=float,
            metavar="D",
            help="dropout probability in the masked_lm pooler layers",
        )
        parser.add_argument(
            "--pocket-encoder-layers",
            type=int,
            help="pocket encoder layers",
        )
        parser.add_argument(
            "--recycling",
            type=int,
            default=1,
            help="recycling nums of decoder",
        )
    def __init__(self, args, mol_dictionary, pocket_dictionary):
        super().__init__()
        unimol_docking_architecture(args)

        self.args = args
        self.mol_model = UniMolModel(args.mol, mol_dictionary)
        self.pocket_model = UniMolModel(args.pocket, pocket_dictionary)
        self.mol_dictionary = mol_dictionary
        self.pocket_dictionary = pocket_dictionary
        self.concat_decoder = TransformerEncoderWithPair(
            encoder_layers=4,
            embed_dim=args.mol.encoder_embed_dim,
            ffn_embed_dim=args.mol.encoder_ffn_embed_dim,
            attention_heads=args.mol.encoder_attention_heads,
            emb_dropout=0.1,
            dropout=0.1,
            attention_dropout=0.1,
            activation_dropout=0.0,
            activation_fn="gelu",
        )
        self.cross_distance_project = NonLinearHead(
            args.mol.encoder_embed_dim * 2 + args.mol.encoder_attention_heads, 1, 'relu'
        )
        self.holo_distance_project = DistanceHead(
            args.mol.encoder_embed_dim + args.mol.encoder_attention_heads, 'relu'
        )

        K = 128
        dict_size = len(mol_dictionary) + len(pocket_dictionary)
        n_edge_type = dict_size * dict_size
        self.concat_gbf = GaussianLayer(K, n_edge_type)
        self.concat_gbf_proj = NonLinearHead(
            K, args.mol.encoder_attention_heads, args.mol.activation_fn
        )

        self.coord_decoder = TransformerEncoderWithPair(
            encoder_layers=4,
            embed_dim=args.mol.encoder_embed_dim,
            ffn_embed_dim=args.mol.encoder_ffn_embed_dim,
            attention_heads=args.mol.encoder_attention_heads,
            emb_dropout=0.1,
            dropout=0.1,
            attention_dropout=0.1,
            activation_dropout=0.0,
            activation_fn="gelu",
        )
        self.coord_delta_project = NonLinearHead(
            args.mol.encoder_attention_heads, 1, args.mol.activation_fn
        )
        self.prmsd_project = NonLinearHead(
            args.mol.encoder_embed_dim, 32, args.mol.activation_fn
        )

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        return cls(args, task.dictionary, task.pocket_dictionary)

    def forward(
        self,
        mol_src_tokens,
        mol_src_distance,
        mol_src_coord,
        mol_src_edge_type,
        pocket_src_tokens,
        pocket_src_distance,
        pocket_src_coord,
        pocket_src_edge_type,
        masked_tokens=None,
        features_only=True,
        **kwargs
    ):

        def get_dist_features(dist, et, flag):
            if flag == 'mol':
                n_node = dist.size(-1)
                gbf_feature = self.mol_model.gbf(dist, et)
                gbf_result = self.mol_model.gbf_proj(gbf_feature)
                graph_attn_bias = gbf_result
                graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
                graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
                return graph_attn_bias
            elif flag == 'pocket':
                n_node = dist.size(-1)
                gbf_feature = self.pocket_model.gbf(dist, et)
                gbf_result = self.pocket_model.gbf_proj(gbf_feature)
                graph_attn_bias = gbf_result
                graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
                graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
                return graph_attn_bias
            elif flag == 'concat':
                n_node = dist.size(-1)
                gbf_feature = self.concat_gbf(dist, et)
                gbf_result = self.concat_gbf_proj(gbf_feature)
                graph_attn_bias = gbf_result
                graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
                graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
                return graph_attn_bias
            else:
                return None

        mol_padding_mask = mol_src_tokens.eq(self.mol_model.padding_idx)
        mol_atom_mask = mol_src_tokens > 2
        pocket_atom_mask = pocket_src_tokens > 2
        mol_x = self.mol_model.embed_tokens(mol_src_tokens)
        mol_graph_attn_bias = get_dist_features(mol_src_distance, mol_src_edge_type, 'mol')
        mol_outputs = self.mol_model.encoder(mol_x, padding_mask=mol_padding_mask, attn_mask=mol_graph_attn_bias)
        mol_encoder_rep = mol_outputs[0]
        mol_encoder_pair_rep = mol_outputs[1]

        pocket_padding_mask = pocket_src_tokens.eq(self.pocket_model.padding_idx)
        pocket_x = self.pocket_model.embed_tokens(pocket_src_tokens)
        pocket_graph_attn_bias = get_dist_features(pocket_src_distance, pocket_src_edge_type, 'pocket')
        pocket_outputs = self.pocket_model.encoder(pocket_x, padding_mask=pocket_padding_mask, attn_mask=pocket_graph_attn_bias)
        pocket_encoder_rep = pocket_outputs[0]
        pocket_encoder_pair_rep = pocket_outputs[1]

        mol_sz = mol_encoder_rep.size(1)
        pocket_sz = pocket_encoder_rep.size(1)
        cross_distance_mask, distance_mask, coord_mask = calc_mask(mol_atom_mask, pocket_atom_mask)

        concat_rep = torch.cat([mol_encoder_rep, pocket_encoder_rep], dim=-2) # [batch, mol_sz+pocket_sz, hidden_dim]
        concat_mask = torch.cat([mol_padding_mask, pocket_padding_mask], dim=-1)   # [batch, mol_sz+pocket_sz]
        attn_bs = mol_graph_attn_bias.size(0)

        concat_attn_bias = torch.zeros(attn_bs, mol_sz+pocket_sz, mol_sz+pocket_sz).type_as(concat_rep)  # [batch, mol_sz+pocket_sz, mol_sz+pocket_sz]
        concat_attn_bias[:,:mol_sz,:mol_sz] = mol_encoder_pair_rep.permute(0, 3, 1, 2).reshape(-1, mol_sz, mol_sz).contiguous()
        concat_attn_bias[:,-pocket_sz:,-pocket_sz:] = pocket_encoder_pair_rep.permute(0, 3, 1, 2).reshape(-1, pocket_sz, pocket_sz).contiguous()

        decoder_rep = concat_rep
        decoder_pair_rep = concat_attn_bias
        for i in range(self.args.recycling):
            decoder_outputs = self.concat_decoder(decoder_rep, padding_mask=concat_mask, attn_mask=decoder_pair_rep)
            decoder_rep = decoder_outputs[0]
            decoder_pair_rep = decoder_outputs[1]
            if i!=(self.args.recycling - 1):
                decoder_pair_rep = decoder_pair_rep.permute(0, 3, 1, 2).reshape(-1, mol_sz+pocket_sz, mol_sz+pocket_sz)

        decoder_rep = decoder_outputs[0]
        decoder_pair_rep = decoder_outputs[1]

        mol_decoder = decoder_rep[:,:mol_sz]
        pocket_decoder = decoder_rep[:,mol_sz:]

        mol_pair_decoder_rep = decoder_pair_rep[:,:mol_sz,:mol_sz,:]
        mol_pocket_pair_decoder_rep = (decoder_pair_rep[:,:mol_sz,mol_sz:,:] + decoder_pair_rep[:,mol_sz:,:mol_sz,:].transpose(1,2))/2.0
        mol_pocket_pair_decoder_rep[mol_pocket_pair_decoder_rep == float('-inf')] = 0

        cross_rep = torch.cat([
                                mol_pocket_pair_decoder_rep,
                                mol_decoder.unsqueeze(-2).repeat(1, 1, pocket_sz, 1), 
                                pocket_decoder.unsqueeze(-3).repeat(1, mol_sz, 1, 1), 
                                ], dim=-1)   # [batch, mol_sz, pocket_sz, 4*hidden_size]
    
        cross_distance_predict = F.elu(self.cross_distance_project(cross_rep).squeeze(-1)) + 1.0  # batch, mol_sz, pocket_sz

        holo_encoder_pair_rep = torch.cat([
                                mol_pair_decoder_rep,
                                mol_decoder.unsqueeze(-2).repeat(1, 1, mol_sz, 1), 
                                ], dim=-1) # [batch, mol_sz, mol_sz, 3*hidden_size]
        holo_distance_predict = self.holo_distance_project(holo_encoder_pair_rep)  # batch, mol_sz, mol_sz

        mol_src_coord_update = dock_with_gradient(mol_src_coord, pocket_src_coord, cross_distance_predict, holo_distance_predict, cross_distance_mask, distance_mask)

        num_types = len(self.mol_dictionary) + len(self.pocket_dictionary)
        node_input = torch.concat([mol_src_tokens, pocket_src_tokens + len(self.mol_dictionary)], dim=1) # [batch, mol_sz+pocket_sz]
        concat_edge_type = node_input.unsqueeze(-1) * num_types + node_input.unsqueeze(-2)
            
        def coord_decoder(mol_src_coord_update):

            concat_coord = torch.cat([mol_src_coord_update, pocket_src_coord], dim=1) # [batch, mol_sz+pocket_sz, 3]
            concat_distance = (concat_coord.unsqueeze(1) - concat_coord.unsqueeze(2)).norm(dim=-1)
            concat_attn_bias = get_dist_features(concat_distance, concat_edge_type, 'concat')
            concat_outputs = self.coord_decoder(decoder_rep, padding_mask=concat_mask, attn_mask=concat_attn_bias)
            coord_decoder_rep = concat_outputs[0]
            coord_decoder_rep = coord_decoder_rep[:,:mol_sz,:]
            delta_decoder_pair_rep = concat_outputs[2]   
            delta_decoder_rep = delta_decoder_pair_rep[:,:mol_sz,:mol_sz,:]

            atom_num = (torch.sum(~mol_padding_mask, dim=1) - 1).view(-1, 1, 1, 1)
            delta_pos = mol_src_coord_update.unsqueeze(1) - mol_src_coord_update.unsqueeze(2)
            attn_probs = self.coord_delta_project(delta_decoder_rep)
            coord_update = delta_pos / atom_num * attn_probs
            coord_update = torch.sum(coord_update, dim=2)
            mol_src_coord_update = mol_src_coord_update + coord_update #* 10

            return mol_src_coord_update, coord_decoder_rep

        if self.training:
            with data_utils.numpy_seed(self.get_num_updates()):
                recycling = np.random.randint(3)
                for i in range(recycling):
                    with torch.no_grad():
                        mol_src_coord_update, _ = coord_decoder(mol_src_coord_update)
            mol_src_coord_update, coord_decoder_rep = coord_decoder(mol_src_coord_update)
        else:
            recycling = 4
            for i in range(recycling):
                mol_src_coord_update, coord_decoder_rep = coord_decoder(mol_src_coord_update)

        prmsd_predict = self.prmsd_project(coord_decoder_rep)

        return cross_distance_predict, holo_distance_predict, mol_src_coord_update, prmsd_predict


    def set_num_updates(self, num_updates):
        """State from trainer to pass along to model at every update."""

        self._num_updates = num_updates

    def get_num_updates(self):
        return self._num_updates


def calc_mask(mol_padding_mask, pocket_padding_mask):
    mol_sz = mol_padding_mask.size()
    pocket_sz = pocket_padding_mask.size()
    cross_distance_mask = torch.zeros(mol_sz[0], mol_sz[1], pocket_sz[1]).type_as(mol_padding_mask)
    cross_distance_mask = mol_padding_mask.unsqueeze(-1) & pocket_padding_mask.unsqueeze(-2)
    distance_mask = torch.zeros(mol_sz[0], mol_sz[1], mol_sz[1]).type_as(mol_padding_mask)
    distance_mask = mol_padding_mask.unsqueeze(-1) & mol_padding_mask.unsqueeze(-2)
    coord_mask = torch.zeros(mol_sz[0], mol_sz[1], 3).type_as(mol_padding_mask)
    coord_mask.masked_fill_(
        mol_padding_mask.unsqueeze(-1),
        True
    )
    return cross_distance_mask, distance_mask, coord_mask


def scoring_function(predict_coords, pocket_coords, distance_predict, holo_distance_predict, cross_distance_mask, distance_mask, dist_threshold=4.5):
    dist = torch.norm(predict_coords.unsqueeze(-2) - pocket_coords.unsqueeze(-3), dim=-1)   # bs, mol_sz, pocket_sz
    holo_dist = torch.norm(predict_coords.unsqueeze(-2) - predict_coords.unsqueeze(-3), dim=-1) # bs, mol_sz, mol_sz

    cross_distance_mask = (distance_predict < dist_threshold) & cross_distance_mask
    cross_dist_score = ((dist[cross_distance_mask] - distance_predict[cross_distance_mask])**2).mean()
    dist_score = ((holo_dist[distance_mask] - holo_distance_predict[distance_mask])**2).mean()
    loss = cross_dist_score + dist_score
    return loss


def dock_with_gradient(mol_coords, pocket_coords, distance_predict, holo_distance_predict, cross_distance_mask, distance_mask, iterations=20, early_stoping=5):

    coords = torch.ones_like(mol_coords).type_as(mol_coords) * mol_coords
    coords.requires_grad = True
    optimizer = torch.optim.LBFGS([coords], lr=1.0)
    bst_loss, times = 10000.0, 0
    distance_predict_detached = distance_predict.detach().clone()
    holo_distance_predict_detached = holo_distance_predict.detach().clone()
    for i in range(iterations):
        def closure():
            optimizer.zero_grad()
            loss = scoring_function(coords, pocket_coords, distance_predict_detached, holo_distance_predict_detached, cross_distance_mask, distance_mask)
            loss.backward(retain_graph=True)
            return loss
        loss = optimizer.step(closure)
        if loss.item() < bst_loss:
            bst_loss = loss.item()
            times = 0 
        else:
            times += 1
            if times > early_stoping:
                break
    return coords.detach()


@register_model_architecture("docking_pose_v2", "docking_pose_v2")
def unimol_docking_architecture(args):

    parser = argparse.ArgumentParser()
    args.mol = parser.parse_args([])
    args.pocket = parser.parse_args([])

    args.mol.encoder_layers = getattr(args, "mol_encoder_layers", 15)
    args.mol.encoder_embed_dim = getattr(args, "mol_encoder_embed_dim", 512)
    args.mol.encoder_ffn_embed_dim = getattr(args, "mol_encoder_ffn_embed_dim", 2048)
    args.mol.encoder_attention_heads = getattr(args, "mol_encoder_attention_heads", 64)
    args.mol.dropout = getattr(args, "mol_dropout", 0.1)
    args.mol.emb_dropout = getattr(args, "mol_emb_dropout", 0.1)
    args.mol.attention_dropout = getattr(args, "mol_attention_dropout", 0.1)
    args.mol.activation_dropout = getattr(args, "mol_activation_dropout", 0.0)
    args.mol.pooler_dropout = getattr(args, "mol_pooler_dropout", 0.0)
    args.mol.max_seq_len = getattr(args, "mol_max_seq_len", 512)
    args.mol.activation_fn = getattr(args, "mol_activation_fn", "gelu")
    args.mol.pooler_activation_fn = getattr(args, "mol_pooler_activation_fn", "tanh")
    args.mol.post_ln = getattr(args, "mol_post_ln", False)
    args.mol.masked_token_loss = -1.0
    args.mol.masked_coord_loss = -1.0
    args.mol.masked_dist_loss = -1.0
    args.mol.x_norm_loss = -1.0
    args.mol.delta_pair_repr_norm_loss = -1.0

    args.pocket.encoder_layers = getattr(args, "pocket_encoder_layers", 15)
    args.pocket.encoder_embed_dim = getattr(args, "pocket_encoder_embed_dim", 512)
    args.pocket.encoder_ffn_embed_dim = getattr(args, "pocket_encoder_ffn_embed_dim", 2048)
    args.pocket.encoder_attention_heads = getattr(args, "pocket_encoder_attention_heads", 64)
    args.pocket.dropout = getattr(args, "pocket_dropout", 0.1)
    args.pocket.emb_dropout = getattr(args, "pocket_emb_dropout", 0.1)
    args.pocket.attention_dropout = getattr(args, "pocket_attention_dropout", 0.1)
    args.pocket.activation_dropout = getattr(args, "pocket_activation_dropout", 0.0)
    args.pocket.pooler_dropout = getattr(args, "pocket_pooler_dropout", 0.0)
    args.pocket.max_seq_len = getattr(args, "pocket_max_seq_len", 512)
    args.pocket.activation_fn = getattr(args, "pocket_activation_fn", "gelu")
    args.pocket.pooler_activation_fn = getattr(args, "pocket_pooler_activation_fn", "tanh")
    args.pocket.post_ln = getattr(args, "pocket_post_ln", False)
    args.pocket.masked_token_loss = -1.0
    args.pocket.masked_coord_loss = -1.0
    args.pocket.masked_dist_loss = -1.0
    args.pocket.x_norm_loss = -1.0
    args.pocket.delta_pair_repr_norm_loss = -1.0
    
    base_architecture(args)