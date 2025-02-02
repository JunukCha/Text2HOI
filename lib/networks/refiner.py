import numpy as np

import torch
import torch.nn as nn

from lib.networks.texthom import TextHOM, InitFC, OutFC
from lib.utils.loss import (
    get_l2_loss, 
    get_penetration_loss, 
    get_joint_contact_loss, 
)


class Refiner(TextHOM):
    def __init__(
        self, 
        hand_input_nfeats, 
        hand_output_nfeats, 
        **kwargs
    ):
        super(Refiner, self).__init__(**kwargs)

        self.embed_text = None
        self.hand_input_nfeats = hand_input_nfeats
        self.hand_output_nfeats = hand_output_nfeats
        
        self.init_fc_lhand = InitFC(self.hand_input_nfeats, self.latent_dim)
        self.init_fc_rhand = InitFC(self.hand_input_nfeats, self.latent_dim)
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        self.sequence_hand_encoder = HandObjectEncoding(self.latent_dim, self.dropout)
        self.out_fc_lhand = OutFC(self.hand_output_nfeats, self.latent_dim)
        self.out_fc_rhand = OutFC(self.hand_output_nfeats, self.latent_dim)
        
    def forward(
        self, 
        input_lhand, input_rhand, 
        valid_mask_lhand=None, 
        valid_mask_rhand=None, 
    ):
        bs = input_lhand.shape[0]
        init_lhand = input_lhand[..., :99]
        init_rhand = input_rhand[..., :99]
        
        x_init_lhand = self.init_fc_lhand(input_lhand)
        x_init_rhand = self.init_fc_rhand(input_rhand)
        
        xseq = torch.stack((x_init_lhand, x_init_rhand), dim=1)
        xseq = xseq.reshape(-1, bs, self.latent_dim)

        xseq = self.sequence_pos_encoder(xseq)
        xseq = self.sequence_hand_encoder(xseq)
        if valid_mask_rhand is not None:
            aug_mask_no_emb = torch.stack([valid_mask_lhand, valid_mask_rhand], dim=2)
            aug_mask_no_emb = aug_mask_no_emb.reshape(bs, -1)
            aug_mask = aug_mask_no_emb
            x_enc = self.seqTransEncoder(xseq, src_key_padding_mask=~aug_mask)
        else:
            x_enc = self.seqTransEncoder(xseq, src_key_padding_mask=None)
        x_enc_lhand = x_enc[::2]
        x_enc_rhand = x_enc[1::2]

        offset_lhand = self.out_fc_lhand(x_enc_lhand)
        offset_rhand = self.out_fc_rhand(x_enc_rhand)
        refined_lhand = init_lhand + offset_lhand
        refined_rhand = init_rhand + offset_rhand
        return refined_lhand, refined_rhand
    
    def get_loss(
        self, input_lhand, input_rhand, refined_obj, 
        targ_lhand, targ_rhand, 
        obj_verts_org, lhand_obj_pc_cont, rhand_obj_pc_cont, 
        lhand_layer, rhand_layer, 
        ldist_map, rdist_map, 
        dataset_name, 
        valid_mask_lhand=None, 
        valid_mask_rhand=None, 
        lhand_cont_joint_mask=None, 
        rhand_cont_joint_mask=None, 
        lambda_dict=None, 
        contact_loss_type="l2", 
        obj_pc_top_idx=None, 
    ):
        refined_lhand, refined_rhand \
            = self.forward(
                input_lhand, input_rhand,  
                valid_mask_lhand=valid_mask_lhand, 
                valid_mask_rhand=valid_mask_rhand, 
            )
        
        lambda_simple = lambda_dict["lambda_simple"]
        lambda_penet = lambda_dict["lambda_penet"]
        lambda_contact = lambda_dict["lambda_contact"]

        if lambda_simple:
            simple_loss = get_l2_loss(
                pred_lhand=refined_lhand, pred_rhand=refined_rhand,
                targ_lhand=targ_lhand, targ_rhand=targ_rhand,
                mask_lhand=valid_mask_lhand, mask_rhand=valid_mask_rhand, 
            )
        else:
            lambda_simple = torch.FloatTensor(1).fill_(0).cuda()
        
        if lambda_penet:
            penet_loss = get_penetration_loss(
                refined_lhand, refined_rhand, refined_obj, 
                lhand_layer, rhand_layer, obj_verts_org, 
                valid_mask_lhand, valid_mask_rhand, 
                dataset_name, 
                obj_pc_top_idx=obj_pc_top_idx, 
            )
        else:
            penet_loss = torch.FloatTensor(1).fill_(0).cuda()
        
        if lambda_contact:
            contact_loss = get_joint_contact_loss(
                refined_lhand, refined_rhand, 
                lhand_obj_pc_cont, rhand_obj_pc_cont, 
                lhand_layer, rhand_layer,
                lhand_cont_joint_mask, rhand_cont_joint_mask, 
                loss_type=contact_loss_type
            )
        else:
            contact_loss = torch.FloatTensor(1).fill_(0).cuda()
            
        losses_dict = {}
        losses_dict["simple_loss"] = simple_loss
        losses_dict["penet_loss"] = penet_loss
        losses_dict["contact_loss"] = contact_loss
        return refined_lhand, refined_rhand, losses_dict

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x[::2] = x[::2] + self.pe[:x.shape[0]//2]
        x[1::2] = x[1::2] + self.pe[:x.shape[0]//2]
        return self.dropout(x)


class HandObjectEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(HandObjectEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x[::2] = x[::2] + self.pe[0:1]
        x[1::2] = x[1::2] + self.pe[len(self.pe)//3:len(self.pe)//3+1]
        return self.dropout(x)
    
    
class ResBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels, dropout=0.5):
        super().__init__()
        self.linear1 = nn.Linear(in_channels, hidden_channels)
        self.linear2 = nn.Linear(hidden_channels, in_channels)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.dropout = nn.Dropout(p=dropout, inplace=True)
    
    def forward(self, x):
        identity = x
        x = self.linear1(x)
        x = self.dropout(x)
        x = self.lrelu(x)
        x = self.linear2(x)
        x = self.dropout(x)
        x = self.lrelu(x)

        out = x + identity
        return out