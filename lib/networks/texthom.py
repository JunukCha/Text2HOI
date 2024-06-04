import numpy as np

import torch
import torch.nn as nn


class TextHOM(nn.Module):
    def __init__(self, hand_nfeats=99, obj_nfeats=9, latent_dim=512, 
                ff_size=1024, num_layers=8, num_heads=4, dropout=0.1,
                activation="gelu", clip_dim=512, obj_dim=1024,
                cond_mask_prob=0.1, use_cond_fc=True, 
                use_obj_scale_centroid=True, 
                use_contact_feat=True, 
                use_frame_pos=True, 
                use_inst_pos=True, 
                **kwargs):
        super().__init__()
        ### Variable

        self.cond_mask_prob = cond_mask_prob
        self.nfeats = hand_nfeats

        self.latent_dim = latent_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.activation = activation
        self.clip_dim = clip_dim
        self.obj_dim = obj_dim
        if use_contact_feat:
            self.obj_dim = 2048

        self.input_feats_hand = hand_nfeats
        self.input_feats_obj = obj_nfeats

        self.use_cond_fc = use_cond_fc
        self.use_obj_scale_centroid = use_obj_scale_centroid
        self.use_frame_pos = use_frame_pos
        self.use_inst_pos = use_inst_pos
        
        ### Architecture
        self.init_fc_lhand = InitFC(self.input_feats_hand, self.latent_dim)
        self.init_fc_rhand = InitFC(self.input_feats_hand, self.latent_dim)
        self.init_fc_obj = InitFC(self.input_feats_obj, self.latent_dim)
        if not self.use_frame_pos:
            self.sequence_pos_encoder = OrgPositionalEncoding(self.latent_dim, self.dropout)
        else:
            self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        self.sequence_hand_encoder = HandObjectEncoding(self.latent_dim, self.dropout)
        seqTransEncoderLayer = nn.TransformerEncoderLayer(
            d_model=self.latent_dim,
            nhead=self.num_heads,
            dim_feedforward=self.ff_size,
            dropout=self.dropout,
            activation=self.activation
        )
        self.seqTransEncoder = nn.TransformerEncoder(
            seqTransEncoderLayer,
            num_layers=self.num_layers
        )
        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)

        self.embed_text = nn.Linear(self.clip_dim, self.latent_dim)
        if use_obj_scale_centroid:
            self.embed_obj = nn.Linear(self.obj_dim+4, self.latent_dim)
        else:
            self.embed_obj = nn.Linear(self.obj_dim, self.latent_dim)

        if use_cond_fc:
            self.out_fc_lhand = CondOutFC(self.input_feats_hand, self.latent_dim)
            self.out_fc_rhand = CondOutFC(self.input_feats_hand, self.latent_dim)
        else:
            self.out_fc_lhand = OutFC(self.input_feats_hand, self.latent_dim)
            self.out_fc_rhand = OutFC(self.input_feats_hand, self.latent_dim)
        self.out_fc_obj = OutFC(self.input_feats_obj, self.latent_dim)
        
    def mask_cond(self, cond, force_mask=False):
        bs, d = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_mask_prob).view(bs, 1)  # 1-> use null_cond, 0-> use real cond
            return cond * (1. - mask)
        else:
            return cond
    
    def forward(
        self, x_lhand, x_rhand, 
        x_obj, obj_feat, 
        timesteps, enc_text, 
        valid_mask_lhand=None, 
        valid_mask_rhand=None, 
        valid_mask_obj=None
    ):
        bs = timesteps.shape[0]
        emb = self.embed_timestep(timesteps)
        emb += self.embed_text(self.mask_cond(enc_text, force_mask=False))
        emb += self.embed_obj(self.mask_cond(obj_feat, force_mask=False))
        x_init_lhand = self.init_fc_lhand(x_lhand)
        x_init_rhand = self.init_fc_rhand(x_rhand)
        x_init_obj = self.init_fc_obj(x_obj)

        x_init = torch.stack((x_init_lhand, x_init_rhand, x_init_obj), dim=1)
        x_init = x_init.reshape(-1, bs, self.latent_dim)

        xseq = torch.cat((emb, x_init), dim=0)
        xseq = self.sequence_pos_encoder(xseq)
        if self.use_inst_pos:
            xseq = self.sequence_hand_encoder(xseq)
        if valid_mask_obj is not None:
            bs = x_obj.shape[0]
            emb_mask = torch.ones((bs, 1), device=x_obj.device, dtype=bool)
            aug_mask_no_emb = torch.stack([valid_mask_lhand, valid_mask_rhand, valid_mask_obj], dim=2)
            aug_mask_no_emb = aug_mask_no_emb.reshape(bs, -1)
            aug_mask = torch.cat([emb_mask, aug_mask_no_emb], dim=1)
            x_enc = self.seqTransEncoder(xseq, src_key_padding_mask=~aug_mask)[1:]
        else:
            x_enc = self.seqTransEncoder(xseq, src_key_padding_mask=None)[1:]
        x_enc_lhand = x_enc[::3]
        x_enc_rhand = x_enc[1::3]
        x_enc_obj = x_enc[2::3]

        if self.use_cond_fc:
            pred_lhand = self.out_fc_lhand(x_enc_lhand, x_enc_obj)
            pred_rhand = self.out_fc_rhand(x_enc_rhand, x_enc_obj)
        else:
            pred_lhand = self.out_fc_lhand(x_enc_lhand)
            pred_rhand = self.out_fc_rhand(x_enc_rhand)
        pred_obj = self.out_fc_obj(x_enc_obj)
        return pred_lhand, pred_rhand, pred_obj


class InitFC(nn.Module):
    def __init__(self, input_feats, latent_dim):
        super().__init__()
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.fc = nn.Linear(self.input_feats, self.latent_dim)

    def forward(self, x):
        x = x.permute(1, 0, 2)
        x = self.fc(x)
        return x


class OutFC(nn.Module):
    def __init__(self, input_feats, latent_dim):
        super().__init__()
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.fc = nn.Linear(self.latent_dim, self.input_feats)
        
    def forward(self, x):
        x = self.fc(x)
        x = x.permute(1, 0, 2)
        return x
    

class CondOutFC(nn.Module):
    def __init__(self, input_feats, latent_dim):
        super().__init__()
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.fc = nn.Linear(self.latent_dim*2, self.input_feats)
        
    def forward(self, x, cond):
        x = self.fc(torch.cat([x, cond], dim=2))
        x = x.permute(1, 0, 2)
        return x


class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps]).permute(1, 0, 2)


class OrgPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(OrgPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)
    
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
        x[0] = x[0] + self.pe[0]
        x[1::3] = x[1::3] + self.pe[1:(x.shape[0]+2)//3]
        x[2::3] = x[2::3] + self.pe[1:(x.shape[0]+2)//3]
        x[3::3] = x[3::3] + self.pe[1:(x.shape[0]+2)//3]
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
        x[1::3] = x[1::3] + self.pe[1:2]
        x[2::3] = x[2::3] + self.pe[len(self.pe)//3:len(self.pe)//3+1]
        x[3::3] = x[3::3] + self.pe[len(self.pe)*2//3:len(self.pe)*2//3+1]
        return self.dropout(x)