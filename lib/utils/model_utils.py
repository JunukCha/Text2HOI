import os
import os.path as osp

from sentence_transformers import SentenceTransformer

import torch
import torch.nn as nn

from lib.networks.cvae import SeqCVAE, CTCVAE
from lib.networks.texthom import TextHOM
from lib.networks.refiner import Refiner
from lib.networks.diffusion import Diffusion
from lib.networks.pointnet import PointNetfeat

def init_weights_to_zero(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        m.weight.data.fill_(0.0)
        if m.bias is not None:
            m.bias.data.fill_(0.0)

def build_mpnet(args):
    print(f"build mpnet {args.mpnet.version}")
    mpnet = SentenceTransformer(args.mpnet.version)
    mpnet = mpnet.cuda()
    return mpnet

def build_refiner(args):
    args_refiner = args.refiner
    weight_path = args_refiner.weight_path
    assert osp.exists(weight_path), f"{weight_path} deosn't exist!"
    
    refiner = Refiner(**args_refiner)
    refiner = refiner.cuda()
    print("build refiner")
    if args_refiner.zero_initialized:
        print("initialize refiner with zero")
        refiner.apply(init_weights_to_zero)
    
    print(f"load refiner, {weight_path}")
    checkpoints = torch.load(weight_path)
    refiner.load_state_dict(checkpoints["model"])
    refiner.eval()
    for p in refiner.parameters():
        p.requires_grad = False
    return refiner

def build_model_and_diffusion(args, lhand_layer, rhand_layer):
    args_texthom = args.texthom
    args_diffusion = args.diffusion
    weight_path = args_texthom.weight_path
    assert osp.exists(weight_path), f"{weight_path} deosn't exist!"
    
    texthom = TextHOM(**args_texthom)
    diffusion = Diffusion(
        lhand_layer=lhand_layer, 
        rhand_layer=rhand_layer, 
        **args_diffusion
    )
    texthom = texthom.cuda()
    diffusion = diffusion.cuda()
    print("build texthom, diffusion")
    
    print(f"load texthom, {weight_path}")
    checkpoints = torch.load(weight_path)
    texthom.load_state_dict(checkpoints["model"])
    texthom.eval()
    diffusion.eval()
    for p in texthom.parameters():
        p.requires_grad = False
    return texthom, diffusion

def build_seq_cvae(args):
    args_cvae = args.seq_cvae
    weight_path = args_cvae.weight_path
    assert osp.exists(weight_path), f"{weight_path} deosn't exist!"
    
    seq_cvae = SeqCVAE(**args_cvae)
    seq_cvae = seq_cvae.cuda()
    print("build seq cvae")
    
    print(f"load seq cvae, {weight_path}")
    checkpoints = torch.load(weight_path)
    seq_cvae.load_state_dict(checkpoints["model"])
    seq_cvae.eval()
    for p in seq_cvae.parameters():
        p.requires_grad = False
    return seq_cvae

def build_pointnetfeat(args):
    args_pointfeat = args.pointfeat
    weight_path = args_pointfeat.weight_path
    assert osp.exists(weight_path), f"{weight_path} deosn't exist!"
    
    point_encoder = PointNetfeat(**args_pointfeat)
    point_encoder = point_encoder.cuda()
    print("build point encoder")
    
    print(f"load point encoder, {weight_path}")
    checkpoints = torch.load(weight_path)
    point_encoder.load_state_dict(checkpoints["model"])
    point_encoder.eval()
    for p in point_encoder.parameters():
        p.requires_grad = False
    return point_encoder

def build_contact_estimator(args):
    args_contact = args.contact
    weight_path = args_contact.weight_path
    assert osp.exists(weight_path), f"{weight_path} deosn't exist!"
    
    contact_estimator = CTCVAE(**args_contact)
    contact_estimator = contact_estimator.cuda()
    print("build contact estimator")

    print(f"load contact estimator, {weight_path}")
    checkpoints = torch.load(weight_path)
    contact_estimator.load_state_dict(checkpoints["model"])
    contact_estimator.eval()
    for p in contact_estimator.parameters():
        p.requires_grad = False
    return contact_estimator