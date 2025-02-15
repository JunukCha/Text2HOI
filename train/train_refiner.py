import os
import os.path as osp
import sys
sys.path.append(osp.dirname(osp.abspath(osp.dirname(__file__))))

import tqdm
import numpy as np
import hydra
from omegaconf import OmegaConf
from easydict import EasyDict as edict

import torch
import torch.nn as nn
import torch.optim as optim
from pytorch3d.structures import Meshes

from lib.models.mano import build_mano_aa
from lib.networks.clip import load_and_freeze_clip, encoded_text
from lib.datasets.datasets import get_dataloader
from lib.utils.model_utils import (
    build_refiner, 
    build_model_and_diffusion, 
    build_seq_cvae, 
    build_pointnetfeat, 
    build_contact_estimator, 
)
from lib.models.object import build_object_model
from lib.utils.renderer import Renderer
from lib.utils.metric import AverageMeter
from lib.utils.file import (
    make_model_result_folder, 
    wandb_login, 
    save_video, 
)
from lib.utils.eval import (
    get_object_hand_info, 
    get_valid_mask_bunch, 
    proc_results, 
)
from lib.utils.proc import (
    proc_obj_feat_final, 
    proc_obj_feat_final_train, 
    proc_refiner_input, 
)
from lib.utils.visualize import render_videos


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(config):
    print(OmegaConf.to_yaml(config))
    config = OmegaConf.to_object(config)
    config = edict(config)
    data_config = config.dataset
    dataset_name = data_config.name
    
    wandb = wandb_login(config, config.refiner, relogin=False)

    model_name = config.refiner.model_name
    best_model_name = model_name+"_best"
    save_root = config.refiner.save_root
    model_folder, result_folder = make_model_result_folder(save_root, train_type="train")
    
    lhand_layer = build_mano_aa(is_rhand=False, flat_hand=data_config.flat_hand).cuda()
    rhand_layer = build_mano_aa(is_rhand=True, flat_hand=data_config.flat_hand).cuda()
    dataloader = get_dataloader("Motion"+dataset_name, config, data_config)
    refiner = build_refiner(config)
    texthom, diffusion \
        = build_model_and_diffusion(config, lhand_layer, rhand_layer, test=True)
    seq_cvae = build_seq_cvae(config, test=True)
    pointnet = build_pointnetfeat(config, test=True)
    object_model = build_object_model(data_config.data_obj_pc_path)
    dump_data = torch.randn([1, 1024, 3]).cuda()
    pointnet(dump_data)
    contact_estimator = build_contact_estimator(config, test=True)
    optimizer = optim.AdamW(refiner.parameters(), lr=config.refiner.lr)
    
    clip_model = load_and_freeze_clip(config.clip.clip_version)
    clip_model = clip_model.cuda()

    renderer = Renderer(device="cuda", camera=f"{dataset_name}_front")

    cur_loss = 9999
    best_loss = 9999
    best_epoch = 0
    
    hand_nfeats = config.texthom.hand_nfeats
    obj_nfeats = config.texthom.obj_nfeats
    
    contact_loss_type = config.refiner.contact_loss_type
    lambda_simple = config.refiner.lambda_simple
    lambda_penet = config.refiner.lambda_penet
    lambda_contact = config.refiner.lambda_contact
    lambda_dict = {
        "lambda_simple": lambda_simple, 
        "lambda_penet": lambda_penet, 
        "lambda_contact": lambda_contact, 
    }
    nepoch = config.refiner.iteration / (data_config.data_num/data_config.text_num)
    nepoch = int(np.ceil(nepoch / 50.0) * 50)
    with tqdm.tqdm(range(nepoch)) as pbar:
        for epoch in pbar:
            refiner.train()
            loss_meter = AverageMeter()
            loss_simple_meter = AverageMeter()
            loss_penet_meter = AverageMeter()
            loss_contact_meter = AverageMeter()
            for item in dataloader:
                if dataset_name == "arctic":
                    obj_pc_top_idx = item["obj_pc_top_idx"].cuda()
                else:
                    obj_pc_top_idx = None
                x_lhand = item["x_lhand"].cuda()
                x_rhand = item["x_rhand"].cuda()
                x_obj = item["x_obj"].cuda()
                obj_pc_org = item["obj_pc"].cuda()
                obj_pc_normal_org = item["obj_pc_normal"].cuda()
                normalized_obj_pc = item["normalized_obj_pc"].cuda()
                obj_scale = item["obj_scale"].cuda()
                obj_cent = item["obj_cent"].cuda()
                ldist_map = item["ldist_map"].cuda()
                rdist_map = item["rdist_map"].cuda()
                cov_map = item["cov_map"].cuda()
                
                valid_mask_lhand = item["valid_mask_lhand"].cuda()
                valid_mask_rhand = item["valid_mask_rhand"].cuda()
                valid_mask_obj = item["valid_mask_obj"].cuda()

                text = item["text"]
                enc_text = encoded_text(clip_model, text)

                bs = x_obj.shape[0]
                with torch.no_grad():
                    obj_feat = pointnet(normalized_obj_pc)
                    obj_feat_final = proc_obj_feat_final_train(
                        cov_map, obj_scale, obj_cent, obj_feat, 
                        config.texthom.use_obj_scale_centroid, 
                        config.texthom.use_contact_feat, 
                    )
                    coarse_x_lhand, coarse_x_rhand, coarse_x_obj \
                        = diffusion(
                            texthom, x_lhand, x_rhand, 
                            x_obj, obj_feat_final, 
                            enc_text=enc_text, get_losses=False, 
                            valid_mask_lhand=valid_mask_lhand, 
                            valid_mask_rhand=valid_mask_rhand, 
                            valid_mask_obj=valid_mask_obj, 
                            ldist_map=ldist_map, 
                            rdist_map=rdist_map, 
                            obj_verts_org=obj_pc_org, 
                            obj_pc_top_idx=obj_pc_top_idx, 
                        )

                    input_lhand, input_rhand, refined_obj, \
                    lhand_obj_pc_cont, rhand_obj_pc_cont, \
                    lhand_cont_joint_mask, rhand_cont_joint_mask, \
                        = proc_refiner_input(
                            coarse_x_lhand, coarse_x_rhand, coarse_x_obj, 
                            lhand_layer, rhand_layer, obj_pc_org, obj_pc_normal_org, 
                            valid_mask_lhand, valid_mask_rhand, valid_mask_obj, 
                            cov_map, dataset_name, return_psuedo_gt=True, obj_pc_top_idx=obj_pc_top_idx, 
                        )

                refined_lhand, refined_rhand, losses_dict \
                    = refiner.get_loss(
                        input_lhand, input_rhand, refined_obj, 
                        x_lhand, x_rhand,
                        obj_pc_org, lhand_obj_pc_cont, rhand_obj_pc_cont, 
                        lhand_layer, rhand_layer, 
                        ldist_map, rdist_map, 
                        dataset_name, 
                        valid_mask_lhand=valid_mask_lhand, 
                        valid_mask_rhand=valid_mask_rhand, 
                        lhand_cont_joint_mask=lhand_cont_joint_mask, 
                        rhand_cont_joint_mask=rhand_cont_joint_mask, 
                        lambda_dict=lambda_dict, 
                        contact_loss_type=contact_loss_type,
                        obj_pc_top_idx=obj_pc_top_idx,  
                    )
                
                simple_loss = losses_dict["simple_loss"]
                penet_loss = losses_dict["penet_loss"]
                contact_loss = losses_dict["contact_loss"]

                losses = lambda_simple*simple_loss \
                       + lambda_penet*penet_loss \
                       + lambda_contact*contact_loss
                
                optimizer.zero_grad()
                losses.backward()
                optimizer.step()
                loss_meter.update(losses.item(), bs)
                loss_simple_meter.update(simple_loss.item(), bs)
                loss_penet_meter.update(penet_loss.item(), bs)
                loss_contact_meter.update(contact_loss.item(), bs)
            
            cur_loss = loss_meter.avg
            if cur_loss < best_loss:
                best_loss = cur_loss
                best_epoch = epoch+1
                torch.save(
                    {
                        "model": refiner.state_dict(), 
                        "epoch": epoch, 
                        "loss": cur_loss, 
                    },
                    osp.join(model_folder, best_model_name+".pth"), 
                )
            wandb.log(
                {
                    "loss": cur_loss,
                    "simple_loss": loss_simple_meter.avg,
                    "penet_loss": loss_penet_meter.avg,
                    "contact_loss": loss_contact_meter.avg,
                }
            )
            pbar.set_description(f"{model_name} | Best loss: {best_loss:.4f} ({best_epoch}), Cur loss: {cur_loss:.4f}")
            
            if (epoch+1)%config.save_pth_freq==0:
                torch.save(
                    {
                        "model": refiner.state_dict(), 
                        "epoch": epoch, 
                        "loss": cur_loss, 
                    },
                    osp.join(model_folder, model_name+f"_{epoch+1}.pth"), 
                )

            refiner.eval()
            if (epoch+1)%config.result_show_freq==0:
                with torch.no_grad():
                    max_nframes = data_config.max_nframes
                    text = config.text

                    is_lhand, is_rhand, \
                    obj_pc_org, obj_pc_normal_org, \
                    normalized_obj_pc, point_sets, \
                    obj_cent, obj_scale, \
                    obj_verts, obj_faces, \
                    obj_top_idx, obj_pc_top_idx \
                        = get_object_hand_info(
                            object_model, clip_model, text, 
                            data_config.obj_root, data_config
                        )
                    bs, npts = normalized_obj_pc.shape[:2]
                    
                    enc_text = encoded_text(clip_model, text)
                    
                    obj_feat = pointnet(normalized_obj_pc)
                    obj_feat_final, est_contact_map = proc_obj_feat_final(
                        contact_estimator, obj_scale, obj_cent, 
                        obj_feat, enc_text, npts, 
                        config.texthom.use_obj_scale_centroid, config.contact.use_scale, config.texthom.use_contact_feat
                    )
                    duration = seq_cvae.decode(enc_text)
                    duration *= 150
                    duration = duration.long()
                    valid_mask_lhand, valid_mask_rhand, valid_mask_obj \
                            = get_valid_mask_bunch(is_lhand, is_rhand, max_nframes, duration)
                    coarse_x_lhand, coarse_x_rhand, coarse_x_obj \
                        = diffusion.sampling(
                            texthom, obj_feat_final, 
                            enc_text, max_nframes, 
                            hand_nfeats, obj_nfeats, 
                            valid_mask_lhand, 
                            valid_mask_rhand, 
                            valid_mask_obj, 
                            device=torch.device("cuda")
                        )
                        
                    input_lhand, input_rhand, refined_x_obj \
                        = proc_refiner_input(
                            coarse_x_lhand, coarse_x_rhand, coarse_x_obj, 
                            lhand_layer, rhand_layer, obj_pc_org, obj_pc_normal_org, 
                            valid_mask_lhand, valid_mask_rhand, valid_mask_obj, 
                            est_contact_map, dataset_name, obj_pc_top_idx=obj_pc_top_idx, 
                        )

                    refined_x_lhand, refined_x_rhand \
                        = refiner(
                            input_lhand, input_rhand,  
                            valid_mask_lhand=valid_mask_lhand, 
                            valid_mask_rhand=valid_mask_rhand, 
                        )
                    for text_idx in range(len(text)):
                        text_duration = duration[text_idx].item()
                        obj_verts_text = obj_verts[text_idx]
                        obj_faces_text = obj_faces[text_idx]
                        point_set_text = point_sets[text_idx]
                        est_contact_map_text = est_contact_map[text_idx]
                        is_lhand_text = is_lhand[text_idx]
                        is_rhand_text = is_rhand[text_idx]
                        if dataset_name == "arctic":
                            obj_top_idx_text = obj_top_idx[text_idx]
                        else:
                            obj_top_idx_text = None
                            
                        coarse_x_lhand_sampled = coarse_x_lhand[text_idx][:text_duration]
                        coarse_x_rhand_sampled = coarse_x_rhand[text_idx][:text_duration]
                        coarse_x_obj_sampled = coarse_x_obj[text_idx][:text_duration]
                        
                        refined_x_lhand_sampled = refined_x_lhand[text_idx][:text_duration]
                        refined_x_rhand_sampled = refined_x_rhand[text_idx][:text_duration]
                        refined_x_obj_sampled = refined_x_obj[text_idx][:text_duration]

                        refined_obj_verts_tf, refined_lhand_verts, lhand_faces, \
                        refined_rhand_verts, rhand_faces = \
                            proc_results(
                                refined_x_lhand_sampled, refined_x_rhand_sampled, refined_x_obj_sampled, 
                                obj_verts_text, lhand_layer, rhand_layer, 
                                is_lhand_text, is_rhand_text, 
                                dataset_name, obj_top_idx_text
                            )
                        
                        merged_video = render_videos(
                            renderer, 
                            refined_lhand_verts, lhand_faces, 
                            refined_rhand_verts, rhand_faces, 
                            refined_obj_verts_tf, obj_faces_text, 
                            is_lhand_text, is_rhand_text, 
                        )
                        save_video(
                            merged_video, fps=30, 
                            save_path=osp.join(
                                result_folder, 
                                f"{text[text_idx]}_{epoch+1}.mp4"
                            )
                        )
                        
    torch.save(
        {
            "model": refiner.state_dict(), 
            "epoch": nepoch, 
            "loss": cur_loss, 
        },
        osp.join(model_folder, model_name+".pth"), 
    )
    wandb.finish()


if __name__ == "__main__":
    main()