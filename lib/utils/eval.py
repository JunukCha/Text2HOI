import os.path as osp

import glob
import trimesh
import numpy as np

import torch

from lib.networks.clip import encoded_text_normalized
from lib.utils.frame import get_valid_mask
from lib.utils.file import read_json
from lib.utils.proc import (
    proc_long_torch_cuda, 
    proc_torch_cuda, 
    pc_normalize, 
    farthest_point_sample, 
)
from lib.utils.data import(
    process_obj_result, 
    process_hand_result, 
)
from constants.h2o_constants import h2o_obj_name
from constants.grab_constants import grab_obj_name
from constants.arctic_constants import arctic_obj_name

def proc_results(
    x_lhand, x_rhand, x_obj, 
    obj_verts, lhand_layer, rhand_layer, 
    is_lhand, is_rhand, 
    dataset_name, obj_top_idx=None
):
    # Object
    if obj_top_idx is not None:
        obj_top_idx = proc_long_torch_cuda(obj_top_idx)
    obj_verts_tf = process_obj_result(obj_verts, x_obj, dataset_name, obj_top_idx)
    if is_lhand:
        lhand_verts, lhand_faces = process_hand_result(lhand_layer, x_lhand)
    else:
        lhand_verts, lhand_faces = None, None
    if is_rhand:
        rhand_verts, rhand_faces = process_hand_result(rhand_layer, x_rhand)
    else:
        rhand_verts, rhand_faces = None, None
        
    return obj_verts_tf, lhand_verts, lhand_faces, rhand_verts, rhand_faces

def get_object_hand_info_shap_e(
    clip_model, text, 
    obj_root, mpnet=None, 
):
    text_feat_clip = encoded_text_normalized(clip_model, text)
    if mpnet is not None:
        text_feat_mpnet = mpnet.encode(text, convert_to_tensor=True)
    else:
        text_feat_mpnet = None
        
    is_lhand_list = []
    is_rhand_list = []
    obj_pc_list = []
    normalized_obj_pc_list = []
    obj_pc_normal_list = []
    point_set_list = []
    obj_cent_list = []
    obj_scale_list = []
    obj_verts_list = []
    obj_faces_list = []
    for text_idx in range(len(text_feat_clip)):
        text_feat_clip_sp = text_feat_clip[text_idx] # sp: sampled
        if text_feat_mpnet is not None:
            text_feat_mpnet_sp = text_feat_mpnet[text_idx] # sp: sampled
        else:
            text_feat_mpnet_sp = None
        obj_names = glob.glob(osp.join(obj_root, "*"))
        obj_names = [osp.basename(obj_name) for obj_name in obj_names]
        object_name = search_object_shape_e(clip_model, mpnet, text_feat_clip_sp, text_feat_mpnet_sp, obj_names)
        is_lhand, is_rhand = search_hand(clip_model, mpnet, text_feat_clip_sp, text_feat_mpnet_sp)
        obj_file = get_shap_e_obj_file(obj_root, object_name)
        obj_mesh = trimesh.load(obj_file, maintain_order=True)
        # obj_mesh = trimesh.exchange.load.load_mesh(obj_file, process=False)
        obj_verts_org = proc_torch_cuda(obj_mesh.vertices.copy())
        obj_normal = obj_mesh.vertex_normals
        obj_normal = obj_normal / np.linalg.norm(obj_normal, axis=1, keepdims=True)
        point_set = farthest_point_sample(obj_verts_org.unsqueeze(0), 1024)
        point_set = point_set.squeeze(0).cpu().numpy()
        obj_pc = obj_verts_org[point_set].cpu().numpy()
        obj_pc_normal = obj_normal[point_set]
        
        normalized_obj_pc, obj_norm_cent, obj_norm_scale = pc_normalize(obj_pc, return_params=True)
        
        obj_pc = proc_torch_cuda(obj_pc).unsqueeze(0)
        obj_pc_normal = proc_torch_cuda(obj_pc_normal).unsqueeze(0)
        normalized_obj_pc = proc_torch_cuda(normalized_obj_pc).unsqueeze(0)
        obj_norm_cent = proc_torch_cuda(obj_norm_cent)
        
        is_lhand_list.append(is_lhand)
        is_rhand_list.append(is_rhand)
        obj_pc_list.append(obj_pc)
        obj_pc_normal_list.append(obj_pc_normal)
        normalized_obj_pc_list.append(normalized_obj_pc)
        point_set_list.append(point_set.tolist())
        obj_cent_list.append(obj_norm_cent)
        obj_scale_list.append(obj_norm_scale)
        obj_verts_list.append(obj_verts_org)
        obj_faces_list.append(obj_mesh.faces)
    
    obj_pc_list = torch.cat(obj_pc_list)
    obj_pc_normal_list = torch.cat(obj_pc_normal_list)
    normalized_obj_pc_list = torch.cat(normalized_obj_pc_list)
    point_set_list = proc_long_torch_cuda(point_set_list)
    obj_cent_list = torch.stack(obj_cent_list)
    obj_scale_list = proc_torch_cuda(obj_scale_list)
    return (
        is_lhand_list, is_rhand_list, 
        obj_pc_list, obj_pc_normal_list, 
        normalized_obj_pc_list, point_set_list, 
        obj_cent_list, obj_scale_list, 
        obj_verts_list, obj_faces_list, 
    )

def get_object_hand_info(
    object_model, 
    clip_model, text, 
    obj_root, data_config, 
    mpnet=None, 
):
    text_feat_clip = encoded_text_normalized(clip_model, text)
    if mpnet is not None:
        text_feat_mpnet = mpnet.encode(text, convert_to_tensor=True)
    else:
        text_feat_mpnet = None
        
    is_lhand_list = []
    is_rhand_list = []
    obj_pc_list = []
    normalized_obj_pc_list = []
    obj_pc_normal_list = []
    point_set_list = []
    obj_cent_list = []
    obj_scale_list = []
    obj_verts_list = []
    obj_faces_list = []
    obj_pc_top_idx_list = []
    obj_top_idx_list = []
    for text_idx in range(len(text_feat_clip)):
        text_feat_clip_sp = text_feat_clip[text_idx] # sp: sampled
        if text_feat_mpnet is not None:
            text_feat_mpnet_sp = text_feat_mpnet[text_idx] # sp: sampled
        else:
            text_feat_mpnet_sp = None
        object_name = search_object(clip_model, mpnet, text_feat_clip_sp, text_feat_mpnet_sp, data_config)
        is_lhand, is_rhand = search_hand(clip_model, mpnet, text_feat_clip_sp, text_feat_mpnet_sp)
        obj_file = get_obj_file(obj_root, object_name, data_config)
        obj_mesh = trimesh.load(obj_file, maintain_order=True)
        # obj_mesh = trimesh.exchange.load.load_mesh(obj_file, process=False)
        obj_verts_org = proc_torch_cuda(obj_mesh.vertices.copy())
        if data_config.name == "arctic":
            obj_verts_org = obj_verts_org/1000
            point_set, obj_pc, obj_pc_normal, _, obj_pc_top_idx = object_model(object_name)
            obj_top_file = get_obj_top_file(obj_root, object_name)
            obj_top_idx = read_json(obj_top_file)
            obj_top_idx = 1-np.array(obj_top_idx)
            obj_pc_top_idx_list.append(obj_pc_top_idx.tolist())
            obj_top_idx_list.append(obj_top_idx.tolist())
        else:
            point_set, obj_pc, obj_pc_normal, _ = object_model(object_name)
        normalized_obj_pc, obj_norm_cent, obj_norm_scale = pc_normalize(obj_pc, return_params=True)
        
        obj_pc = proc_torch_cuda(obj_pc).unsqueeze(0)
        obj_pc_normal = proc_torch_cuda(obj_pc_normal).unsqueeze(0)
        normalized_obj_pc = proc_torch_cuda(normalized_obj_pc).unsqueeze(0)
        obj_norm_cent = proc_torch_cuda(obj_norm_cent)
        
        is_lhand_list.append(is_lhand)
        is_rhand_list.append(is_rhand)
        obj_pc_list.append(obj_pc)
        obj_pc_normal_list.append(obj_pc_normal)
        normalized_obj_pc_list.append(normalized_obj_pc)
        point_set_list.append(point_set.tolist())
        obj_cent_list.append(obj_norm_cent)
        obj_scale_list.append(obj_norm_scale)
        obj_verts_list.append(obj_verts_org)
        obj_faces_list.append(obj_mesh.faces)
    
    obj_pc_list = torch.cat(obj_pc_list)
    obj_pc_normal_list = torch.cat(obj_pc_normal_list)
    normalized_obj_pc_list = torch.cat(normalized_obj_pc_list)
    point_set_list = proc_long_torch_cuda(point_set_list)
    obj_pc_top_idx_list = proc_long_torch_cuda(obj_pc_top_idx_list)
    obj_cent_list = torch.stack(obj_cent_list)
    obj_scale_list = proc_torch_cuda(obj_scale_list)
    return (
        is_lhand_list, is_rhand_list, 
        obj_pc_list, obj_pc_normal_list, 
        normalized_obj_pc_list, point_set_list, 
        obj_cent_list, obj_scale_list, 
        obj_verts_list, obj_faces_list, 
        obj_top_idx_list, obj_pc_top_idx_list, 
    )

def get_valid_mask_bunch(is_lhand, is_rhand, max_nframes, duration):
    valid_mask_bunch_lhand = []
    valid_mask_bunch_rhand = []
    valid_mask_bunch_obj = []
    for idx in range(len(duration)):
        valid_mask_lhand, valid_mask_rhand, valid_mask_obj \
            = get_valid_mask(is_lhand[idx], is_rhand[idx], max_nframes, duration[idx])
        valid_mask_lhand = torch.BoolTensor(valid_mask_lhand).unsqueeze(0).cuda()
        valid_mask_rhand = torch.BoolTensor(valid_mask_rhand).unsqueeze(0).cuda()
        valid_mask_obj = torch.BoolTensor(valid_mask_obj).unsqueeze(0).cuda()
        valid_mask_bunch_lhand.append(valid_mask_lhand)
        valid_mask_bunch_rhand.append(valid_mask_rhand)
        valid_mask_bunch_obj.append(valid_mask_obj)
    valid_mask_bunch_lhand = torch.cat(valid_mask_bunch_lhand)
    valid_mask_bunch_rhand = torch.cat(valid_mask_bunch_rhand)
    valid_mask_bunch_obj = torch.cat(valid_mask_bunch_obj)
    return valid_mask_bunch_lhand, valid_mask_bunch_rhand, valid_mask_bunch_obj

def search_object(
    clip_model, mpnet, 
    text_feat_clip, text_feat_mpnet, 
    data_config
):
    if data_config.name == "h2o":
        obj_names = h2o_obj_name[1:]
    elif data_config.name == "grab":
        obj_names = grab_obj_name
    elif data_config.name == "arctic":
        obj_names = arctic_obj_name
    proc_obj_names = [f"A photo of {obj}" for obj in obj_names]
    obj_feat_clip = encoded_text_normalized(clip_model, proc_obj_names)
    sim_clip = torch.cosine_similarity(text_feat_clip, obj_feat_clip)
    if mpnet is not None:
        obj_feat_mpnet = mpnet.encode(proc_obj_names, convert_to_tensor=True)
        sim_mpnet = torch.cosine_similarity(text_feat_mpnet, obj_feat_mpnet)
    else:
        sim_mpnet = 0
    sim = sim_clip + sim_mpnet
    obj_idx = sim.argmax().item()
    if "obj_name_org" in data_config:
        obj_name = data_config.obj_name_org[obj_idx]
    else:
        obj_name = obj_names[obj_idx]
    print("object name:", obj_name)
    return obj_name

def search_object_shape_e(
    clip_model, mpnet, 
    text_feat_clip, text_feat_mpnet, 
    obj_names, 
):
    proc_obj_names = [f"A photo of {obj}" for obj in obj_names]
    obj_feat_clip = encoded_text_normalized(clip_model, proc_obj_names)
    sim_clip = torch.cosine_similarity(text_feat_clip, obj_feat_clip)
    if mpnet is not None:
        obj_feat_mpnet = mpnet.encode(proc_obj_names, convert_to_tensor=True)
        sim_mpnet = torch.cosine_similarity(text_feat_mpnet, obj_feat_mpnet)
    else:
        sim_mpnet = 0
    sim = sim_clip + sim_mpnet
    obj_idx = sim.argmax().item()
    obj_name = obj_names[obj_idx]
    print("object name:", obj_name)
    return obj_name

def search_hand(clip_model, mpnet, 
                text_feat_clip, text_feat_mpnet):
    hand_types = ["right hand", "left hand", "both hands"]
    proc_hand_types = ["A photo of right hand", "A photo of left hand", "A photo of both hands"]
    hand_feat_clip = encoded_text_normalized(clip_model, proc_hand_types)
    sim_clip = torch.cosine_similarity(text_feat_clip, hand_feat_clip)
    if mpnet is not None:
        hand_feat_mpnet = mpnet.encode(proc_hand_types, convert_to_tensor=True)
        sim_mpnet = torch.cosine_similarity(text_feat_mpnet, hand_feat_mpnet)
    else:
        sim_mpnet = 0
    sim = sim_clip + sim_mpnet
    hand_type_idx = sim.argmax().item()
    hand_type_selc = hand_types[hand_type_idx]
    print("hand type:", hand_type_selc)
    if hand_type_selc=="right hand":
        is_lhand = 0
        is_rhand = 1
    elif hand_type_selc=="left hand":
        is_lhand = 1
        is_rhand = 0
    elif hand_type_selc=="both hands":
        is_lhand = 1
        is_rhand = 1
    return is_lhand, is_rhand

def get_obj_file(obj_root, object_name, data_config):
    if data_config.name == "h2o":
        obj_files = glob.glob(osp.join(obj_root, object_name, "*.obj"))
        obj_idx = np.random.randint(len(obj_files))
        obj_file = obj_files[obj_idx]
    elif data_config.name == "grab":
        # object_name = "pyramidmedium"
        obj_file = osp.join(obj_root, f"{object_name}.ply")
    elif data_config.name == "arctic":
        obj_file = osp.join(obj_root, object_name, "mesh.obj")
    return obj_file

def get_shap_e_obj_file(obj_root, object_name):
    obj_files = glob.glob(osp.join(obj_root, object_name, "mesh*.obj"))
    obj_idx = np.random.randint(len(obj_files))
    obj_file = obj_files[obj_idx]
    return obj_file

def get_obj_top_file(obj_root, object_name):
    obj_top_file = osp.join(obj_root, object_name, "parts.json")
    return obj_top_file