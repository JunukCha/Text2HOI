import os
import os.path as osp

import glob
import numpy as np
import trimesh
import tqdm
import pickle
import time
import json
from collections import Counter

import torch

from lib.models.mano import build_mano_aa
from lib.models.object import build_object_model
from lib.utils.frame import align_frame
from lib.utils.file import load_config, read_json
from lib.utils.rot import axis_angle_to_rotmat
from lib.utils.proc import (
    proc_torch_cuda, 
    proc_numpy, 
    get_hand_org, 
    get_contact_info, 
    transform_hand_to_xdata, 
    transform_xdata_to_joints, 
    transform_obj_to_xdata, 
    farthest_point_sample, 
)
from lib.utils.proc_grab import process_text
from constants.grab_constants import (
    grab_obj_name, object_proc, motion_proc, 
    present_participle, third_verb, passive_verb, 
)

def preprocessing_object():
    grab_config = load_config("configs/dataset/grab.yaml")
    object_pathes = glob.glob(osp.join(grab_config.obj_root, "*.ply"))

    obj_pcs = {}
    obj_pc_normals = {}
    point_sets = {}
    obj_path = {}
    for object_path in tqdm.tqdm(object_pathes):
        mesh = trimesh.load(object_path, maintain_order=True)
        verts = torch.FloatTensor(mesh.vertices).unsqueeze(0).cuda()
        normal = torch.FloatTensor(mesh.vertex_normals).unsqueeze(0).cuda()
        normal = normal / torch.norm(normal, dim=2, keepdim=True)
        point_set = farthest_point_sample(verts, 1024)
        sampled_pc = verts[0, point_set[0]].cpu().numpy()
        sampled_normal = normal[0, point_set[0]].cpu().numpy()
        object_name = object_path.split("/")[-1].replace(".ply", "")
        key = f"{object_name}"
        obj_pcs[key] = sampled_pc
        obj_pc_normals[key] = sampled_normal
        point_sets[key] = point_set[0].cpu().numpy()
        obj_path[key] = object_path.split("/")[-1]

    os.makedirs("data/grab", exist_ok=True)
    with open("data/grab/obj.pkl", "wb") as f:
        pickle.dump(
            {
                "object_name": grab_obj_name, 
                "obj_pcs": obj_pcs, 
                "obj_pc_normals": obj_pc_normals, 
                "point_sets": point_sets, 
                "obj_path": obj_path, 
            }, f)

def preprocessing_data():
    start_time = time.time()
    grab_config = load_config("configs/dataset/grab.yaml")
    data_root = grab_config.root
    data_save_path = grab_config.data_path
    object_model = build_object_model(grab_config.data_obj_pc_path)

    lhand_layer = build_mano_aa(is_rhand=False, flat_hand=grab_config.flat_hand)
    lhand_layer = lhand_layer.cuda()
    rhand_layer = build_mano_aa(is_rhand=True, flat_hand=grab_config.flat_hand)
    rhand_layer = rhand_layer.cuda()
    
    data_list = glob.glob(osp.join(data_root, "s1", "*.npz"))
    data_list += glob.glob(osp.join(data_root, "s2", "*.npz"))
    data_list += glob.glob(osp.join(data_root, "s3", "*.npz"))
    data_list += glob.glob(osp.join(data_root, "s4", "*.npz"))
    data_list += glob.glob(osp.join(data_root, "s5", "*.npz"))
    data_list += glob.glob(osp.join(data_root, "s6", "*.npz"))
    data_list += glob.glob(osp.join(data_root, "s7", "*.npz"))
    data_list += glob.glob(osp.join(data_root, "s8", "*.npz"))
    data_list += glob.glob(osp.join(data_root, "s9", "*.npz"))
    data_list += glob.glob(osp.join(data_root, "s10", "*.npz")) # comment or not
    data_list.sort()

    x_lhand_total = []
    x_rhand_total = []
    j_lhand_total = []
    j_rhand_total = []
    x_obj_total = []
    x_lhand_org_total = [] # root position of left hand
    x_rhand_org_total = [] # root position of right hand
    lcf_idx_total = []
    lcov_idx_total = []
    lchj_idx_total = []
    ldist_value_total = []
    rcf_idx_total = []
    rcov_idx_total = []
    rchj_idx_total = []
    rdist_value_total = []
    is_lhand_total = []
    is_rhand_total = []
    lhand_beta_total = []
    rhand_beta_total = []
    action_name_total = []
    obj_name_total = []
    proc_obj_name_total = []
    nframes_total = []

    for data_path in tqdm.tqdm(data_list):
        data = np.load(data_path, allow_pickle=True)
        obj_name = data["obj_name"].item()
        action_name = data["motion_intent"].item()
        if obj_name in object_proc:
            proc_obj_name = object_proc[obj_name]
        else:
            proc_obj_name = obj_name
        if action_name in motion_proc:
            action_name = motion_proc[action_name]

        contact_frame = np.nonzero(np.sum(data['contact'][()]['object'][::4], axis=1))
        start_frame = np.array([int(contact_frame[0][0])])[0]
        end_frame = np.array([int(contact_frame[0][-1])])[0]
    
        lhand_params = data["lhand"][()]["params"]
        lhand_glob_ori = lhand_params["global_orient"][::4][start_frame:end_frame]
        lhand_pose = lhand_params["fullpose"][::4][start_frame:end_frame]
        lhand_trans = lhand_params["transl"][::4][start_frame:end_frame]
        lhand_beta = np.zeros((lhand_pose.shape[0], 10))

        lhand_glob_ori = proc_torch_cuda(lhand_glob_ori)
        lhand_pose = proc_torch_cuda(lhand_pose)
        lhand_pose = torch.cat([lhand_glob_ori, lhand_pose], dim=1)
        lhand_trans = proc_torch_cuda(lhand_trans)
        lhand_beta = proc_torch_cuda(lhand_beta)

        rhand_params = data["rhand"][()]["params"]
        rhand_glob_ori = rhand_params["global_orient"][::4][start_frame:end_frame]
        rhand_pose = rhand_params["fullpose"][::4][start_frame:end_frame]
        rhand_trans = rhand_params["transl"][::4][start_frame:end_frame]
        rhand_beta = np.zeros((rhand_pose.shape[0], 10))
        
        rhand_glob_ori = proc_torch_cuda(rhand_glob_ori)
        rhand_pose = proc_torch_cuda(rhand_pose)
        rhand_pose = torch.cat([rhand_glob_ori, rhand_pose], dim=1)
        rhand_trans = proc_torch_cuda(rhand_trans)
        rhand_beta = proc_torch_cuda(rhand_beta)
                
        obj_params = data["object"][()]['params']
        obj_glob_ori = obj_params["global_orient"][::4][start_frame:end_frame]
        obj_trans = obj_params["transl"][::4][start_frame:end_frame]

        obj_glob_ori = proc_torch_cuda(obj_glob_ori)
        obj_trans = proc_torch_cuda(obj_trans)
        obj_rotmat = axis_angle_to_rotmat(obj_glob_ori)
        obj_trans = obj_trans.unsqueeze(2)
        obj_extmat = torch.cat([obj_rotmat, obj_trans], dim=2)
        
        _, obj_pc, _, _   = object_model(obj_name)
        
        lcf_idx, lcov_idx, lchj_idx, ldist_value, \
        rcf_idx, rcov_idx, rchj_idx, rdist_value, \
        is_lhand, is_rhand = get_contact_info(
            lhand_pose, lhand_beta, lhand_trans, 
            rhand_pose, rhand_beta, rhand_trans, 
            obj_extmat, lhand_layer, rhand_layer, 
            obj_pc, mul_rv=False, 
        )
        assert is_lhand != 0 or is_rhand != 0, f"{action_name} {obj_name} {start_frame} {end_frame}"
        x_lhand_org = get_hand_org(lhand_pose, lhand_beta, lhand_trans, lhand_layer)
        x_rhand_org = get_hand_org(rhand_pose, rhand_beta, rhand_trans, rhand_layer)
        
        x_lhand_org = proc_numpy(x_lhand_org)
        x_rhand_org = proc_numpy(x_rhand_org)
        lhand_beta = proc_numpy(lhand_beta)
        rhand_beta = proc_numpy(rhand_beta)
        
        x_lhand = transform_hand_to_xdata(lhand_trans, lhand_pose)
        x_rhand = transform_hand_to_xdata(rhand_trans, rhand_pose)
        j_lhand = transform_xdata_to_joints(x_lhand, lhand_layer)
        j_rhand = transform_xdata_to_joints(x_rhand, rhand_layer)
        x_obj = transform_obj_to_xdata(obj_extmat)
        x_lhand_total.append(x_lhand)
        x_rhand_total.append(x_rhand)
        j_lhand_total.append(j_lhand)
        j_rhand_total.append(j_rhand)
        x_obj_total.append(x_obj)
        x_lhand_org_total.append(x_lhand_org)
        x_rhand_org_total.append(x_rhand_org)
        lcf_idx_total.append(lcf_idx)
        lcov_idx_total.append(lcov_idx)
        lchj_idx_total.append(lchj_idx)
        ldist_value_total.append(ldist_value)
        rcf_idx_total.append(rcf_idx)
        rcov_idx_total.append(rcov_idx)
        rchj_idx_total.append(rchj_idx)
        rdist_value_total.append(rdist_value)
        is_lhand_total.append(is_lhand)
        is_rhand_total.append(is_rhand)
        lhand_beta_total.append(lhand_beta)
        rhand_beta_total.append(rhand_beta)
        action_name_total.append(action_name)
        obj_name_total.append(obj_name)
        proc_obj_name_total.append(proc_obj_name)
        nframes_total.append(len(obj_extmat))

    total_dict = {
        "x_lhand": x_lhand_total,
        "x_rhand": x_rhand_total,
        "j_lhand": j_lhand_total,
        "j_rhand": j_rhand_total, 
        "x_obj": x_obj_total,
        "lhand_beta": lhand_beta_total,
        "rhand_beta": rhand_beta_total,
        "lhand_org": x_lhand_org_total, 
        "rhand_org": x_rhand_org_total, 
    }

    final_dict = align_frame(total_dict)

    np.savez(
        data_save_path,
        **final_dict,
        lcf_idx=np.array(lcf_idx_total), 
        lcov_idx=np.array(lcov_idx_total), 
        lchj_idx=np.array(lchj_idx_total), 
        ldist_value=np.array(ldist_value_total), 
        rcf_idx=np.array(rcf_idx_total), 
        rcov_idx=np.array(rcov_idx_total), 
        rchj_idx=np.array(rchj_idx_total), 
        rdist_value=np.array(rdist_value_total), 
        is_lhand=np.array(is_lhand_total), 
        is_rhand=np.array(is_rhand_total), 
        action_name=np.array(action_name_total),
        obj_name=np.array(obj_name_total),
        proc_obj_name=np.array(proc_obj_name_total),
        nframes=np.array(nframes_total),
    )
    print("Finish:", time.time()-start_time)

def preprocessing_text():
    grab_config = load_config("configs/dataset/grab.yaml")
    data_root = grab_config.root
    text_json = grab_config.text_json
    
    data_list = glob.glob(osp.join(data_root, "s*", "*.npz"))
    data_list.sort()

    action_list = []

    for data_path in tqdm.tqdm(data_list):
        data = np.load(data_path, allow_pickle=True)
        motion = data["motion_intent"].item()
        if motion in motion_proc:
            motion = motion_proc[motion]
        obj_name = data["obj_name"].item()
        if obj_name in object_proc:
            obj_name = object_proc[obj_name]
        action_list.append(motion + "," + obj_name)
    
    action_list = np.unique(action_list)
    
    text_description = {}
    for action in action_list:
        action_v, action_o = action.split(",")[0], action.split(",")[1]
        
        text_left = f"{action_v} {action_o} with left hand.".capitalize()
        text_right = f"{action_v} {action_o} with right hand.".capitalize()
        text_both = f"{action_v} {action_o} with both hands.".capitalize()
        
        action_ving = present_participle[action_v]
        text_left1 = f"{action_ving} {action_o} with left hand.".capitalize()
        text_right1 = f"{action_ving} {action_o} with right hand.".capitalize()
        text_both1 = f"{action_ving} {action_o} with both hands.".capitalize()

        action_3rd_v = third_verb[action_v]
        text_left2 = f"Left hand {action_3rd_v} {action_o}."
        text_right2 = f"Right hand {action_3rd_v} {action_o}."
        text_both2 = f"Both hands {action_v} {action_o}."

        action_passive = passive_verb[action_v]
        text_left3 = f"{action_o} {action_passive} with left hand.".capitalize()
        text_right3 = f"{action_o} {action_passive} with right hand.".capitalize()
        text_both3 = f"{action_o} {action_passive} with both hands.".capitalize()

        text_description[text_left] = [text_left, text_left1, text_left2, text_left3]
        text_description[text_right] = [text_right, text_right1, text_right2, text_right3]
        text_description[text_both] = [text_both, text_both1, text_both2, text_both3]

    with open(text_json, "w") as f:
        json.dump(text_description, f)

def preprocessing_balance_weights():
    grab_config = load_config("configs/dataset/grab.yaml")
    data_path = grab_config.data_path
    balance_weights_path = grab_config.balance_weights_path
    t2c_json_path = grab_config.t2c_json
    
    with np.load(data_path, allow_pickle=True) as data:
        is_lhand = data["is_lhand"]
        is_rhand = data["is_rhand"]
        action_name = data["action_name"]
        proc_obj_name = data["proc_obj_name"]

    text_list = []
    for i in range(len(action_name)):
        text_key = process_text(
            action_name[i], proc_obj_name[i], 
            is_lhand[i], is_rhand[i], 
            text_descriptions=None, return_key=True
        )
        text_list.append(text_key)
    
    text_counter = Counter(text_list)
    text_dict = dict(text_counter)
    text_prob = {k:1/v for k, v in text_dict.items()}
    balance_weights = [text_prob[text] for text in text_list]
    with open(balance_weights_path, "wb") as f:
        pickle.dump(balance_weights, f)
    with open(t2c_json_path, "w") as f:
        json.dump(text_dict, f)
        
def preprocessing_text2length():
    grab_config = load_config("configs/dataset/grab.yaml")
    data_path = grab_config.data_path
    t2l_json_path = grab_config.t2l_json
    
    with np.load(data_path, allow_pickle=True) as data:
        is_lhand = data["is_lhand"]
        is_rhand = data["is_rhand"]
        action_name = data["action_name"]
        proc_obj_name = data["proc_obj_name"]
        nframes = data["nframes"]

    text_dict = {}
    for i in range(len(action_name)):
        text_key = process_text(
            action_name[i], proc_obj_name[i], 
            is_lhand[i], is_rhand[i], 
            text_descriptions=None, return_key=True
        )
        num_frames = int(nframes[i])
        if num_frames > 150:
            num_frames = 150
        if text_key not in text_dict:
            text_dict[text_key] = [num_frames]
        else:
            text_dict[text_key].append(num_frames)
    with open(t2l_json_path, "w") as f:
        json.dump(text_dict, f)

def print_text_data_num():
    grab_config = load_config("configs/dataset/grab.yaml")
    data_path = grab_config.data_path
    t2l_json_path = grab_config.t2l_json
    
    with np.load(data_path, allow_pickle=True) as data:
        action_name = data["action_name"]
    print(f"data num: {len(action_name)}")
    
    with open(t2l_json_path, "r") as f:
        text = json.load(f)
    print(f"text num: {len(text)}")