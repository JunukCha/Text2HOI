import os
import os.path as osp

import numpy as np
import time
import tqdm
import glob
import json
import trimesh
import pickle
from collections import Counter

import torch
import tqdm

from lib.models.mano import build_mano_aa
from lib.models.object import build_object_model
from lib.models.object_arctic import ObjectTensors
from lib.utils.data import processe_params
from lib.utils.file import load_config, read_json
from lib.utils.rot import axis_angle_to_rotmat
from lib.utils.frame import align_frame
from lib.utils.proc import (
    proc_torch_cuda, 
    proc_numpy, 
    transform_hand_to_xdata,
    transform_xdata_to_joints,  
    transform_obj_to_xdata, 
    farthest_point_sample, 
)
from lib.utils.proc_arctic import (
    process_hand_keypoints, 
    process_object, 
    get_contact_info_arctic, 
    process_text, 
)
from constants.arctic_constants import (
    arctic_obj_name, 
    subj_list, 
    not_save_list, 
    action_list, 
    present_participle, 
    third_verb, 
    passive_verb, 
)

def preprocessing_object():
    arctic_config = load_config("configs/dataset/arctic.yaml")
    objects_folder = glob.glob(osp.join(arctic_config.obj_root, "*"))
    
    obj_pcs = {}
    obj_pc_normals = {}
    obj_pc_top = {}
    point_sets = {}
    obj_path = {}
    for object_folder in tqdm.tqdm(objects_folder):
        object_path = glob.glob(osp.join(object_folder, "mesh.obj"))[0]
        parts_path = glob.glob(osp.join(object_folder, "parts.json"))[0]
        parts = read_json(parts_path)
        parts = np.array(parts) # 0: top, 1: bottom
        mesh = trimesh.load(object_path, maintain_order=True)
        verts = torch.FloatTensor(mesh.vertices).unsqueeze(0).cuda()
        normal = torch.FloatTensor(mesh.vertex_normals).unsqueeze(0).cuda()
        normal = normal / torch.norm(normal, dim=2, keepdim=True)
        point_set = farthest_point_sample(verts, 1024)
        sampled_pc = verts[0, point_set[0]].cpu().numpy()/1000
        sampled_normal = normal[0, point_set[0]].cpu().numpy()
        object_name = object_path.split("/")[-2]
        key = f"{object_name}"
        obj_pcs[key] = sampled_pc
        obj_pc_normals[key] = sampled_normal
        obj_pc_top[key] = 1-parts[point_set[0].cpu().numpy()] # 0: bottom, 1: top
        point_sets[key] = point_set[0].cpu().numpy()
        obj_path[key] = "/".join(object_path.split("/")[-2:])

    os.makedirs("data/arctic", exist_ok=True)
    with open("data/arctic/obj.pkl", "wb") as f:
        pickle.dump(
            {
                "object_name": arctic_obj_name, 
                "obj_pcs": obj_pcs, 
                "obj_pc_normals": obj_pc_normals, 
                "obj_pc_top": obj_pc_top, 
                "point_sets": point_sets, 
                "obj_path": obj_path, 
            }, f)

def preprocessing_data():
    start_time = time.time()
    arctic_config = load_config("configs/dataset/arctic.yaml")
    data_root = arctic_config.root
    data_save_path = arctic_config.data_path
    text_root = arctic_config.text_root
    object_model = build_object_model(arctic_config.data_obj_pc_path)

    lhand_layer = build_mano_aa(is_rhand=False, flat_hand=arctic_config.flat_hand)
    lhand_layer = lhand_layer.cuda()
    rhand_layer = build_mano_aa(is_rhand=True, flat_hand=arctic_config.flat_hand)
    rhand_layer = rhand_layer.cuda()
    
    arctic_object_model = ObjectTensors("cuda")

    x_lhand_total = []
    x_rhand_total = []
    j_lhand_total = []
    j_rhand_total = []
    x_obj_total = []
    x_obj_angle_total = []
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
    object_name_total = []
    action_name_total = []
    nframes_total = []
    
    check_text_list = []
    for subj in subj_list:
        for object_name in tqdm.tqdm(arctic_obj_name, desc=f"{subj}"):
            hand_data_pathes = glob.glob(osp.join(data_root, subj, f"{object_name}*.mano.npy"))
            hand_data_pathes.sort()
            obj_data_pathes = glob.glob(osp.join(data_root, subj, f"{object_name}*.object.npy"))
            obj_data_pathes.sort()
            point_set, _, _, _, _   = object_model(object_name)
            for hand_data_path, obj_data_path in zip(hand_data_pathes, obj_data_pathes):
                hand_data, object_data = processe_params(hand_data_path, obj_data_path)
                lhand_poses = hand_data["left.pose"]
                lhand_betas = hand_data["left.shape"] # np.zeros((lhand_poses.shape[0], 10))
                lhand_trans = hand_data["left.trans"]
                lhand_joints = process_hand_keypoints(lhand_poses, lhand_betas, lhand_trans, lhand_layer)
                lhand_joints = proc_numpy(lhand_joints)
                rhand_poses = hand_data["right.pose"]
                rhand_betas = hand_data["right.shape"] # np.zeros((rhand_poses.shape[0], 10))
                rhand_trans = hand_data["right.trans"]
                rhand_joints = process_hand_keypoints(rhand_poses, rhand_betas, rhand_trans, rhand_layer)
                rhand_joints = proc_numpy(rhand_joints)
                obj_angle = object_data["object.angle"]
                obj_rots = object_data["object.global_rot"]
                obj_trans = object_data["object.trans"]
                obj_dict = process_object(obj_angle, obj_rots, obj_trans/1000, object_name, arctic_object_model)
                obj_verts = obj_dict["v"]
                
                data_basename = obj_data_path.split("/")[-1].replace(".object.npy", "")
                text_file = osp.join(text_root, subj, data_basename, "description.txt")
                with open(text_file, "r") as f:
                    text_data = f.readlines()
                duplicate_text = []
                for text_info in text_data:
                    if text_info == "\n":
                        continue
                    start_end_frame, action_name, hand_type, _ = text_info.split(" ")
                    duplicate_text.append(f"{start_end_frame} {hand_type}")
                    if len(duplicate_text) != len(np.unique(duplicate_text)):
                        raise Exception("duplicate")
                    
                    action_name = action_name.replace(",", "")
                    check_text = f"{action_name} {object_name}"
                    check_text_list.append(check_text)
                    if check_text in not_save_list:
                        continue
                    start_frame, end_frame = start_end_frame.split("-")
                    start_frame = int(start_frame)-1
                    end_frame = int(end_frame)+1-1
                    if end_frame-start_frame < 20:
                        continue
                    lhand_pose_list = lhand_poses[start_frame:end_frame]
                    lhand_beta_list = lhand_betas[start_frame:end_frame]
                    lhand_trans_list = lhand_trans[start_frame:end_frame]
                    x_lhand_org_list = lhand_joints[start_frame:end_frame, 0]
                    rhand_pose_list = rhand_poses[start_frame:end_frame]
                    rhand_beta_list = rhand_betas[start_frame:end_frame]
                    rhand_trans_list = rhand_trans[start_frame:end_frame]
                    x_rhand_org_list = rhand_joints[start_frame:end_frame, 0]
                    obj_angle_list = obj_angle[start_frame:end_frame]
                    obj_rot_list = obj_rots[start_frame:end_frame]
                    obj_trans_list = obj_trans[start_frame:end_frame]
                    
                    obj_rot_list = proc_torch_cuda(obj_rot_list)
                    obj_trans_list = proc_torch_cuda(obj_trans_list)/1000 # mm -> m
                    obj_rotmat_list = axis_angle_to_rotmat(obj_rot_list)
                    obj_trans_list = obj_trans_list.unsqueeze(2)
                    obj_extmat_list = torch.cat([obj_rotmat_list, obj_trans_list], dim=2)
                    
                    lcf_idx, lcov_idx, lchj_idx, ldist_value, \
                    rcf_idx, rcov_idx, rchj_idx, rdist_value, \
                    is_lhand, is_rhand = get_contact_info_arctic(
                        lhand_pose_list, lhand_beta_list, lhand_trans_list, 
                        rhand_pose_list, rhand_beta_list, rhand_trans_list, 
                        obj_verts[start_frame:end_frame, point_set], 
                        lhand_layer, rhand_layer, 
                    )
                    assert is_lhand != 0 or is_rhand != 0
                    
                    text_vis = f"{subj, data_basename, start_frame, end_frame}"
                    if hand_type == "both":
                        assert is_lhand == 1 and is_rhand == 1, text_vis
                    elif hand_type == "left":
                        assert is_lhand == 1, text_vis
                        is_rhand = 0
                    elif hand_type == "right":
                        assert is_rhand == 1, text_vis
                        is_lhand = 0
                    else:
                        raise Exception("Hand type not supported")
                    
                    x_lhand = transform_hand_to_xdata(lhand_trans_list, lhand_pose_list)
                    x_rhand = transform_hand_to_xdata(rhand_trans_list, rhand_pose_list)
                    j_lhand = transform_xdata_to_joints(x_lhand, lhand_layer)
                    j_rhand = transform_xdata_to_joints(x_rhand, rhand_layer)
                    j_lhand_total.append(j_lhand)
                    j_rhand_total.append(j_rhand)
                    x_obj = transform_obj_to_xdata(obj_extmat_list)
                    x_obj_angle = obj_angle_list
                    x_lhand_total.append(x_lhand)
                    x_rhand_total.append(x_rhand)
                    x_obj_total.append(x_obj)
                    x_obj_angle_total.append(x_obj_angle)
                    x_lhand_org_total.append(x_lhand_org_list)
                    x_rhand_org_total.append(x_rhand_org_list)
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
                    lhand_beta_total.append(lhand_beta_list)
                    rhand_beta_total.append(rhand_beta_list)
                    object_name_total.append(object_name)
                    action_name_total.append(action_name)
                    nframes_total.append(len(obj_extmat_list))

    total_dict = {
        "x_lhand": x_lhand_total,
        "x_rhand": x_rhand_total,
        "j_lhand": j_lhand_total,
        "j_rhand": j_rhand_total, 
        "x_obj": x_obj_total,
        "x_obj_angle": x_obj_angle_total,
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
        object_name=np.array(object_name_total),
        action_name=np.array(action_name_total),
        nframes=np.array(nframes_total),
    )
    print("Finish:", time.time()-start_time)
    print("Length of data:", len(x_obj_total))
    
def preprocessing_text():
    text_description = {}
    for action in action_list:
        text_left = f"{action} with left hand.".capitalize()
        text_right = f"{action} with right hand.".capitalize()
        text_both = f"{action} with both hands.".capitalize()

        action_v, action_o = " ".join(action.split(" ")[:-1]), action.split(" ")[-1]
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

    with open("data/arctic/text.json", "w") as f:
        json.dump(text_description, f)
        
def preprocessing_balance_weights(is_action=False):
    arctic_config = load_config("configs/dataset/arctic.yaml")
    if is_action:
        data_path = arctic_config.action_train_data_path
        balance_weights_path = arctic_config.action_balance_weights_path
    else:
        data_path = arctic_config.data_path
        balance_weights_path = arctic_config.balance_weights_path
    t2c_json_path = arctic_config.t2c_json
    
    with np.load(data_path, allow_pickle=True) as data:
        is_lhand = data["is_lhand"]
        is_rhand = data["is_rhand"]
        action_name = data["action_name"]
        object_name = data["object_name"]

    text_list = []
    for i in range(len(action_name)):
        text_key = process_text(
            action_name[i], 
            object_name[i], 
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
    with np.load("data/arctic/data.npz", allow_pickle=True) as data:
        is_lhand = data["is_lhand"]
        is_rhand = data["is_rhand"]
        action_name = data["action_name"]
        object_name = data["object_name"]
        nframes = data["nframes"]

    text_dict = {}
    for i in range(len(action_name)):
        text_key = process_text(
            action_name[i], 
            object_name[i], 
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
    with open("data/arctic/text_length.json", "w") as f:
        json.dump(text_dict, f)

def print_text_data_num():
    arctic_config = load_config("configs/dataset/arctic.yaml")
    data_path = arctic_config.data_path
    t2l_json_path = arctic_config.t2l_json
    
    with np.load(data_path, allow_pickle=True) as data:
        action_name = data["action_name"]
    print(f"data num: {len(action_name)}")
    
    with open(t2l_json_path, "r") as f:
        text = json.load(f)
    print(f"text num: {len(text)}")