import os
import os.path as osp

import glob
import numpy as np
import tqdm
import time
import json
import trimesh
import pickle
from collections import Counter

import torch

from constants.h2o_constants import (
    h2o_obj_name, 
    action_list, 
    present_participle, 
    third_verb, 
    passive_verb, 
)
from lib.models.mano import build_mano_aa
from lib.models.object import build_object_model
from lib.utils.file import load_config
from lib.utils.frame import align_frame
from lib.utils.proc_h2o import (
    get_data_path_h2o, 
    process_hand_pose_h2o,
    process_hand_trans_h2o,
    process_text, 
)
from lib.utils.proc import (
    get_contact_info, 
    transform_hand_to_xdata,
    transform_xdata_to_joints, 
    transform_obj_to_xdata, 
    farthest_point_sample, 
)

def preprocessing_object():
    h2o_config = load_config("configs/dataset/h2o.yaml")
    objects_folder = glob.glob(osp.join(h2o_config.obj_root, "*"))
    
    obj_pcs = {}
    obj_pc_normals = {}
    point_sets = {}
    obj_path = {}
    for object_folder in tqdm.tqdm(objects_folder):
        object_paths = glob.glob(osp.join(object_folder, "*.obj"))
        for object_path in tqdm.tqdm(object_paths):
            mesh = trimesh.load(object_path, maintain_order=True)
            verts = torch.FloatTensor(mesh.vertices).unsqueeze(0).cuda()
            normal = torch.FloatTensor(mesh.vertex_normals).unsqueeze(0).cuda()
            normal = normal / torch.norm(normal, dim=2, keepdim=True)
            point_set = farthest_point_sample(verts, 1024)
            sampled_pc = verts[0, point_set[0]].cpu().numpy()
            sampled_normal = normal[0, point_set[0]].cpu().numpy()
            object_name = object_path.split("/")[-2]
            key = f"{object_name}"
            obj_pcs[key] = sampled_pc
            obj_pc_normals[key] = sampled_normal
            point_sets[key] = point_set[0].cpu().numpy()
            obj_path[key] = "/".join(object_path.split("/")[-2:])
    
    os.makedirs("data/h2o", exist_ok=True)
    with open("data/h2o/obj.pkl", "wb") as f:
        pickle.dump(
            {
                "object_name": h2o_obj_name, 
                "obj_pcs": obj_pcs, 
                "obj_pc_normals": obj_pc_normals, 
                "point_sets": point_sets, 
                "obj_path": obj_path, 
            }, f)

def preprocessing_data():
    start_time = time.time()
    h2o_config = load_config("configs/dataset/h2o.yaml")
    data_root = h2o_config.root
    data_save_path = h2o_config.data_path
        
    object_model = build_object_model(h2o_config.data_obj_pc_path)

    lhand_layer = build_mano_aa(is_rhand=False, flat_hand=h2o_config.flat_hand)
    lhand_layer = lhand_layer.cuda()
    rhand_layer = build_mano_aa(is_rhand=True, flat_hand=h2o_config.flat_hand)
    rhand_layer = rhand_layer.cuda()

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
    object_idx_total = []
    action_total = []
    action_name_total = []
    nframes_total = []
    subject_total = []
    background_total = []

    no_inter_action_name = []
    '''
        subject, background, object class, cam, 
    '''
    data_paths = glob.glob(osp.join(data_root, "subject1", "*", "*", "cam*"))
    data_paths += glob.glob(osp.join(data_root, "subject2", "*", "*", "cam*"))
    data_paths += glob.glob(osp.join(data_root, "subject3", "*", "*", "cam*")) # comment or not
    data_paths.sort()

    for data_idx, data_path in enumerate(data_paths):
        hand_pose_manos, obj_pose_rts, cam_poses, action_labels \
            = get_data_path_h2o(data_path)

        prev_action = 0

        lhand_pose_list = []
        lhand_beta_list = []
        lhand_trans_list = []
        x_lhand_org_list = []
        rhand_pose_list = []
        rhand_beta_list = []
        rhand_trans_list = []
        x_rhand_org_list = []
        object_rotmat_list = []

        for hand_pose_mano, obj_pose_rt, cam_pose, action_label in \
            tqdm.tqdm(zip(hand_pose_manos, obj_pose_rts, cam_poses, action_labels), desc=f"{data_idx}/{len(data_paths)}", total=len(obj_pose_rts)):

            hand_pose_mano_data = np.loadtxt(hand_pose_mano)
            obj_pose_rt_data = np.loadtxt(obj_pose_rt)
            extrinsic_matrix = np.loadtxt(cam_pose).reshape(4, 4)
            action = int(np.loadtxt(action_label))
            
            if action != prev_action and prev_action != 0:
                if len(object_rotmat_list) > 20:
                    _, obj_pc, _, _ = object_model(int(obj_idx))
                    lcf_idx, lcov_idx, lchj_idx, ldist_value, \
                    rcf_idx, rcov_idx, rchj_idx, rdist_value, \
                    is_lhand, is_rhand = get_contact_info(
                        lhand_pose_list, lhand_beta_list, lhand_trans_list, 
                        rhand_pose_list, rhand_beta_list, rhand_trans_list, 
                        object_rotmat_list, lhand_layer, rhand_layer,
                        obj_pc, 
                    )
                    if is_lhand == 0 and is_rhand == 0:
                        action_name = action_list[prev_action]
                        no_inter_action_name.append(action_name)
                        lhand_pose_list = []
                        lhand_beta_list = []
                        lhand_trans_list = []
                        x_lhand_org_list = []
                        rhand_pose_list = []
                        rhand_beta_list = []
                        rhand_trans_list = []
                        x_rhand_org_list = []
                        object_rotmat_list = []
                        prev_action = action
                        continue
                    x_lhand = transform_hand_to_xdata(lhand_trans_list, lhand_pose_list)
                    x_rhand = transform_hand_to_xdata(rhand_trans_list, rhand_pose_list)
                    j_lhand = transform_xdata_to_joints(x_lhand, lhand_layer)
                    j_rhand = transform_xdata_to_joints(x_rhand, rhand_layer)
                    x_obj = transform_obj_to_xdata(object_rotmat_list)
                    x_lhand_total.append(x_lhand)
                    x_rhand_total.append(x_rhand)
                    j_lhand_total.append(j_lhand)
                    j_rhand_total.append(j_rhand)
                    x_obj_total.append(x_obj)
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
                    object_idx_total.append(int(obj_idx))
                    action_name = action_list[prev_action]
                    action_total.append(prev_action)
                    action_name_total.append(action_name)
                    nframes_total.append(len(object_rotmat_list))
                    subject_total.append(data_path.split("/")[3])
                    background_total.append(data_path.split("/")[4])

                lhand_pose_list = []
                lhand_beta_list = []
                lhand_trans_list = []
                x_lhand_org_list = []
                rhand_pose_list = []
                rhand_beta_list = []
                rhand_trans_list = []
                x_rhand_org_list = []
                object_rotmat_list = []
            
            if action == 0:
                prev_action = action
                continue

            lhand_trans = hand_pose_mano_data[1:4]
            lhand_pose = hand_pose_mano_data[4:52]
            lhand_beta = hand_pose_mano_data[52:62]

            left_rotvec = process_hand_pose_h2o(lhand_pose, lhand_trans, extrinsic_matrix)
            lhand_pose[:3] = left_rotvec

            new_left_trans, lhand_origin = process_hand_trans_h2o(lhand_pose, lhand_beta, lhand_trans, extrinsic_matrix, lhand_layer)
            lhand_trans_list.append(new_left_trans)
            lhand_pose_list.append(lhand_pose)
            lhand_beta_list.append(lhand_beta)
            x_lhand_org_list.append(lhand_origin)

            rhand_trans = hand_pose_mano_data[63:66]
            rhand_pose = hand_pose_mano_data[66:114]
            rhand_beta = hand_pose_mano_data[114:124]

            right_rotvec = process_hand_pose_h2o(rhand_pose, rhand_trans, extrinsic_matrix)
            rhand_pose[:3] = right_rotvec

            new_right_trans, rhand_origin = process_hand_trans_h2o(rhand_pose, rhand_beta, rhand_trans, extrinsic_matrix, rhand_layer)
            rhand_trans_list.append(new_right_trans)
            rhand_pose_list.append(rhand_pose)
            rhand_beta_list.append(rhand_beta)
            x_rhand_org_list.append(rhand_origin)

            obj_idx = obj_pose_rt_data[0]
            object_ext = obj_pose_rt_data[1:].reshape(4, 4)

            new_object_matrix = np.dot(extrinsic_matrix, object_ext)
            object_rotmat_list.append(new_object_matrix)

            prev_action = action

        if len(object_rotmat_list) > 20:
            _, obj_pc, _, _ = object_model(int(obj_idx))
            lcf_idx, lcov_idx, lchj_idx, ldist_value, \
            rcf_idx, rcov_idx, rchj_idx, rdist_value, \
            is_lhand, is_rhand = get_contact_info(
                lhand_pose_list, lhand_beta_list, lhand_trans_list, 
                rhand_pose_list, rhand_beta_list, rhand_trans_list, 
                object_rotmat_list, lhand_layer, rhand_layer,
                obj_pc
            )
            if is_lhand == 0 and is_rhand == 0:
                action_name = action_list[prev_action]
                no_inter_action_name.append(action_name)
                continue
            x_lhand = transform_hand_to_xdata(lhand_trans_list, lhand_pose_list)
            x_rhand = transform_hand_to_xdata(rhand_trans_list, rhand_pose_list)
            j_lhand = transform_xdata_to_joints(x_lhand, lhand_layer)
            j_rhand = transform_xdata_to_joints(x_rhand, rhand_layer)
            x_obj = transform_obj_to_xdata(object_rotmat_list)
            x_lhand_total.append(x_lhand)
            x_rhand_total.append(x_rhand)
            j_lhand_total.append(j_lhand)
            j_rhand_total.append(j_rhand)
            x_obj_total.append(x_obj)
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
            object_idx_total.append(int(obj_idx))
            action_name = action_list[prev_action]
            action_total.append(prev_action)
            action_name_total.append(action_name)
            nframes_total.append(len(object_rotmat_list))
            subject_total.append(data_path.split("/")[3])
            background_total.append(data_path.split("/")[4])
        
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
        object_idx=np.array(object_idx_total),
        action=np.array(action_total),
        action_name=np.array(action_name_total),
        nframes=np.array(nframes_total),
        subject=np.array(subject_total),
        background=np.array(background_total),
    )
    print("Finish:", time.time()-start_time)
    
def preprocessing_text():
    h2o_config = load_config("configs/dataset/h2o.yaml")
    text_json = h2o_config.text_json
    
    text_description = {}
    for action in action_list[1:]:
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

    with open(text_json, "w") as f:
        json.dump(text_description, f)

def preprocessing_balance_weights():
    h2o_config = load_config("configs/dataset/h2o.yaml")
    data_path = h2o_config.data_path
    balance_weights_path = h2o_config.balance_weights_path
    t2c_json_path = h2o_config.t2c_json

    with np.load(data_path, allow_pickle=True) as data:
        is_lhand = data["is_lhand"]
        is_rhand = data["is_rhand"]
        action_name = data["action_name"]

    text_list = []
    for i in range(len(action_name)):
        text_key = process_text(
            action_name[i], is_lhand[i], is_rhand[i], 
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
    h2o_config = load_config("configs/dataset/h2o.yaml")
    data_path = h2o_config.data_path
    t2l_json = h2o_config.t2l_json
    
    with np.load(data_path, allow_pickle=True) as data:
        is_lhand = data["is_lhand"]
        is_rhand = data["is_rhand"]
        action_name = data["action_name"]
        nframes = data["nframes"]

    text_dict = {}
    for i in range(len(action_name)):
        text_key = process_text(
            action_name[i], is_lhand[i], is_rhand[i], 
            text_descriptions=None, return_key=True
        )
        num_frames = int(nframes[i])
        if num_frames > 150:
            num_frames = 150
        if text_key not in text_dict:
            text_dict[text_key] = [num_frames]
        else:
            text_dict[text_key].append(num_frames)
    with open(t2l_json, "w") as f:
        json.dump(text_dict, f)

def print_text_data_num():
    h2o_config = load_config("configs/dataset/h2o.yaml")
    data_path = h2o_config.data_path
    t2l_json_path = h2o_config.t2l_json
    
    with np.load(data_path, allow_pickle=True) as data:
        action_name = data["action_name"]
    print(f"data num: {len(action_name)}")
    
    with open(t2l_json_path, "r") as f:
        text = json.load(f)
    print(f"text num: {len(text)}")