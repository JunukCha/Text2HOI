import numpy as np

import torch

from lib.utils.proc import (
    proc_numpy, 
    proc_torch_cuda, 
    proc_torch_frame, 
    get_contact_idx, 
    get_hand_object_dist, 
    get_which_hands_inter, 
)

def process_text(
    action_name, 
    object_name, 
    is_lhand, is_rhand, 
    text_descriptions, return_key=False, 
):
    if is_lhand and is_rhand:
        text = f"{action_name} {object_name} with both hands."
    elif is_lhand:
        text = f"{action_name} {object_name} with left hand."
    elif is_rhand:
        text = f"{action_name} {object_name} with right hand."
    text_key = text.capitalize()
    if return_key:
        return text_key
    else:
        text_description = text_descriptions[text_key]
        text = np.random.choice(text_description)
        return text

def process_hand_keypoints(hand_poses, hand_betas, hand_trans, hand_layer):
    hand_betas_th = proc_torch_cuda(hand_betas)
    hand_poses_th = proc_torch_cuda(hand_poses)
    hand_trans_th = proc_torch_cuda(hand_trans)
    hand_joints_w_tip = hand_layer(
        hand_betas_th,
        hand_poses_th[:, :3],
        hand_poses_th[:, 3:],
    ).joints_w_tip

    hand_joints_w_tip = hand_joints_w_tip + hand_trans_th.unsqueeze(1)
    return hand_joints_w_tip

def process_object(obj_angle, obj_rot, obj_trans, obj_name, object_model):
    obj_name = [obj_name]*obj_angle.shape[0]
    obj_angle_th = proc_torch_cuda(obj_angle)
    obj_rot_th = proc_torch_cuda(obj_rot)
    obj_trans_th = proc_torch_cuda(obj_trans)
    obj_dict = object_model(obj_angle_th, obj_rot_th, obj_trans_th, obj_name)
    return obj_dict

def get_contact_info_arctic(
        lhand_pose_list, lhand_beta_list, lhand_trans_list, 
        rhand_pose_list, rhand_beta_list, rhand_trans_list, 
        sampled_obj_pc_tf, lhand_layer, rhand_layer,
    ):
    contact_threshold = 0.02

    sampled_obj_pc_tf = proc_torch_cuda(sampled_obj_pc_tf)

    if len(lhand_pose_list) > 0:
        lpl, lbl, ltl = proc_torch_frame(lhand_pose_list), proc_torch_frame(lhand_beta_list), proc_torch_frame(lhand_trans_list)
        out_l = lhand_layer(lbl, lpl[..., :3], lpl[..., 3:])
        lhand_joints = out_l.joints_w_tip+ltl.unsqueeze(1)
        ldist = get_hand_object_dist(
            lhand_joints,
            sampled_obj_pc_tf, 
        )
        lcf_idx, lcov_idx, lchj_idx = get_contact_idx(ldist, contact_threshold)
        ldist_value = ldist[lcf_idx, lcov_idx, lchj_idx]
        
    else:
        lcf_idx, lcov_idx, lchj_idx = np.array([]), np.array([]), np.array([])
        ldist_value = np.array([])

    if len(rhand_pose_list) > 0:
        rpl, rbl, rtl = proc_torch_frame(rhand_pose_list), proc_torch_frame(rhand_beta_list), proc_torch_frame(rhand_trans_list)
        out_r = rhand_layer(rbl, rpl[..., :3], rpl[..., 3:])
        rhand_joints = out_r.joints_w_tip+rtl.unsqueeze(1)
        rdist = get_hand_object_dist(
            rhand_joints,
            sampled_obj_pc_tf, 
        )
        rcf_idx, rcov_idx, rchj_idx = get_contact_idx(rdist, contact_threshold)
        rdist_value = rdist[rcf_idx, rcov_idx, rchj_idx]
        
    else:
        rcf_idx, rcov_idx, rchj_idx = np.array([]), np.array([]), np.array([])
        rdist_value = np.array([])

    is_lhand, is_rhand = get_which_hands_inter(lcf_idx, rcf_idx)
    
    lcf_idx = proc_numpy(lcf_idx)
    lcov_idx = proc_numpy(lcov_idx)
    lchj_idx = proc_numpy(lchj_idx)
    ldist_value = proc_numpy(ldist_value)

    rcf_idx = proc_numpy(rcf_idx)
    rcov_idx = proc_numpy(rcov_idx)
    rchj_idx = proc_numpy(rchj_idx)
    rdist_value = proc_numpy(rdist_value)
    
    return (lcf_idx, lcov_idx, lchj_idx, ldist_value, 
            rcf_idx, rcov_idx, rchj_idx, rdist_value, 
            is_lhand, is_rhand)

def get_contact_frame_idx(obj_v, hand_j, contact_threshold=0.01):
    dist = torch.cdist(obj_v, hand_j)
    cf_idx, _, _ = torch.where(dist < contact_threshold)
    cf_uni_v, cf_cnt = torch.unique(cf_idx, return_counts=True)
    cf_cnt_mean = cf_cnt.float().mean()
    cf_filter = torch.where(cf_cnt > cf_cnt_mean)[0]
    cf_uni_v_filt = cf_uni_v[cf_filter]
    return cf_uni_v_filt

def clustering_frame_idx(cf_idx):
    clusters = []
    cluster = []
    for i in range(len(cf_idx)):
        cluster.append(cf_idx[i].item())
        if len(cluster) > 1 and cf_idx[i] != cf_idx[i-1]+1:
            if len(cluster) >= 20:
                clusters.append(cluster)
            cluster = []
    clusters
        