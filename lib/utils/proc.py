import numpy as np

import torch

from lib.utils.rot import (
    axis_angle_to_rot6d, 
    rotmat_to_rot6d, 
)
from lib.utils.proc_output import (
    get_hand_joints_w_tip, 
    get_transformed_obj_pc, 
    get_NN, get_interior, 
)

def process_dist_map(
    max_nframes, init_frame, 
    cf_idx, cov_idx, chj_idx, 
    dist_value, is_hand
):
    dist_map = np.zeros((max_nframes, 1024, 21), dtype=np.float32)
    if is_hand:
        f_idx_filtered = np.where((init_frame<=cf_idx) & (cf_idx<init_frame+max_nframes))[0]
        cf_idx_selected = cf_idx[f_idx_filtered]
        cf_idx_moved = cf_idx_selected-init_frame
        cov_idx_selected = cov_idx[f_idx_filtered]
        chj_idx_selected = chj_idx[f_idx_filtered]
        dist_value_selected = dist_value[f_idx_filtered]
        dist_map[cf_idx_moved, cov_idx_selected, chj_idx_selected] = dist_value_selected
    return dist_map

def process_contact_map(
    dist_map, contact_frames
):  
    dist_map = dist_map.copy()[contact_frames]
    contact_obj_map = dist_map.sum(2) != 0
    dist_map[dist_map==0] = 1
    contact_hand_map = dist_map.min(1) != 1
    return contact_obj_map, contact_hand_map
    
def process_contact_frame_idx(lcf_idx, rcf_idx):
    if len(lcf_idx) > 0:
        min_lcf_idx = lcf_idx.min()
        max_lcf_idx = lcf_idx.max()
    else:
        min_lcf_idx = 999
        max_lcf_idx = -1

    if len(rcf_idx) > 0:
        min_rcf_idx = rcf_idx.min()
        max_rcf_idx = rcf_idx.max()
    else:
        min_rcf_idx = 999
        max_rcf_idx = -1

    min_cf_idx = min(min_lcf_idx, min_rcf_idx)
    max_cf_idx = max(max_lcf_idx, max_rcf_idx)
    return np.array([min_cf_idx, max_cf_idx])

def get_contact_map(idx, v_num, is_hand):
    contact_map = np.zeros(v_num)
    if is_hand:
        contact_map[idx] = 1
    return contact_map

def pc_normalize(pc, return_params=False):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    scale = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / scale
    if return_params:
        return pc, centroid, scale
    else:
        return pc
    
def transform_hand_to_xdata(trans, pose):
    trans_torch, pose_torch = proc_torch_frame(trans), proc_torch_frame(pose)
    nframes = pose_torch.shape[0]
    rot6d_torch = axis_angle_to_rot6d(pose_torch.reshape(-1, 3)).reshape(nframes, 16*6)
    xdata = torch.cat([trans_torch, rot6d_torch], dim=1)
    xdata = proc_numpy(xdata)
    return xdata

def transform_xdata_to_joints(xdata, hand_layer):
    xdata = proc_torch_cuda(xdata).unsqueeze(0)
    hand_joints = get_hand_joints_w_tip(xdata, hand_layer)
    hand_joints = proc_numpy(hand_joints.squeeze(0))
    return hand_joints

def transform_obj_to_xdata(obj_matrix):
    orl = proc_torch_frame(obj_matrix) # object rotation list
    obj_rotmat = orl[:, :3, :3]
    obj_trans = orl[:, :3, 3]
    nframes = obj_rotmat.shape[0]
    rot6d_torch = rotmat_to_rot6d(obj_rotmat).reshape(nframes, 6)
    xdata = torch.cat([obj_trans, rot6d_torch], dim=1)
    xdata = proc_numpy(xdata)
    return xdata

def get_contact_info(
        lhand_pose_list, lhand_beta_list, lhand_trans_list, 
        rhand_pose_list, rhand_beta_list, rhand_trans_list, 
        object_rotmat_list, lhand_layer, rhand_layer,
        sampled_obj_verts_org, mul_rv=True,
    ):
    contact_threshold = 0.02

    sampled_obj_verts_org = proc_torch_cuda(sampled_obj_verts_org)
    orl = proc_torch_frame(object_rotmat_list)
    obj_rotmat = orl[:, :3, :3]
    obj_trans = orl[:, :3, 3]

    if mul_rv:
        sampled_obj_verts = torch.einsum("tij,kj->tki", obj_rotmat, sampled_obj_verts_org) \
                            + obj_trans.unsqueeze(1)
    else:
        sampled_obj_verts = torch.einsum("tij,ki->tkj", obj_rotmat, sampled_obj_verts_org) \
                            + obj_trans.unsqueeze(1)
                            
    if len(lhand_pose_list) > 0:
        lpl, lbl, ltl = proc_torch_frame(lhand_pose_list), proc_torch_frame(lhand_beta_list), proc_torch_frame(lhand_trans_list)
        out_l = lhand_layer(lbl, lpl[..., :3], lpl[..., 3:])
        lhand_joints = out_l.joints_w_tip+ltl.unsqueeze(1)
        ldist = get_hand_object_dist(
            lhand_joints,
            sampled_obj_verts, 
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
            sampled_obj_verts, 
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

def get_hand_object_dist(hand_joints, sampled_obj_verts):
    hand_joints = proc_torch_cuda(hand_joints)
    sampled_obj_verts = proc_torch_cuda(sampled_obj_verts)

    # sampled_obj_verts = obj_verts[:, point_set]
    dist = torch.cdist(sampled_obj_verts, hand_joints)
    return dist

def get_contact_idx(dist, contact_threshold):
    # Contact frame idx, Contact object verts idx, Contact hand joints idx
    cf_idx, cov_idx, chj_idx = torch.where(dist < contact_threshold)
    return cf_idx, cov_idx, chj_idx

def get_which_hands_inter(lcf_idx, rcf_idx):
    # Contact frame idx, Contact object verts idx, Contact hand joints idx
    is_lhand = 0
    is_rhand = 0
    if len(lcf_idx) > 0:
        is_lhand = 1
    if len(rcf_idx) > 0:
        is_rhand = 1
    return is_lhand, is_rhand

def get_hand_org(hand_pose, hand_beta, hand_trans, hand_layer):
    hand_pose = proc_torch_cuda(hand_pose)
    hand_beta = proc_torch_cuda(hand_beta)
    hand_trans = proc_torch_cuda(hand_trans)
    mano_keypoints_3d = hand_layer(
        hand_beta, 
        hand_pose[:, :3], 
        hand_pose[:, 3:], 
    ).joints

    hand_origin = mano_keypoints_3d[:, 0]
    return hand_origin

def proc_torch_cuda(d):
    if not isinstance(d, torch.Tensor):
        d = torch.FloatTensor(d)
    if d.device != "cuda":
        d = d.cuda()
    return d

def proc_long_torch_cuda(d):
    if not isinstance(d, torch.Tensor):
        d = torch.LongTensor(d)
    if d.device != "cuda":
        d = d.cuda()
    return d


def proc_torch_frame(l):
    if isinstance(l, list) or isinstance(l, np.ndarray):
        l = [torch.FloatTensor(_l).unsqueeze(0) for _l in l]
        l = torch.cat(l)
        l = l.cuda()
    return l

def proc_numpy(d):
    if isinstance(d, torch.Tensor):
        if d.requires_grad:
            d = d.detach()
        if d.is_cuda:
            d = d.cpu()
        d = d.numpy()
    return d

def proc_cond_contact_estimator(obj_scale, obj_feat, enc_text, npts, use_scale):
    if use_scale:
        enc_text_expand = enc_text.unsqueeze(1)
        enc_text_expand = enc_text_expand.expand(-1, npts, -1)
        obj_scale_expand2 = obj_scale.unsqueeze(1).unsqueeze(2)
        obj_scale_expand2 = obj_scale_expand2.expand(-1, npts, -1)
        condition = torch.cat([obj_scale_expand2, obj_feat, enc_text_expand], dim=2)
    else:
        condition = torch.cat([obj_feat, enc_text_expand], dim=2)
    return condition

def proc_obj_feat_cov(
    contact_estimator, 
    obj_scale, obj_feat, 
    enc_text, npts, 
    use_scale, use_contact_feat, 
):
    obj_feat_global = obj_feat[:, 0, :1024]
    if use_contact_feat:
        condition = proc_cond_contact_estimator(obj_scale, obj_feat, enc_text, npts, use_scale)
        est_contact_map = contact_estimator.decode(condition)
        est_contact_map_plot = est_contact_map.squeeze(-1).clone()
        est_contact_map = (est_contact_map.squeeze(-1) > 0.5).long()
        obj_feat_cov = torch.cat([obj_feat_global, est_contact_map], dim=1)
    else:
        obj_feat_cov = obj_feat_global
        est_contact_map = None
        est_contact_map_plot = None
    return obj_feat_cov, est_contact_map, est_contact_map_plot

def proc_obj_feat_final(
    contact_estimator, 
    obj_scale, obj_cent, 
    obj_feat, enc_text, npts, 
    use_obj_scale_centroid, 
    use_scale, use_contact_feat, 
    return_plot=False, 
):
    obj_feat_cov, est_contact_map, est_contact_map_plot = proc_obj_feat_cov(
        contact_estimator, 
        obj_scale, obj_feat, enc_text, npts, 
        use_scale, use_contact_feat, 
    )
    if use_obj_scale_centroid:
        obj_scale_expand1 = obj_scale.unsqueeze(1)
        obj_feat_final = torch.cat([obj_feat_cov, obj_scale_expand1, obj_cent], dim=1)
    else:
        obj_feat_final = obj_feat_cov
    if return_plot:
        return obj_feat_final, est_contact_map, est_contact_map_plot
    else:
        return obj_feat_final, est_contact_map

def proc_obj_feat_final_train(cov_map, obj_scale, obj_cent, obj_feat, use_obj_scale_centroid, use_contact_feat):
    obj_feat_global = obj_feat[:, 0, :1024]
    if use_contact_feat:
        obj_feat_cov = torch.cat([obj_feat_global, cov_map], dim=1)
    else:
        obj_feat_cov = obj_feat_global

    if use_obj_scale_centroid:
        obj_scale_expand1 = obj_scale.unsqueeze(1)
        obj_feat_final = torch.cat([obj_feat_cov, obj_scale_expand1, obj_cent], dim=1)
    else:
        obj_feat_final = obj_feat_cov
    return obj_feat_final

def get_hand2obj_dist(hand_joints, obj_pc, obj_pc_normal):
    B, T = hand_joints.shape[:2]
    hand_joints = hand_joints.reshape(B*T, -1, 3)
    obj_pc = obj_pc.reshape(B*T, -1, 3)
    obj_pc_normal = obj_pc_normal.reshape(B*T, -1, 3)
    hand_nn_dist, hand_nn_idx = get_NN(hand_joints, obj_pc)
    hand_interior = get_interior(obj_pc_normal, obj_pc, hand_joints, hand_nn_idx)
    hand_nn_dist = hand_nn_dist.sqrt()
    # hand_nn_dist[hand_interior] *= -1
    hand_nn_dist = torch.abs(hand_nn_dist)
    hand_nn_idx_expand = hand_nn_idx.unsqueeze(-1).expand(*hand_nn_idx.shape, 3)
    obj_pc_contact = torch.gather(obj_pc, 1, hand_nn_idx_expand)
    hand_dist_values_xyz = (hand_joints-obj_pc_contact)**2
    hand_dist_values_xyz = hand_dist_values_xyz.reshape(B, T, -1, 3)
    hand_nn_dist = hand_nn_dist.reshape(B, T, -1)
    obj_pc_contact = obj_pc_contact.reshape(B, T, -1, 3)
    return hand_dist_values_xyz, hand_nn_dist, obj_pc_contact

def get_contact_frame(
    lhand_dist_values, rhand_dist_values, 
    valid_mask_lhand=None, 
    valid_mask_rhand=None, 
    valid_mask_obj=None, 
    threshold=0.005,
    num_keypoints=2, 
):
    if valid_mask_obj is not None:
        lhand_contact_frame_mask = ((lhand_dist_values < threshold).sum(2) >= num_keypoints)*(torch.logical_and(valid_mask_obj, valid_mask_lhand))
        rhand_contact_frame_mask = ((rhand_dist_values < threshold).sum(2) >= num_keypoints)*(torch.logical_and(valid_mask_obj, valid_mask_rhand))
    else:
        lhand_contact_frame_mask = (lhand_dist_values < threshold).sum(2) >= num_keypoints
        rhand_contact_frame_mask = (rhand_dist_values < threshold).sum(2) >= num_keypoints
    contact_frame_mask = torch.logical_or(lhand_contact_frame_mask, rhand_contact_frame_mask)
    return contact_frame_mask

def proc_refiner_input(
    pred_lhand, pred_rhand, pred_obj, 
    lhand_layer, rhand_layer, obj_pc_org, obj_pc_normal_org, 
    valid_mask_lhand, valid_mask_rhand, valid_mask_obj, 
    cov_map, dataset_name, return_psuedo_gt=False, obj_pc_top_idx=None
):
    threshold = 0.02
    bs, T = pred_obj.shape[:2]
    # Post-Processing object generation
    lhand_joints = get_hand_joints_w_tip(pred_lhand, lhand_layer)
    rhand_joints = get_hand_joints_w_tip(pred_rhand, rhand_layer)
    tf_obj_pc = get_transformed_obj_pc(pred_obj, obj_pc_org, dataset_name, obj_pc_top_idx)
    tf_obj_pc_normal = get_transformed_obj_pc(pred_obj, obj_pc_normal_org, dataset_name, obj_pc_top_idx)
    
    lhand_dist_values_xyz, lhand_dist_values, obj_pc_contact_lhand \
        = get_hand2obj_dist(lhand_joints, tf_obj_pc, tf_obj_pc_normal)
    rhand_dist_values_xyz, rhand_dist_values, obj_pc_contact_rhand \
        = get_hand2obj_dist(rhand_joints, tf_obj_pc, tf_obj_pc_normal)
    
    lhand_attn = torch.exp(-50*lhand_dist_values_xyz)
    rhand_attn = torch.exp(-50*rhand_dist_values_xyz)
    
    cov_map = cov_map.unsqueeze(1)
    cov_map = cov_map.expand(-1, T, -1)
    input_lhand = torch.cat([pred_lhand, lhand_joints.reshape(bs, T, -1), lhand_attn.reshape(bs, T, -1), tf_obj_pc.norm(dim=-1), cov_map], dim=2)
    input_rhand = torch.cat([pred_rhand, rhand_joints.reshape(bs, T, -1), rhand_attn.reshape(bs, T, -1), tf_obj_pc.norm(dim=-1), cov_map], dim=2)
    if return_psuedo_gt:
        lhand_contact_joint_mask = (lhand_dist_values < threshold)*(torch.logical_and(valid_mask_obj, valid_mask_lhand)).unsqueeze(2)
        rhand_contact_joint_mask = (rhand_dist_values < threshold)*(torch.logical_and(valid_mask_obj, valid_mask_rhand)).unsqueeze(2)
        obj_pc_contact_lhand_psuedo = obj_pc_contact_lhand.clone()
        obj_pc_contact_rhand_psuedo = obj_pc_contact_rhand.clone()
        return (
            input_lhand, input_rhand, pred_obj, 
            obj_pc_contact_lhand_psuedo, obj_pc_contact_rhand_psuedo, 
            lhand_contact_joint_mask, rhand_contact_joint_mask, 
        )
    else:
        return input_lhand, input_rhand, pred_obj
    
def filter_obj_params(pred_obj, contact_mask):
    bs, nframes = pred_obj.shape[:2]
    input_obj = pred_obj.clone()
    for B in range(bs):
        start_idx = -1
        end_idx = -1
        for T in range(nframes):
            if start_idx == -1 and not contact_mask[B, T]:
                start_idx = T
            if start_idx != -1 and contact_mask[B, T]:
                end_idx = T
                if start_idx != 0:
                    input_obj[B, start_idx:end_idx] = input_obj[B, start_idx-1:start_idx]
                else:
                    input_obj[B, start_idx:end_idx] = input_obj[B, end_idx:end_idx+1]
                    
                start_idx = -1
                end_idx = -1
            if start_idx != -1 and T == nframes-1:
                end_idx = T
                if start_idx != 0:
                    input_obj[B, start_idx:end_idx] = input_obj[B, start_idx-1:start_idx]
                else:
                    input_obj[B, start_idx:end_idx] = input_obj[B, 0:1]
                    
                start_idx = -1
                end_idx = -1
    return input_obj

def farthest_point_sample(xyz, npoint, random=False):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    if random:
        farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    else:
        farthest = 0
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def select_from_groups(data, num_groups=15):
    N = data.shape[0]
    ave_duration, remainder = divmod(N, num_groups)
    offset1 = np.multiply(np.arange(remainder), (ave_duration + 1)) + np.random.randint(ave_duration+1, size=remainder)
    offset2 = np.multiply(np.arange(remainder, num_groups), (ave_duration)) + remainder + np.random.randint(ave_duration, size=num_groups-remainder)
    final_offsets = np.append(offset1, offset2)
    selected_data = data[final_offsets]
    return selected_data