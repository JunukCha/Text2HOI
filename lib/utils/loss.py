import numpy as np

import torch
import torch.nn.functional as F

from lib.utils.frame import sample_with_window_size
from lib.utils.rot import rot6d_to_rotmat
from lib.utils.proc_output import (
    get_pytorch3d_meshes, 
    get_hand_joints_w_tip, 
    get_transformed_obj_pc, 
    get_NN, get_interior
)

def l2_loss_unit(pred, targ, mask=None, weight=None):
    l2_loss = F.mse_loss(pred, targ, reduction='none')
    if mask is not None:
        filtered_loss = get_filtered_loss_valid_mask(l2_loss, mask, weight)
    else:
        filtered_loss = l2_loss.mean()
    return filtered_loss

def get_l2_loss(
        pred_lhand=None, pred_rhand=None, pred_obj=None, 
        targ_lhand=None, targ_rhand=None, targ_obj=None, 
        mask_lhand=None, mask_rhand=None, mask_obj=None, 
        weight=None, 
    ):
    total_loss = 0
    if targ_lhand is not None:
        lhand_loss = l2_loss_unit(pred_lhand, targ_lhand, mask_lhand, weight)
        total_loss += lhand_loss
    if targ_rhand is not None:
        rhand_loss = l2_loss_unit(pred_rhand, targ_rhand, mask_rhand, weight)
        total_loss += rhand_loss
    if targ_obj is not None:
        obj_loss = l2_loss_unit(pred_obj, targ_obj, mask_obj, weight)
        total_loss += obj_loss
    return total_loss

def get_joint_contact_loss(
    pred_lhand, pred_rhand, 
    lhand_obj_cont_v, rhand_obj_cont_v,
    lhand_layer, rhand_layer, 
    mask_lhand=None, mask_rhand=None, 
    weight=None, loss_type="l2"
):
    
    lhand_joints = get_hand_joints_w_tip(pred_lhand, lhand_layer)
    rhand_joints = get_hand_joints_w_tip(pred_rhand, rhand_layer)
    cont_lhand_loss = joint_contact_loss_unit(lhand_joints, lhand_obj_cont_v, mask_lhand, weight=weight, loss_type=loss_type)
    cont_rhand_loss = joint_contact_loss_unit(rhand_joints, rhand_obj_cont_v, mask_rhand, weight=weight, loss_type=loss_type)
    return cont_lhand_loss + cont_rhand_loss
    
def joint_contact_loss_unit(pred, targ, mask=None, weight=None, loss_type="l2"):
    if loss_type == "l2":
        loss = F.mse_loss(pred, targ, reduction='none')
    elif loss_type == "l1":
        loss = F.l1_loss(pred, targ, reduction='none')
    if mask is not None:
        filtered_loss = get_filtered_joint_loss_valid_mask(loss, mask, weight)
    else:
        filtered_loss = loss.mean()
    return filtered_loss

def get_smth_loss(
        model, window_size, window_step, 
        pred_lhand, pred_rhand, pred_obj, 
        targ_lhand, targ_rhand, targ_obj, 
        mask_lhand, mask_rhand, mask_obj, 
        weight, 
    ):
    pred_X0_lhand_sampled, target_lhand_sampled = sample_with_window_size(pred_lhand, targ_lhand, mask_lhand, window_size, window_step)
    pred_X0_rhand_sampled, target_rhand_sampled = sample_with_window_size(pred_rhand, targ_rhand, mask_rhand, window_size, window_step)
    pred_X0_obj_sampled, target_obj_sampled = sample_with_window_size(pred_obj, targ_obj, mask_obj, window_size, window_step)

    pred_X0_lhand_smoothed = model.hand_smoother(pred_X0_lhand_sampled.detach())
    pred_X0_rhand_smoothed = model.hand_smoother(pred_X0_rhand_sampled.detach())
    pred_X0_obj_smoothed = model.obj_smoother(pred_X0_obj_sampled.detach())

    # Smth pos loss
    smth_pos_loss_lhand = smth_pos_loss_unit(pred_X0_lhand_smoothed, target_lhand_sampled, weight)
    smth_pos_loss_rhand = smth_pos_loss_unit(pred_X0_rhand_smoothed, target_rhand_sampled, weight)
    smth_pos_loss_obj = smth_pos_loss_unit(pred_X0_obj_smoothed, target_obj_sampled, weight)
    smth_pos_loss = smth_pos_loss_lhand+smth_pos_loss_rhand+smth_pos_loss_obj

    # Smth accel loss
    smth_accel_loss_lhand = smth_accel_loss_unit(pred_X0_lhand_smoothed, target_lhand_sampled, weight)
    smth_accel_loss_rhand = smth_accel_loss_unit(pred_X0_rhand_smoothed, target_rhand_sampled, weight)
    smth_accel_loss_obj = smth_accel_loss_unit(pred_X0_obj_smoothed, target_obj_sampled, weight)
    smth_accel_loss = smth_accel_loss_lhand+smth_accel_loss_rhand+smth_accel_loss_obj
    return smth_pos_loss, smth_accel_loss

def smth_pos_loss_unit(pred, targ, weight):
    smth_pos_loss = weight*F.l1_loss(pred, targ, reduction='none').mean([1, 2])
    smth_pos_loss = smth_pos_loss.mean()
    return smth_pos_loss

def smth_accel_loss_unit(pred, targ, weight):
    pred_accel = pred[:, 2:]-2*pred[:, 1:-1]+pred[:, :-2]
    targ_accel = targ[:, 2:]-2*targ[:, 1:-1]+targ[:, :-2]
    smth_acc_loss = weight*F.l1_loss(pred_accel, targ_accel, reduction='none').mean([1, 2])
    smth_acc_loss = smth_acc_loss.mean()
    return smth_acc_loss

def get_distance_map_loss(
        pred_ldist, pred_rdist, 
        targ_ldist, targ_rdist, 
        weight=None, 
    ):
    
    ldist_loss = F.mse_loss(pred_ldist, targ_ldist, reduction="none")
    rdist_loss = F.mse_loss(pred_rdist, targ_rdist, reduction="none")
    valid_map_ldist = targ_ldist > 0
    valid_map_rdist = targ_rdist > 0
    filtered_ldist_loss = get_filtered_loss_valid_map(ldist_loss, valid_map_ldist, weight)
    filtered_rdist_loss = get_filtered_loss_valid_map(rdist_loss, valid_map_rdist, weight)
    return filtered_ldist_loss+filtered_rdist_loss

def get_relative_orientation_loss(
        pred_lhand, pred_rhand, pred_obj, 
        targ_lhand, targ_rhand, targ_obj, 
        mask_lhand, mask_rhand, 
        weight=None
    ):
    pred_ro_lhand = get_ro(pred_lhand, pred_obj, mask_lhand)
    pred_ro_rhand = get_ro(pred_rhand, pred_obj, mask_rhand)
    targ_ro_lhand = get_ro(targ_lhand, targ_obj, mask_lhand)
    targ_ro_rhand = get_ro(targ_rhand, targ_obj, mask_rhand)
    if weight is not None:
        nframes = targ_obj.shape[1]
        weight = weight.unsqueeze(1).expand(-1, nframes)
        weight_lhand = weight[mask_lhand]
        weight_rhand = weight[mask_rhand]
        ro_lhand_loss = F.mse_loss(pred_ro_lhand, targ_ro_lhand, reduction="none")
        ro_rhand_loss = F.mse_loss(pred_ro_rhand, targ_ro_rhand, reduction="none")
        ro_lhand_loss = ro_lhand_loss.mean([1, 2])*weight_lhand
        ro_rhand_loss = ro_rhand_loss.mean([1, 2])*weight_rhand
        ro_lhand_loss = ro_lhand_loss.mean()
        ro_rhand_loss = ro_rhand_loss.mean()
    else:
        ro_lhand_loss = F.mse_loss(pred_ro_lhand, targ_ro_lhand)
        ro_rhand_loss = F.mse_loss(pred_ro_rhand, targ_ro_rhand)
    return ro_lhand_loss + ro_rhand_loss

# ro: relative orientation
def get_ro(hand, obj, valid_mask):
    hand_orient = hand[valid_mask][..., 3:9]
    obj_orient = obj[valid_mask][..., 3:9]
    hand_orient_rotmat = rot6d_to_rotmat(hand_orient)
    obj_orient_rotmat = rot6d_to_rotmat(obj_orient)
    ro_hand_obj = relative_rotation_matrix(hand_orient_rotmat, obj_orient_rotmat)
    return ro_hand_obj

def get_filtered_loss_valid_mask(loss, valid_mask, loss_weight=None):
    if len(loss.shape)==3:
        loss_mean = loss.mean([2])
    elif len(loss.shape)==4:
        loss_mean = loss.mean([2, 3])
    filtered_loss = torch.where(valid_mask, loss_mean, torch.zeros_like(loss_mean))
    filtered_loss_summed = filtered_loss.sum(1)
    valid_mask_summed = valid_mask.sum(1)
    valid_mask_summed = torch.where(valid_mask_summed!=0, valid_mask_summed, torch.tensor(1).to(valid_mask_summed.device))
    # batch_mean
    filtered_loss_bm = filtered_loss_summed/valid_mask_summed
    if loss_weight is not None:
        filtered_loss_bm = filtered_loss_bm*loss_weight
    filtered_loss = filtered_loss_bm.mean()
    return filtered_loss

def get_filtered_joint_loss_valid_mask(loss, valid_mask, loss_weight=None):
    loss_mean = loss.mean([3])
    filtered_loss = torch.where(valid_mask, loss_mean, torch.zeros_like(loss_mean))
    filtered_loss_summed = filtered_loss.sum(2)
    valid_mask_summed = valid_mask.sum(2)
    valid_mask_summed = torch.where(valid_mask_summed!=0, valid_mask_summed, torch.tensor(1).to(valid_mask_summed.device))
    # batch_mean
    filtered_loss_bm = filtered_loss_summed/valid_mask_summed
    if loss_weight is not None:
        filtered_loss_bm = filtered_loss_bm*loss_weight
    filtered_loss = filtered_loss_bm.mean()
    return filtered_loss

def get_filtered_loss_valid_map(loss, valid_map, weight=None):
    filtered_loss = torch.where(valid_map, loss, torch.zeros_like(loss))
    filtered_loss_summed = filtered_loss.sum([1, 2, 3]) # batch, nframes, 1024, 21
    valid_map_summed = valid_map.sum([1, 2, 3])
    valid_map_summed = torch.where(valid_map_summed!=0, valid_map_summed, torch.tensor(1).to(valid_map_summed.device))
    # batch_mean
    filtered_loss_bm = filtered_loss_summed/valid_map_summed
    if weight is not None:
        filtered_loss_bm = filtered_loss_bm*weight
    filtered_loss = filtered_loss_bm.mean()
    return filtered_loss

def relative_rotation_matrix(R1, R2):
    R1_inv = torch.inverse(R1)
    relative_matrix = torch.matmul(R2, R1_inv)
    return relative_matrix

def get_penetration_loss(
    pred_X0_lhand, pred_X0_rhand, pred_X0_obj, 
    lhand_layer, rhand_layer, obj_pc_org, 
    valid_mask_lhand, valid_mask_rhand, 
    dataset_name, obj_pc_top_idx=None
):
    lhand_mesh, lhand_verts = get_pytorch3d_meshes(pred_X0_lhand, lhand_layer)
    rhand_mesh, rhand_verts = get_pytorch3d_meshes(pred_X0_rhand, rhand_layer)
    lhand_normal = lhand_mesh.verts_normals_packed().view(-1, 778, 3)
    rhand_normal = rhand_mesh.verts_normals_packed().view(-1, 778, 3)
    batch_size, npts = obj_pc_org.shape[:2]
    transf_obj_pc = get_transformed_obj_pc(pred_X0_obj, obj_pc_org, dataset_name, obj_pc_top_idx)
    transf_obj_pc = transf_obj_pc.reshape(-1, npts, 3)
    
    valid_mask_lhand = valid_mask_lhand.reshape(-1)
    valid_mask_rhand = valid_mask_rhand.reshape(-1)
    
    penet_loss_lhand = get_penet_hand_obj_loss(
        lhand_verts, lhand_normal, 
        transf_obj_pc, valid_mask_lhand
    )
    
    penet_loss_rhand = get_penet_hand_obj_loss(
        rhand_verts, rhand_normal, 
        transf_obj_pc, valid_mask_rhand
    )
    return penet_loss_lhand + penet_loss_rhand

def get_penet_hand_obj_loss(hand_verts, hand_normal, obj_pc, valid_mask_hand):
    nn_dist, nn_idx = get_NN(obj_pc, hand_verts)
    interior = get_interior(hand_normal, hand_verts, obj_pc, nn_idx)
    nn_dist = nn_dist.sqrt()
    nn_dist = nn_dist[valid_mask_hand]
    interior = interior[valid_mask_hand]
    if interior.sum() > 0:
        penet_loss = nn_dist[interior].mean()
    else:
        penet_loss = torch.FloatTensor(1).fill_(0).cuda()
    return penet_loss