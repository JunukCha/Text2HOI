import numpy as np

import torch
from pytorch3d.ops.knn import knn_points
from pytorch3d.structures import Meshes

from lib.utils.rot import (
    rot6d_to_axis_angle, 
    rot6d_to_rotmat, 
    axis_angle_to_quaternion, 
    quaternion_apply, 
)

def select_from_groups(data, num_groups=15, test=False):
    B, N = data.shape[:2]
    if not test:
        ave_duration, remainder = divmod(N, num_groups)
        offset1 = np.multiply(np.arange(remainder), (ave_duration + 1)) + np.random.randint(ave_duration+1, size=remainder)
        offset2 = np.multiply(np.arange(remainder, num_groups), (ave_duration)) + remainder + np.random.randint(ave_duration, size=num_groups-remainder)
        final_offsets = np.append(offset1, offset2)
    elif test:
        final_offsets = np.linspace(0, N-1, num_groups).astype(np.int32)
    selected_data = data[:, final_offsets]
    return selected_data

def get_hand_layer_out(hand_params, hand_layer):
    bs, nframes = hand_params.shape[:2]
    pred_rot6d = hand_params[..., 3:]
    pred_pose = rot6d_to_axis_angle(pred_rot6d).reshape(-1, 48)
    out = hand_layer(
        torch.zeros(bs*nframes, 10).to(hand_params.device), 
        pred_pose[..., :3].reshape(-1, 3), 
        pred_pose[..., 3:].reshape(-1, 45), 
    )
    return out

def get_hand_verts(hand_params, hand_layer):
    bs, nframes = hand_params.shape[:2]
    pred_trans = hand_params[..., :3]
    out = get_hand_layer_out(hand_params, hand_layer)
    verts = out.vertices.reshape(bs, nframes, 778, 3)
    verts = verts + pred_trans.unsqueeze(2)
    return verts
    
def get_hand_joints_w_tip(hand_params, hand_layer):
    bs, nframes = hand_params.shape[:2]
    pred_trans = hand_params[..., :3]
    out = get_hand_layer_out(hand_params, hand_layer)
    hand_joints_w_tip = out.joints_w_tip.reshape(bs, nframes, 21, 3)
    hand_joints_w_tip = hand_joints_w_tip + pred_trans.unsqueeze(2)
    return hand_joints_w_tip

# pc: point cloud
def get_transformed_obj_pc(obj_params, obj_pc, dataset_name, obj_top_idx=None):
    bs, nframes = obj_params.shape[:2]
    obj_trans = obj_params[..., :3]
    obj_rot6d = obj_params[..., 3:9]
    obj_rotmat = rot6d_to_rotmat(obj_rot6d).reshape(bs, nframes, 3, 3)
    if dataset_name == "arctic" and obj_params.shape[-1] == 10 and obj_top_idx is not None:
        obj_top_idx = obj_top_idx.bool()
        obj_angle = obj_params[..., 9:10]
        quat_arti = axis_angle_to_quaternion(torch.FloatTensor([0, 0, -1]).to(obj_params.device).view(1, 1, 3)*obj_angle)
        obj_pc = obj_pc.unsqueeze(1).expand(-1, nframes, -1, -1)
        obj_pc2 = obj_pc.clone()
        obj_top_idx = obj_top_idx.unsqueeze(1).expand(-1, nframes, -1)
        obj_pc2[obj_top_idx] = quaternion_apply(quat_arti[:, :, None], obj_pc)[obj_top_idx]
        obj_pc_rotated = torch.einsum("btij,btkj->btki", obj_rotmat, obj_pc2)
    elif dataset_name == "h2o":
        obj_pc_rotated = torch.einsum("btij,bkj->btki", obj_rotmat, obj_pc)
    elif dataset_name == "grab":
        obj_pc_rotated = torch.einsum("btij,bki->btkj", obj_rotmat, obj_pc)
    else:
        obj_pc_rotated = torch.einsum("btij,bkj->btki", obj_rotmat, obj_pc)
    obj_pc_transformed = obj_pc_rotated + obj_trans.unsqueeze(2)
    return obj_pc_transformed

def get_action_classifier_input(
    x_lhand, x_rhand, 
    lhand_layer, rhand_layer, 
    is_lhand, is_rhand, 
    test=False, 
):
    assert len(x_lhand.shape) == 2 # T, D
    assert len(x_rhand.shape) == 2 # T, D
    
    lhand_joints = get_hand_joints_w_tip(x_lhand.unsqueeze(0), lhand_layer)
    rhand_joints = get_hand_joints_w_tip(x_rhand.unsqueeze(0), rhand_layer)
    lhand_joints = select_from_groups(lhand_joints, num_groups=15, test=test)
    rhand_joints = select_from_groups(rhand_joints, num_groups=15, test=test)
    if is_lhand:
        hand_org = lhand_joints[:, 0, 0].clone()
    if is_rhand:
        hand_org = rhand_joints[:, 0, 0].clone()
        
    if is_lhand:
        lhand_joints_ra = lhand_joints - hand_org
    else:
        lhand_joints_ra = torch.zeros_like(lhand_joints).to(lhand_joints.device)
    if is_rhand:
        rhand_joints_ra = rhand_joints - hand_org
    else:
        rhand_joints_ra = torch.zeros_like(rhand_joints).to(rhand_joints.device)
    action_input = torch.cat([lhand_joints_ra, rhand_joints_ra], dim=2)
    return action_input

def get_hand_obj_dist_map(
    pred_lhand, pred_rhand, pred_obj, obj_pc,
    lhand_layer, rhand_layer, dataset_name, obj_pc_top_idx=None
):
    bs, nframes = pred_obj.shape[:2]
    
    lhand_joints_w_tip = get_hand_joints_w_tip(pred_lhand, lhand_layer)
    rhand_joints_w_tip = get_hand_joints_w_tip(pred_rhand, rhand_layer)
    obj_pc_transformed = get_transformed_obj_pc(pred_obj, obj_pc, dataset_name, obj_pc_top_idx)

    pred_ldist_map = torch.cdist(
        obj_pc_transformed.reshape(-1, 1024, 3), 
        lhand_joints_w_tip.reshape(-1, 21, 3)
    ).reshape(bs, nframes, 1024, 21)
    
    pred_rdist_map = torch.cdist(
        obj_pc_transformed.reshape(-1, 1024, 3), 
        rhand_joints_w_tip.reshape(-1, 21, 3)
    ).reshape(bs, nframes, 1024, 21)

    return pred_ldist_map, pred_rdist_map

def get_pytorch3d_meshes(pred_X0_hand, hand_layer):
    hand_verts = get_hand_verts(pred_X0_hand, hand_layer)
    hand_verts = hand_verts.reshape(-1, 778, 3)
    hand_faces = torch.LongTensor(hand_layer.faces.astype(np.int16)).cuda()
    hand_faces = hand_faces.unsqueeze(0).expand(hand_verts.shape[0], -1, -1)
    hand_mesh = Meshes(hand_verts, hand_faces)
    return hand_mesh, hand_verts

def get_NN(src_xyz, trg_xyz, k=1):
    '''
    :param src_xyz: [B, N1, 3]
    :param trg_xyz: [B, N2, 3]
    :return: nn_dists, nn_dix: all [B, 3000] tensor for NN distance and index in N2
    '''
    B = src_xyz.size(0)
    src_lengths = torch.full(
        (src_xyz.shape[0],), src_xyz.shape[1], dtype=torch.int64, device=src_xyz.device
    )  # [B], N for each num
    trg_lengths = torch.full(
        (trg_xyz.shape[0],), trg_xyz.shape[1], dtype=torch.int64, device=trg_xyz.device
    )
    src_nn = knn_points(src_xyz, trg_xyz, lengths1=src_lengths, lengths2=trg_lengths, K=k)  # [dists, idx]
    nn_dists = src_nn.dists[..., 0]
    nn_idx = src_nn.idx[..., 0]
    return nn_dists, nn_idx

def get_interior(src_face_normal, src_xyz, trg_xyz, trg_NN_idx):
    '''
    :param src_face_normal: [B, 778, 3], surface normal of every vert in the source mesh
    :param src_xyz: [B, 778, 3], source mesh vertices xyz
    :param trg_xyz: [B, 3000, 3], target mesh vertices xyz
    :param trg_NN_idx: [B, 3000], index of NN in source vertices from target vertices
    :return: interior [B, 3000], inter-penetrated trg vertices as 1, instead 0 (bool)
    '''
    N1, N2 = src_xyz.size(1), trg_xyz.size(1)

    # get vector from trg xyz to NN in src, should be a [B, 3000, 3] vector
    NN_src_xyz = batched_index_select(src_xyz, trg_NN_idx)  # [B, 3000, 3]
    NN_vector = NN_src_xyz - trg_xyz  # [B, 3000, 3]

    # get surface normal of NN src xyz for every trg xyz, should be a [B, 3000, 3] vector
    NN_src_normal = batched_index_select(src_face_normal, trg_NN_idx)

    interior = (NN_vector * NN_src_normal).sum(dim=-1) > 0  # interior as true, exterior as false
    return interior

def batched_index_select(input, index, dim=1):
    '''
    :param input: [B, N1, *]
    :param dim: the dim to be selected
    :param index: [B, N2]
    :return: [B, N2, *] selected result
    '''
    views = [input.size(0)] + [1 if i != dim else -1 for i in range(1, len(input.shape))]
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.view(views).expand(expanse)
    return torch.gather(input, dim=dim, index=index)