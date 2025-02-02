import os.path as osp

import numpy as np

import torch

from lib.utils.rot import (
    rot6d_to_axis_angle,
    rot6d_to_rotmat,
    axis_angle_to_quaternion, 
    quaternion_apply, 
)

def process_hand_result(hand_layer, hand_params):
    hand_pose = hand_params[:, 3:]
    hand_pose = rot6d_to_axis_angle(hand_pose).reshape(-1, 48)
    hand_trans = hand_params[:, :3]
    duration = hand_trans.shape[0]
    out = hand_layer(
        global_orient=hand_pose[:, :3],
        hand_pose=hand_pose[:, 3:48],
        betas=torch.zeros((duration, 10)).to(hand_pose.device)
    )
    hand_trans = hand_trans.unsqueeze(1)
    hand_vertices = out.vertices+hand_trans
    hand_faces = hand_layer.faces.copy().astype(np.int16)
    hand_faces = torch.LongTensor(hand_faces).to(hand_pose.device)
    return hand_vertices, hand_faces

def process_obj_result(obj_verts, obj_params, dataset_name, obj_top_idx=None):
    nframes = obj_params.shape[0]
    obj_trans = obj_params[:, :3]
    obj_rot6d = obj_params[:, 3:9]
    obj_rotmat = rot6d_to_rotmat(obj_rot6d).reshape(-1, 3, 3)
    if dataset_name == "arctic" and obj_params.shape[-1] == 10 and obj_top_idx is not None:
        obj_top_idx = obj_top_idx.bool()
        obj_angle = obj_params[..., 9:10]
        quat_arti = axis_angle_to_quaternion(torch.FloatTensor([0, 0, -1]).to(obj_params.device).view(1, 3)*obj_angle)
        obj_verts = obj_verts.unsqueeze(0).expand(nframes, -1, -1)
        obj_verts2 = obj_verts.clone()
        obj_top_idx = obj_top_idx.unsqueeze(0).expand(nframes, -1)
        obj_verts2[obj_top_idx] = quaternion_apply(quat_arti[:, None], obj_verts)[obj_top_idx]
        obj_pc_rotated = torch.einsum("tij,tkj->tki", obj_rotmat, obj_verts2)
    elif dataset_name == "h2o":
        obj_pc_rotated = torch.einsum("tij,kj->tki", obj_rotmat, obj_verts)
    elif dataset_name == "grab":
        obj_pc_rotated = torch.einsum("tij,ki->tkj", obj_rotmat, obj_verts)
    else:
        obj_pc_rotated = torch.einsum("tij,kj->tki", obj_rotmat, obj_verts)
    obj_verts_transformed = obj_pc_rotated+obj_trans.unsqueeze(1)
    return obj_verts_transformed

def processe_params(hand_data_path, obj_data_path):
    hand_data = np.load(hand_data_path, allow_pickle=True)
    object_data = np.load(obj_data_path, allow_pickle=True)
    left_hand_data = hand_data[()]["left"]
    left_global_rot = left_hand_data["rot"]
    left_pose = left_hand_data["pose"]
    left_pose = np.concatenate([left_global_rot, left_pose], axis=1)
    frame_cnt = left_pose.shape[0]
    left_shape = left_hand_data["shape"]
    matrix_size = (frame_cnt, ) + left_shape.shape
    left_shape = np.full(matrix_size, left_shape)
    left_trans = left_hand_data["trans"]

    right_hand_data = hand_data[()]["right"]
    right_global_rot = right_hand_data["rot"]
    right_pose = right_hand_data["pose"]
    right_pose = np.concatenate([right_global_rot, right_pose], axis=1)
    frame_cnt = right_pose.shape[0]
    right_shape = right_hand_data["shape"]
    matrix_size = (frame_cnt, ) + right_shape.shape
    right_shape = np.full(matrix_size, right_shape)
    right_trans = right_hand_data["trans"]

    hand_data = {
        "left.pose": left_pose, 
        "left.shape": left_shape, 
        "left.trans": left_trans, 
        "right.pose": right_pose, 
        "right.shape": right_shape, 
        "right.trans": right_trans, 
    }
    
    object_angle = object_data[:, :1]
    object_global_rot = object_data[:, 1:4]
    object_trans = object_data[:, 4:]
    object_data = {
        "object.angle": object_angle,
        "object.global_rot": object_global_rot,
        "object.trans": object_trans,
    }
    return hand_data, object_data

