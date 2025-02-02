import os
import os.path as osp

import glob
from scipy.spatial.transform import Rotation as R
import numpy as np

import torch

def process_text(
    action_name, is_lhand, is_rhand, 
    text_descriptions, return_key=False, 
):
    if is_lhand and is_rhand:
        text = f"{action_name} with both hands."
    elif is_lhand:
        text = f"{action_name} with left hand."
    elif is_rhand:
        text = f"{action_name} with right hand."
    text_key = text.capitalize()
    if return_key:
        return text_key
    else:
        text_description = text_descriptions[text_key]
        text = np.random.choice(text_description)
        return text
    
def get_data_path_h2o(data_path):
    hand_pose_manos = glob.glob(osp.join(data_path, "hand_pose_mano", "*.txt"))
    obj_pose_rts = glob.glob(osp.join(data_path, "obj_pose_rt", "*.txt"))
    cam_poses = glob.glob(osp.join(data_path, "cam_pose", "*.txt"))
    action_labels = glob.glob(osp.join(data_path, "action_label", "*.txt"))
    
    hand_pose_manos.sort()
    obj_pose_rts.sort()
    cam_poses.sort()
    action_labels.sort()
    return hand_pose_manos, obj_pose_rts, cam_poses, action_labels

def process_hand_pose_h2o(hand_pose, hand_trans, extrinsic_matrix):
    rot = R.from_rotvec(hand_pose[:3])
    mat = np.concatenate((np.concatenate((rot.as_matrix(), np.array(
        hand_trans[np.newaxis]).T), axis=1), [[0, 0, 0, 1]]), axis=0)
    mat_proj = np.dot(extrinsic_matrix, mat)
    rot_vec = R.from_matrix(mat_proj[:3, :3]).as_rotvec()
    return rot_vec

def process_hand_trans_h2o(hand_pose, hand_beta, hand_trans, extrinsic_matrix, hand_layer):
    mano_keypoints_3d = hand_layer(
        torch.FloatTensor(np.array([hand_beta])).cuda(),
        torch.FloatTensor(np.array([hand_pose[:3]])).cuda(),
        torch.FloatTensor(np.array([hand_pose[3:]])).cuda(),
    ).joints

    hand_origin = mano_keypoints_3d[0][0]
    origin = torch.unsqueeze(
        hand_origin, 1) + torch.tensor([hand_trans]).cuda().T
    origin = origin.float()
    extrinsic_matrix = torch.FloatTensor(extrinsic_matrix).cuda()
    mat_proj = torch.matmul(
        extrinsic_matrix, torch.cat((origin, torch.ones((1, 1)).cuda())))
    new_trans = mat_proj.T[0, :3] - hand_origin
    return new_trans.cpu().numpy(), hand_origin.cpu().numpy()