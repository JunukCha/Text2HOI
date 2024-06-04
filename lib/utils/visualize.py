import os
import os.path as osp

import numpy as np
import torch

from lib.utils.proc import proc_torch_cuda
from lib.utils.rot import get_rotmat_x, get_rotmat_y

def proc_rot_obj(
    obj_verts, duration, rot_axis="y"
):
    rotmat_org = proc_torch_cuda(get_rotmat_x(-np.pi/2))
    
    obj_verts_rot = [
        torch.einsum("ij,bkj->bki", rotmat_org, obj_verts.clone().unsqueeze(0))
    ]
    
    radians = 2*np.pi/(duration-1)
    
    if rot_axis=="x":
        rotmat = proc_torch_cuda(get_rotmat_x(radians))
    elif rot_axis=="y":
        rotmat = proc_torch_cuda(get_rotmat_y(radians))
        
    for _ in range(duration-1):
        obj_verts_rot.append(
            # N, 3
            torch.einsum("ij,bkj->bki", rotmat, obj_verts_rot[-1])
        )
    obj_verts_rot = torch.cat(obj_verts_rot)
    return obj_verts_rot

def render_motion(
    renderer, 
    obj_verts_tf, obj_faces, 
    lhand_vertices, lhand_faces, is_lhand, 
    rhand_vertices, rhand_faces, is_rhand, 
):
    vis_vertices = []
    vis_faces = []
    mesh_kind = []
    
    vis_vertices.append(obj_verts_tf)
    vis_faces.append(obj_faces)
    mesh_kind.append("obj")
    
    if is_lhand:
        vis_vertices.append(lhand_vertices)
        vis_faces.append(lhand_faces)
        mesh_kind.append("lhand")
    
    if is_rhand:
        vis_vertices.append(rhand_vertices)
        vis_faces.append(rhand_faces)
        mesh_kind.append("rhand")
    
    motion_video = renderer.render_video(
        vis_vertices, 
        vis_faces, 
        mesh_kind
    )
    return motion_video

def merge_videos(
    motion_video, motion_video_side, 
    lhand_contact_video, rhand_contact_video, 
    obj_cont_video
):
    if motion_video_side is not None:
        merged_video = np.concatenate([motion_video, motion_video_side], axis=2) # width
    else:
        merged_video = motion_video
    if lhand_contact_video is not None:
        merged_video = np.concatenate([merged_video, lhand_contact_video], axis=2)
    if rhand_contact_video is not None:
        merged_video = np.concatenate([merged_video, rhand_contact_video], axis=2)
    if obj_cont_video is not None:
        if merged_video.shape[2] == obj_cont_video.shape[2]:
            merged_video = np.concatenate([merged_video, obj_cont_video], axis=2)
    return merged_video
    
def render_videos(
    renderer, lhand_verts, lhand_faces, 
    rhand_verts, rhand_faces, 
    obj_verts_tf, obj_faces, 
    is_lhand, is_rhand, 
):
    motion_video = render_motion(
        renderer, 
        obj_verts_tf, obj_faces, 
        lhand_verts, lhand_faces, is_lhand, 
        rhand_verts, rhand_faces, is_rhand, 
    )
    
    return motion_video