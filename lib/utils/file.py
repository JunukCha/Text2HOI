import os
import os.path as osp

import cv2
import numpy as np
import tqdm
import json
import pickle
from argparse import ArgumentParser
import trimesh

import yaml
import wandb
from easydict import EasyDict as edict
from lib.utils.proc import proc_numpy

def make_model_result_folder(root, train_type):
    model_folder = osp.join(root, "model")
    result_folder = osp.join(root, "result", train_type)
    
    os.makedirs(model_folder, exist_ok=True)
    os.makedirs(result_folder, exist_ok=True)
    return model_folder, result_folder

def make_save_folder(save_root):
    os.makedirs(save_root, exist_ok=True)
    return save_root

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return edict(config)

def update_config(config):
    parser = ArgumentParser()
    def add_argument(config, parser, parent_key=None):
        for key, value in config.items():
            if parent_key is not None:
                key = f"{parent_key}.{key}"
            if not isinstance(value, dict):
                parser.add_argument(f"--{key}", type=type(value), default=None)
            else:
                add_argument(value, parser, key)
    add_argument(config, parser)
    args, unknown_args = parser.parse_known_args()
    for unknown_arg in unknown_args:
        if "--" in unknown_arg:
            raise Exception(f"Not allowed argument! {unknown_arg}")
    for key, value in vars(args).items():
        if value is not None:
            if "." in key:
                keys = key.split(".")
                d = config
                for key in keys[:-1]:
                    d = d[key]
                d[keys[-1]] = value
            else:
                config[key] = value
    return config

def read_json(json_file):
    with open(json_file, "r") as f:
        data = json.load(f)
    return data

def read_pkl(pkl_file):
    with open(pkl_file, "rb") as f:
        data = pickle.load(f)
    return data

def wandb_login(
    config, 
    config_model, 
    project_name=None, 
    model_name=None, 
    relogin=True
):
    wandb.login(relogin=relogin)
    if project_name is None:
        project_name = config.project_name
    if model_name is None:
        model_name = config_model.model_name
    wandb.init(
        project=project_name, 
        name=model_name, 
        config=config, 
    )

    return wandb

def save_video(frames, fps, save_path):
    os.makedirs(osp.dirname(save_path), exist_ok=True)
    height, width = frames.shape[1:3]
    writer = cv2.VideoWriter(
        save_path, 
        cv2.VideoWriter_fourcc(*'mp4v'), 
        fps, (width, height),
    )
    if frames.shape[-1] == 4:
        frames = frames[..., :3] # remove alpha channel

    if frames.max() <= 1:
        frames = (frames*255).astype(np.uint8)
    for frame in tqdm.tqdm(frames, desc="saving video"):
        writer.write(frame)
    writer.release()
    
def save_mesh_obj(
    vertices, faces, save_folder,
):
    os.makedirs(save_folder, exist_ok=True)
    faces = proc_numpy(faces)
    for obj_idx, verts in enumerate(vertices):
        verts = proc_numpy(verts)
        trimesh.Trimesh(verts, faces).export(osp.join(save_folder, f"{obj_idx:03d}.obj"))