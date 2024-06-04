import numpy as np
import tqdm
import cv2
import matplotlib.cm as cm

import torch

from pytorch3d.renderer import (
    FoVPerspectiveCameras, look_at_view_transform,
    OrthographicCameras, 
    RasterizationSettings, TexturesVertex,
    MeshRenderer, MeshRasterizer, HardPhongShader,
    PointsRasterizationSettings,
    PointsRasterizer,
    PointsRenderer,
    AlphaCompositor, 
    PointLights
)
from pytorch3d.structures import Meshes
from pytorch3d.structures import join_meshes_as_scene, join_meshes_as_batch
from lib.utils.rot import get_rotmat_x, get_rotmat_y, get_rotmat_z

def get_h2o_front_camera():
    R_z = get_rotmat_z(np.pi)
    R_x = get_rotmat_x(-np.pi/9)
    R = np.matmul(R_x, R_z)
    R = torch.from_numpy(R).unsqueeze(0)
    T = torch.FloatTensor([0., 0., 0.5]).unsqueeze(0)
    return R, T

def get_grab_front_camera():
    R_x = get_rotmat_x(-np.pi/2)
    R_y = get_rotmat_y(np.pi)
    R = np.matmul(R_y, R_x)
    R = torch.from_numpy(R).unsqueeze(0)
    T = torch.FloatTensor([0., -1.0, 0.8]).unsqueeze(0)
    # T = torch.FloatTensor([0., -1.3, 1.5]).unsqueeze(0)
    return R, T

def get_arctic_front_camera():
    R = get_rotmat_x(np.pi/2+np.pi/8)
    R = torch.from_numpy(R).unsqueeze(0)
    T = torch.FloatTensor([0., -1., 1.8]).unsqueeze(0)
    return R, T


class Renderer():
    def __init__(self, device, img_size=512, camera=None, fps=30):
        self.device = device
        self.fps = fps
        self.camera = camera
        if camera == "h2o_front":
            lights_front = PointLights(
                device=device, 
                location=[[0.0, -1, 0.5]], 
                ambient_color=((0.6, 0.6, 0.6),), 
                diffuse_color=((0.2, 0.2, 0.2),), 
                specular_color=((0.2, 0.2, 0.2),), 
            )
            R, T = get_h2o_front_camera()
        elif camera == "grab_front":
            lights_front = PointLights(
                device=device, 
                location=[[0.0, -1, 0.5]], 
                ambient_color=((0.6, 0.6, 0.6),), 
                diffuse_color=((0.2, 0.2, 0.2),), 
                specular_color=((0.2, 0.2, 0.2),), 
            )
            R, T = get_grab_front_camera()
        elif camera == "arctic_front":
            lights_front = PointLights(
                device=device, 
                location=[[0.0, 1.0, 0.5]], 
                ambient_color=((0.6, 0.6, 0.6),), 
                diffuse_color=((0.2, 0.2, 0.2),), 
                specular_color=((0.2, 0.2, 0.2),), 
            )
            R, T = get_arctic_front_camera()
        cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
        
        # BGR
        self.colors = {
            "lhand": [180/255, 105/255, 255/255],
            "rhand": [47/255, 180/255, 37/255],
            "obj": [235/255, 206/255, 135/255],
        }
        # Define the settings for rasterization and shading. Here we set the output image to be of size
        # 512x512. As we are rendering images for visualization purposes only we will set faces_per_pixel=1
        # and blur_radius=0.0. Refer to rasterize_meshes.py for explanations of these parameters.
        raster_settings = RasterizationSettings(
            image_size=img_size, 
            blur_radius=0.0, 
            max_faces_per_bin=1, 
            bin_size=0, 
            faces_per_pixel=1, 
        )
        
        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
            shader=HardPhongShader(device=device, cameras=cameras, lights=lights_front), 
        )

    def render_video(self, vertices, faces, mesh_kind, return_numpy=True):
        duration = vertices[0].shape[0]
        frames_mesh = []
        for time in tqdm.tqdm(range(duration), desc="render video"):
            meshes = []
            for vs, f, kind in zip(vertices, faces, mesh_kind):
                v = vs[time]
                v = self.proess_data(v)
                f = self.proess_data(f)
                mesh = Meshes(v, f)
                mesh = self.put_colors(mesh, kind)
                meshes.append(mesh)
            meshes = join_meshes_as_scene(meshes)
            frames_mesh.append(meshes)
        frames_mesh = join_meshes_as_batch(frames_mesh)
        frames = self.renderer(frames_mesh)
        if return_numpy:
            frames = frames.cpu().numpy()
        return frames
    
    def save_video(self, vertices, faces, mesh_kind, save_path):
        frames = self.render_video(vertices, faces, mesh_kind)
        height, width = frames.shape[1:3]
        writer = cv2.VideoWriter(
            save_path, 
            cv2.VideoWriter_fourcc(*'mp4v'), 
            self.fps, (width, height),
        )
        if frames.shape[-1] == 4:
            frames = frames[..., :3] # remove alpha channel

        if frames.max() <= 1:
            frames = (frames*255).astype(np.uint8)
        for frame in tqdm.tqdm(frames, desc="saving video"):
            writer.write(frame)
        writer.release()

    def renderer_video_contact(self, mesh):
        frames = self.renderer(mesh)
        frames = frames.cpu().numpy()
        
        if frames.shape[-1] == 4:
            frames = frames[..., :3] # remove alpha channel
        if frames.max() <= 1.001:
            frames = (frames*255)
        if frames.dtype != np.uint8:
            frames = frames.astype(np.uint8)
        return frames
        
    def save_video_contact(self, mesh, save_path):
        frames = self.renderer(mesh)
        frames = frames.cpu().numpy()
        height, width = frames.shape[1:3]
        writer = cv2.VideoWriter(
            save_path, 
            cv2.VideoWriter_fourcc(*'mp4v'), 
            self.fps, (width, height),
        )
        if frames.shape[-1] == 4:
            frames = frames[..., :3] # remove alpha channel
        if frames.max() <= 1.001:
            frames = (frames*255)
        if frames.dtype != np.uint8:
            frames = frames.astype(np.uint8)
        
        for frame in tqdm.tqdm(frames, desc="saving video"):
            writer.write(frame)
        writer.release()
        
    def save_video_w_frame(self, frames, save_path):
        height, width = frames.shape[1:3]
        writer = cv2.VideoWriter(
            save_path, 
            cv2.VideoWriter_fourcc(*'mp4v'), 
            self.fps, (width, height),
        )
        for frame in tqdm.tqdm(frames, desc="saving video"):
            writer.write(frame)
        writer.release()

    def proess_data(self, d):
        if isinstance(d, np.ndarray):
            try:
                d = torch.FloatTensor(d)
            except TypeError:
                if d.dtype == "uint32":
                    d = torch.FloatTensor(d.astype(np.int32))
                else:
                    raise Exception("Something wrong in type!")
        if d.device != self.device:
            d = d.to(self.device)
        d = d.unsqueeze(0)
        return d
    
    def put_colors(self, mesh, mesh_idx):
        mesh.textures = TexturesVertex(
            # torch.randn_like(v)
            torch.FloatTensor(self.colors[mesh_idx])\
                .unsqueeze(0)\
                .unsqueeze(0).expand(-1, mesh.verts_padded().shape[1], -1).to(self.device)
        )
        return mesh