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

def get_h2o_side_camera():
    R_z = get_rotmat_z(np.pi)
    R_y = get_rotmat_y(-np.pi/2)
    R = np.matmul(R_y, R_z)
    R = torch.from_numpy(R).unsqueeze(0)
    T = torch.FloatTensor([0.5, 0, 1.]).unsqueeze(0)
    return R, T

def get_grab_front_camera():
    R_x = get_rotmat_x(-np.pi/2)
    R_y = get_rotmat_y(np.pi)
    R = np.matmul(R_y, R_x)
    R = torch.from_numpy(R).unsqueeze(0)
    T = torch.FloatTensor([0., -1.0, 0.8]).unsqueeze(0)
    # T = torch.FloatTensor([0., -1.3, 1.5]).unsqueeze(0)
    return R, T

def get_grab_side_camera():
    R_x = get_rotmat_x(-np.pi/2)
    R_y = get_rotmat_y(np.pi)
    R_z = get_rotmat_z(np.pi/2)
    R = np.matmul(R_y, R_x)
    R = np.matmul(R_z, R)
    R = torch.from_numpy(R).unsqueeze(0)
    T = torch.FloatTensor([0., -1.3, 1.5]).unsqueeze(0)
    return R, T

def get_arctic_front_camera():
    R = get_rotmat_x(np.pi/2+np.pi/8)
    R = torch.from_numpy(R).unsqueeze(0)
    T = torch.FloatTensor([0., -1., 1.8]).unsqueeze(0)
    return R, T

def get_arctic_side_camera():
    R_x = get_rotmat_x(np.pi/2)
    R_z = get_rotmat_z(np.pi/2)
    R = np.matmul(R_z, R_x)
    R = torch.from_numpy(R).unsqueeze(0)
    T = torch.FloatTensor([0., -1, 1.]).unsqueeze(0)
    return R, T

def get_left_hand_camera():
    R_z = get_rotmat_z(-np.pi/2)
    R_x_front = get_rotmat_x(-np.pi/2)
    R_x_back = get_rotmat_x(np.pi/2)
    R_front = np.matmul(R_x_front, R_z)
    R_back = np.matmul(R_x_back, R_z)
    R_front = torch.from_numpy(R_front).unsqueeze(0)
    R_back = torch.from_numpy(R_back).unsqueeze(0)
    T = torch.FloatTensor([0, 0, 0.8]).unsqueeze(0)
    return R_front, R_back, T

def get_right_hand_camera():
    R_z = get_rotmat_z(np.pi/2)
    R_x_front = get_rotmat_x(-np.pi/2)
    R_x_back = get_rotmat_x(np.pi/2)
    R_front = np.matmul(R_x_front, R_z)
    R_back = np.matmul(R_x_back, R_z)
    R_front = torch.from_numpy(R_front).unsqueeze(0)
    R_back = torch.from_numpy(R_back).unsqueeze(0)
    T = torch.FloatTensor([0, 0, 0.8]).unsqueeze(0)
    return R_front, R_back, T


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
        elif camera == "h2o_side":
            lights_front = PointLights(
                device=device, 
                location=[[0.0, -1, 0.5]], 
                ambient_color=((0.6, 0.6, 0.6),), 
                diffuse_color=((0.2, 0.2, 0.2),), 
                specular_color=((0.2, 0.2, 0.2),), 
            )
            R, T = get_h2o_side_camera()
        elif camera == "grab_front":
            lights_front = PointLights(
                device=device, 
                location=[[0.0, -1, 0.5]], 
                ambient_color=((0.6, 0.6, 0.6),), 
                diffuse_color=((0.2, 0.2, 0.2),), 
                specular_color=((0.2, 0.2, 0.2),), 
            )
            R, T = get_grab_front_camera()
        elif camera == "grab_side":
            lights_front = PointLights(
                device=device, 
                location=[[0.0, -1, 0.5]], 
                ambient_color=((0.6, 0.6, 0.6),), 
                diffuse_color=((0.2, 0.2, 0.2),), 
                specular_color=((0.2, 0.2, 0.2),), 
            )
            R, T = get_grab_side_camera()
        elif camera == "arctic_front":
            lights_front = PointLights(
                device=device, 
                location=[[0.0, 1.0, 0.5]], 
                ambient_color=((0.6, 0.6, 0.6),), 
                diffuse_color=((0.2, 0.2, 0.2),), 
                specular_color=((0.2, 0.2, 0.2),), 
            )
            R, T = get_arctic_front_camera()
        elif camera == "arctic_side":
            lights_front = PointLights(
                device=device, 
                location=[[0.0, 1.0, 0.5]], 
                ambient_color=((0.6, 0.6, 0.6),),  # 더 어둡게 조정하여 세부 사항을 강조
                diffuse_color=((0.2, 0.2, 0.2),),  # 밝기를 증가시켜 물체의 색상을 더 돋보이게 함
                specular_color=((0.2, 0.2, 0.2),), # 반사도를 높여 반짝임을 강조
            )
            R, T = get_arctic_side_camera()
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
    
class Renderer_penet():
    def __init__(
        self, 
        device, hand_verts_T, hand_faces, 
        img_size=(256, 256), fps=30,
        is_right=False, 
    ):
        self.device = device
        self.fps = fps
        self.hand_verts_T = hand_verts_T*8
        self.hand_faces = hand_faces

        if is_right:
            R_front, R_back, T = get_right_hand_camera()
        else:
            R_front, R_back, T = get_left_hand_camera()
        camera_front = OrthographicCameras(device=device, R=R_front, T=T)
        camera_back = OrthographicCameras(device=device, R=R_back, T=T)
        
        self.colors = cm.jet
        raster_settings = RasterizationSettings(
            image_size=img_size, 
            blur_radius=0.0, 
            max_faces_per_bin=1, 
            bin_size=0, 
            faces_per_pixel=1, 
        )
        lights_front = PointLights(device=device, location=[[0.0, -1, 0.0]])
        self.renderer_front = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=camera_front, raster_settings=raster_settings),
            shader=HardPhongShader(device=device, cameras=camera_front, lights=lights_front), 
        )
        self.renderer_back = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=camera_back, raster_settings=raster_settings),
            shader=HardPhongShader(device=device, cameras=camera_back), 
        )

    def render_video(self, colors, return_numpy=True):
        '''
            colors: T, N, 3 (tensor)
        '''
        nframes = colors.shape[0]
        verts = self.hand_verts_T.to(self.device).expand(nframes, -1, -1)
        faces = self.hand_faces.to(self.device).unsqueeze(0).expand(nframes, -1, -1)
        mesh = Meshes(verts, faces)
        colors_selected = self.colors(colors.cpu().numpy())[..., :3]
        colors_selected = colors_selected/np.max(colors_selected, axis=2, keepdims=True)
        mesh.textures = TexturesVertex(torch.FloatTensor(colors_selected).to(self.device))
        frames_front = self.renderer_front(mesh)
        frames_back = self.renderer_back(mesh)
        if return_numpy:
            frames_front = frames_front.cpu().numpy()
            frames_back = frames_back.cpu().numpy()
        frames = np.concatenate([frames_front, frames_back], axis=1)
        return frames
    
    
class Renderer_contact():
    def __init__(
        self, 
        device, img_size=512, fps=30, 
    ):
        self.device = device
        self.fps = fps

        # Define camera translation vector T
        T = torch.FloatTensor([0, 0, 0.8]).unsqueeze(0)
        
        # Define rotation matrix R (Identity matrix)
        R = np.eye(3)
        R = torch.from_numpy(R).unsqueeze(0)
        
        # Initialize an orthographic camera
        camera = OrthographicCameras(device=device, R=R, T=T)
        
        # Define colormap for visualization
        self.colors = cm.inferno
        # self.colors = cm.jet  # Alternative colormap option
        
        # Define rasterization settings for rendering
        raster_settings = RasterizationSettings(
            image_size=img_size, 
            blur_radius=0.0, 
            max_faces_per_bin=1, 
            bin_size=0, 
            faces_per_pixel=1, 
        )
        
        # Define a point light source in front of the scene
        lights_front = PointLights(device=device, location=[[0.0, 0.0, -2.0]])

        # Initialize the mesh renderer with rasterizer and shader
        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=camera, raster_settings=raster_settings),
            shader=HardPhongShader(device=device, cameras=camera, lights=lights_front), 
        )

    def propagate_colors(self, points, colors, faces):
        """
        Propagates colors to adjacent vertices in the mesh.
        
        Args:
            points (array): List of vertex coordinates.
            colors (array): Color values assigned to each vertex.
            faces (array): Mesh faces, defined by indices of vertices.
        
        Returns:
            array: Updated colors after propagation.
        """

        # Create adjacency list for vertices
        adjacency_list = [[] for _ in range(len(points))]
        for face in faces:
            for i in range(len(face)):
                adjacency_list[face[i]].extend(face[j] for j in range(len(face)) if j != i)
        
        # Array to track visited vertices
        visited = np.zeros(len(points), dtype=bool)
        
        # Recursive function to propagate colors
        def propagate(point_index):
            # Skip if vertex is already visited
            if visited[point_index]:
                return
            visited[point_index] = True
            
            # Get current vertex color
            current_color = colors[point_index]
            
            # Propagate color to adjacent vertices
            for neighbor in adjacency_list[point_index]:
                if colors[neighbor] == 0:
                    colors[neighbor] = current_color / 2  # Reduce intensity for propagation
                    propagate(neighbor)
        
        # Start propagation for vertices with assigned colors
        for point_index, color in enumerate(colors):
            if color > 0:
                propagate(point_index)
        
        return colors

    def render_video(
        self, 
        verts_y_axis, verts_x_axis, 
        faces, colors, duration, 
        return_numpy=True
    ):
        """
        Renders a video from given vertex positions, faces, and colors.

        Args:
            verts_y_axis (tensor): Vertex positions for the Y-axis view.
            verts_x_axis (tensor): Vertex positions for the X-axis view.
            faces (tensor): Mesh face definitions.
            colors (tensor): Initial vertex colors.
            duration (int): Number of frames in the video.
            return_numpy (bool): Whether to return frames as numpy arrays.

        Returns:
            tuple: Rendered video frames and final colors.
        """

        # Get initial vertex positions from the first frame
        verts = verts_y_axis[0]

        # Propagate colors across the mesh
        colors = self.propagate_colors(verts, colors, faces[0].cpu().numpy())

        # Apply sigmoid normalization to colors
        idx = colors > 0
        ci = colors[idx]
        if len(ci) == 0:
            return colors
        
        a = 0.05  # Sigmoid scaling parameter

        # Fit a sigmoid function to normalize colors
        x1 = min(ci); y1 = a
        x2 = max(ci); y2 = 1 - a
        lna = np.log((1 - y1) / y1)
        lnb = np.log((1 - y2) / y2)
        k = (lnb - lna) / (x1 - x2)
        mu = (x2 * lna - x1 * lnb) / (lna - lnb)

        # Apply the sigmoid function to normalize color intensity
        ci = torch.exp(k * (ci - mu)) / (1 + torch.exp(k * (ci - mu)))
        colors[idx] = ci

        # Expand heatmap values for the video duration
        heatmap_values = colors.unsqueeze(0).expand(duration, -1)

        # Create meshes for Y-axis and X-axis views
        mesh_y_axis = Meshes(verts_y_axis, faces)
        mesh_x_axis = Meshes(verts_x_axis, faces)

        # Map colors using the selected colormap
        # colors_selected = self.colors(colors.cpu().numpy())[..., :3]
        colors_selected = self.colors(heatmap_values.cpu().numpy())[..., :3]

        # Normalize colors
        colors_selected = colors_selected / np.max(colors_selected, axis=2, keepdims=True)

        # Assign textures to the meshes
        mesh_y_axis.textures = TexturesVertex(torch.FloatTensor(colors_selected).to(self.device))
        mesh_x_axis.textures = TexturesVertex(torch.FloatTensor(colors_selected).to(self.device))

        # Render frames for Y-axis and X-axis views
        frames_y_axis = self.renderer(mesh_y_axis)
        frames_x_axis = self.renderer(mesh_x_axis)

        # Convert frames to numpy arrays if required
        if return_numpy:
            frames_y_axis = frames_y_axis.cpu().numpy()
            frames_x_axis = frames_x_axis.cpu().numpy()
        
        # Combine frames from both views
        frames = np.concatenate([frames_y_axis, frames_x_axis], axis=1)
        
        return frames, colors_selected
