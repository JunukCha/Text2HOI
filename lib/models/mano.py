from typing import Optional
from smplx import MANO
import torch
from lib.utils.file import load_config
from dataclasses import dataclass, fields
from typing import NewType, Optional

config = load_config("configs/mano.yaml")
mano_config = config.mano
MODEL_DIR = mano_config.root
SKELETONS = mano_config.skeletons
SKELETONS_W_TIP = mano_config.skeletons_w_tip
left_hand_mean = mano_config.left_hand_mean
right_hand_mean = mano_config.right_hand_mean

Tensor = NewType('Tensor', torch.Tensor)

@dataclass
class ModelOutput_C:
    vertices: Optional[Tensor] = None
    joints: Optional[Tensor] = None
    joints_w_tip: Optional[Tensor] = None
    full_pose: Optional[Tensor] = None
    global_orient: Optional[Tensor] = None
    transl: Optional[Tensor] = None
    v_shaped: Optional[Tensor] = None

    def __getitem__(self, key):
        return getattr(self, key)

    def get(self, key, default=None):
        return getattr(self, key, default)

    def __iter__(self):
        return self.keys()

    def keys(self):
        keys = [t.name for t in fields(self)]
        return iter(keys)

    def values(self):
        values = [getattr(self, t.name) for t in fields(self)]
        return iter(values)

    def items(self):
        data = [(t.name, getattr(self, t.name)) for t in fields(self)]
        return iter(data)


@dataclass
class MANOOutput_C(ModelOutput_C):
    betas: Optional[Tensor] = None
    hand_pose: Optional[Tensor] = None
    skeletons: list = None
    skeletons_w_tip: list = None


class MANO_C(MANO):
    def __init__(
        self, 
        model_path, 
        is_rhand=True, 
        data_struct=None, 
        create_hand_pose=True, 
        hand_pose=None, 
        use_pca=True, 
        num_pca_comps=6, 
        flat_hand_mean=False, 
        batch_size=1, 
        dtype=torch.float32, 
        vertex_ids=None, 
        use_compressed=True, 
        ext='pkl', 
        **kwargs):
        super().__init__(
            model_path, is_rhand, data_struct, 
            create_hand_pose, hand_pose, use_pca, 
            num_pca_comps, flat_hand_mean, 
            batch_size, dtype, vertex_ids, 
            use_compressed, ext, **kwargs)
        
    def forward(
        self, 
        betas=None, 
        global_orient=None, 
        hand_pose=None, 
        transl=None, 
        return_verts=True, 
        return_full_pose=False, 
        **kwargs) -> MANOOutput_C:
        super_output = super().forward(betas, global_orient, hand_pose, transl, return_verts, return_full_pose, **kwargs)
        thumb_tip = super_output.vertices[:, 745].unsqueeze(1)
        index_tip = super_output.vertices[:, 317].unsqueeze(1)
        middle_tip = super_output.vertices[:, 445].unsqueeze(1)
        ring_tip = super_output.vertices[:, 556].unsqueeze(1)
        pinky_tip = super_output.vertices[:, 673].unsqueeze(1)
        
        joints_w_tip = super_output.joints.clone()
        joints_w_tip = torch.cat([
                joints_w_tip, index_tip, middle_tip, 
                pinky_tip, ring_tip, thumb_tip, 
            ], dim=1)

        output = MANOOutput_C(
            **super_output, 
            joints_w_tip=joints_w_tip, 
            skeletons=SKELETONS,
            skeletons_w_tip=SKELETONS_W_TIP,
        )
        return output

def build_mano_aa(is_rhand, create_transl=False, flat_hand=False):
    return MANO_C(
        MODEL_DIR,
        create_transl=create_transl,
        use_pca=False,
        flat_hand_mean=flat_hand,
        is_rhand=is_rhand,
    )