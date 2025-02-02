import time
import numpy as np
import json

from torch.utils.data import Dataset

from lib.models.object import build_object_model
from lib.utils.frame import get_valid_mask
from lib.utils.augm import (
    augmentation, 
    augmentation_joints, 
    get_augm_rot, 
    get_augm_scale, 
)
from lib.utils.proc_h2o import process_text
from lib.utils.proc import (
    get_contact_map, 
    pc_normalize, 
    process_dist_map, 
    select_from_groups, 
)


class SequenceH2O(Dataset): # point encoder
    def __init__(
        self, 
        data_path, 
        data_obj_pc_path, 
        text_json, 
        max_nframes, 
        data_ratio=1.0, 
        augm=False, 
        **kwargs
    ):
        super().__init__()

        self.data_path = data_path
        self.data_obj_pc_path = data_obj_pc_path
        self.max_nframes = max_nframes
        self.data_ratio = data_ratio
        self.augm = augm

        start_time = time.time()
        print("Start to read data h2o!!!")
        with np.load(data_path, allow_pickle=True) as data:
            self.is_lhand = data["is_lhand"]
            self.is_rhand = data["is_rhand"]
            self.action_name = data["action_name"]
            self.nframes = data["nframes"]
        
        with open(text_json, "r") as f:
            self.text_description = json.load(f)

        self.object_model = build_object_model(data_obj_pc_path)

        print("Finish to read data h2o!!!", f"{time.time()-start_time:.2f}s")
        print(f"length of data: {self.__len__()}")
        
    def __len__(self):
        return int(len(self.action_name)*self.data_ratio)
    
    def __getitem__(self, index):
        item = {}
        
        nframes = self.nframes[index]
        if nframes > self.max_nframes:
            nframes = self.max_nframes
        seq_time = np.array([nframes/150], dtype=np.float32)
        if self.augm:
            augm_scale = 1-(2*np.random.rand()*0.05-0.05)
            seq_time *= augm_scale
            if seq_time > 1:
                seq_time = np.array([1.0], dtype=np.float32)
        item["seq_time"] = seq_time
            
        action_name = self.action_name[index]
        is_lhand = self.is_lhand[index]
        is_rhand = self.is_rhand[index]

        text = process_text(
            action_name, 
            is_lhand, is_rhand,
            self.text_description, 
        )
        item["text"] = text
        return item
    
    
class ContactH2O(Dataset): # point encoder
    def __init__(
        self, 
        data_path, 
        data_obj_pc_path, 
        text_json, 
        max_nframes, 
        obj_name, 
        data_ratio=1.0, 
        augm=False, 
        **kwargs
    ):
        super().__init__()

        self.data_path = data_path
        self.data_obj_pc_path = data_obj_pc_path
        self.max_nframes = max_nframes
        self.object_name = obj_name
        self.data_ratio = data_ratio
        self.augm = augm

        start_time = time.time()
        print("Start to read data h2o!!!")
        with np.load(data_path, allow_pickle=True) as data:
            self.object_idx = data["object_idx"]
            self.lcov_idx = data["lcov_idx"] # left contact object verts idx
            self.rcov_idx = data["rcov_idx"] # right contact object verts idx
            self.is_lhand = data["is_lhand"]
            self.is_rhand = data["is_rhand"]
            self.action_name = data["action_name"]
        with open(text_json, "r") as f:
            self.text_description = json.load(f)

        self.object_model = build_object_model(data_obj_pc_path)

        print("Finish to read data h2o!!!", f"{time.time()-start_time:.2f}s")

    def __len__(self):
        return int(len(self.action_name)*self.data_ratio)
    
    def __getitem__(self, index):
        item = {}
        
        is_lhand = self.is_lhand[index]
        is_rhand = self.is_rhand[index]
        item["is_lhand"] = is_lhand
        item["is_rhand"] = is_rhand
        
        action_name = self.action_name[index]
        
        text = process_text(
            action_name, 
            is_lhand, is_rhand,
            self.text_description, 
        )
        item["text"] = text
        
        object_idx = self.object_idx[index]
        object_name = self.object_name[object_idx]
        item["action_name"] = action_name
        _, obj_pc, _, _  = self.object_model(object_name)
        
        if self.augm:
            aug_scale = get_augm_scale(0.2).numpy()
            obj_pc = obj_pc*aug_scale
            aug_rotmat = get_augm_rot(15, 15, 15).numpy()
            obj_pc = np.einsum("ij,kj->ki", aug_rotmat, obj_pc)
        normalized_obj_pc, _, obj_norm_scale = pc_normalize(obj_pc, return_params=True)
        item["normalized_obj_pc"] = normalized_obj_pc
        item["obj_scale"] = obj_norm_scale

        lcov_idx = self.lcov_idx[index]
        rcov_idx = self.rcov_idx[index]
        lcov_map = get_contact_map(lcov_idx, 1024, is_lhand)
        rcov_map = get_contact_map(rcov_idx, 1024, is_rhand)
        cov_map = (lcov_map+rcov_map)>0
        item["cov_map"] = cov_map.astype(np.float32)
        return item
    
    
class MotionH2O(Dataset):
    def __init__(
        self, 
        data_path, 
        data_obj_pc_path, 
        text_json, 
        max_nframes, 
        obj_name, 
        data_ratio=1.0, 
        augm=False, 
        **kwargs
    ):
        super().__init__()

        self.data_path = data_path
        self.data_obj_pc_path = data_obj_pc_path
        self.max_nframes = max_nframes
        self.object_name = obj_name
        self.data_ratio = data_ratio
        self.augm = augm

        start_time = time.time()
        print("Start to read data h2o!!!")
        with np.load(data_path, allow_pickle=True) as data:
            self.object_idx = data["object_idx"]
            self.x_lhand = data["x_lhand"]
            self.x_rhand = data["x_rhand"]
            self.x_obj = data["x_obj"]
            self.lhand_org = data["lhand_org"]
            self.rhand_org = data["rhand_org"]
            self.lcf_idx = data["lcf_idx"] # left hand contact frame idx
            self.lcov_idx = data["lcov_idx"] # left contact object verts idx
            self.lchj_idx = data["lchj_idx"] # left contact hand joints idx
            self.ldist_value = data["ldist_value"]
            self.rcf_idx = data["rcf_idx"] # right hand contact frame idx
            self.rcov_idx = data["rcov_idx"] # right contact object verts idx
            self.rchj_idx = data["rchj_idx"] # right contact hand joints idx
            self.rdist_value = data["rdist_value"]
            self.is_lhand = data["is_lhand"]
            self.is_rhand = data["is_rhand"]
            self.action_name = data["action_name"]
            self.nframes = data["nframes"]

        with open(text_json, "r") as f:
            self.text_description = json.load(f)

        self.object_model = build_object_model(data_obj_pc_path)

        print("Finish to read data h2o!!!", f"{time.time()-start_time:.2f}s")

    def __len__(self):
        return int(len(self.action_name)*self.data_ratio)
    
    def __getitem__(self, index):
        item = {}

        nframes = self.nframes[index]
        is_lhand = self.is_lhand[index]
        is_rhand = self.is_rhand[index]
        
        item["is_lhand"] = is_lhand
        item["is_rhand"] = is_rhand

        if nframes > self.max_nframes:
            init_frame = np.random.randint(0, nframes-self.max_nframes)
            nframes = self.max_nframes
        else:
            init_frame = 0
        
        x_obj = self.x_obj[index][init_frame:init_frame+self.max_nframes]
        if self.augm:
            x_obj[:nframes], aug_rotmat, aug_trans = augmentation(x_obj[:nframes])
        item["x_obj"] = x_obj

        if is_lhand:
            x_lhand = self.x_lhand[index][init_frame:init_frame+self.max_nframes]
            if self.augm:
                lhand_org = self.lhand_org[index][init_frame:init_frame+nframes]
                x_lhand[:nframes], _, _ \
                    = augmentation(
                        x_lhand[:nframes], 
                        hand_org=lhand_org, 
                        aug_rotmat=aug_rotmat, 
                        aug_trans=aug_trans
                    )
        else:
            x_lhand = np.zeros((150, 99), dtype=np.float32)
        
        item["x_lhand"] = x_lhand

        if is_rhand:
            x_rhand = self.x_rhand[index][init_frame:init_frame+self.max_nframes]
            if self.augm:
                rhand_org = self.rhand_org[index][init_frame:init_frame+nframes]
                x_rhand[:nframes], _, _ \
                    = augmentation(
                        x_rhand[:nframes], 
                        hand_org=rhand_org, 
                        aug_rotmat=aug_rotmat, 
                        aug_trans=aug_trans
                    )
        else:
            x_rhand = np.zeros((150, 99), dtype=np.float32)
        item["x_rhand"] = x_rhand

        action_name = self.action_name[index]
        max_nframes = self.max_nframes
        
        valid_mask_lhand, valid_mask_rhand, valid_mask_obj = get_valid_mask(is_lhand, is_rhand, max_nframes, nframes) # max_nframes: 2x frames
        item["valid_mask_lhand"] = valid_mask_lhand
        item["valid_mask_rhand"] = valid_mask_rhand
        item["valid_mask_obj"] = valid_mask_obj

        text = process_text(
            action_name, 
            is_lhand, is_rhand, 
            self.text_description, 
        )
        item["text"] = text
        
        object_idx = self.object_idx[index]
        object_name = self.object_name[object_idx]
        _, obj_pc, obj_pc_normal, _ = self.object_model(object_name)

        normalized_obj_pc, obj_norm_cent, obj_norm_scale = pc_normalize(obj_pc, return_params=True)
        item["obj_pc"] = obj_pc
        item["normalized_obj_pc"] = normalized_obj_pc
        item["obj_pc_normal"] = obj_pc_normal
        item["obj_cent"] = obj_norm_cent
        item["obj_scale"] = obj_norm_scale

        lcf_idx = self.lcf_idx[index] 
        lcov_idx = self.lcov_idx[index]
        lchj_idx = self.lchj_idx[index]
        ldist_value = self.ldist_value[index]

        ldist_map = process_dist_map(
            self.max_nframes, 
            init_frame, lcf_idx, 
            lcov_idx, lchj_idx, 
            ldist_value, is_lhand)
        item["ldist_map"] = ldist_map

        rcf_idx = self.rcf_idx[index]
        rcov_idx = self.rcov_idx[index]
        rchj_idx = self.rchj_idx[index]
        rdist_value = self.rdist_value[index]
        
        rdist_map = process_dist_map(
            self.max_nframes, 
            init_frame, rcf_idx, 
            rcov_idx, rchj_idx, 
            rdist_value, is_rhand)
        item["rdist_map"] = rdist_map
        
        lcov_map = get_contact_map(lcov_idx, 1024, is_lhand)
        rcov_map = get_contact_map(rcov_idx, 1024, is_rhand)
        cov_map = (lcov_map+rcov_map)>0
        item["cov_map"] = cov_map.astype(np.float32)
        return item