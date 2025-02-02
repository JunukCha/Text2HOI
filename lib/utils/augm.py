import numpy as np

import torch

from lib.utils.rot import (
    get_rotmat_x, 
    get_rotmat_y, 
    get_rotmat_z, 
    rot6d_to_rotmat, 
    rotmat_to_rot6d, 
)
from lib.utils.proc import proc_numpy

def augmentation(
    X, 
    r_rot=[5, 20, 5], r_trans=[0.0, 0.0, 0.0],
    hand_org=None, 
    aug_rotmat=None,
    aug_trans=None,
):
    """
        x: x_obj or x_hand (nframes, 9 or 99)
        r_rot: range of random rotation
        r_trans: range of random translation
    """
    nframes = X.shape[0]
    trans = torch.FloatTensor(X[:, :3])
    rot6d = torch.FloatTensor(X[:, 3:9])

    rotmat = rot6d_to_rotmat(rot6d)
    if hand_org is not None:
        trans += hand_org
    trans = trans.unsqueeze(2)
    extmat = torch.cat([rotmat, trans], dim=2)
    homo = torch.FloatTensor([0, 0, 0, 1]).unsqueeze(0).unsqueeze(1)
    homo = homo.expand(nframes, -1, -1)
    extmat = torch.cat([extmat, homo], dim=1)

    if aug_rotmat is None:
        aug_rotmat = get_augm_rot(*r_rot)
    if aug_trans is None:
        aug_trans = get_augm_trans(*r_trans)
    aug_extmat = torch.cat([aug_rotmat, aug_trans], dim=1)
    aug_homo = torch.FloatTensor([0, 0, 0, 1]).unsqueeze(0)
    aug_extmat = torch.cat([aug_extmat, aug_homo], dim=0)

    augmented_extmat = torch.einsum("ij,fjk->fik", aug_extmat, extmat)
    augmented_trans = augmented_extmat[:, :3, 3]
    augmented_rotmat = augmented_extmat[:, :3, :3]
    augmented_rot6d = rotmat_to_rot6d(augmented_rotmat)
    augmented_X = torch.cat([augmented_trans, augmented_rot6d], dim=1)
    augmented_X = augmented_X.numpy()
    X[..., :9] = augmented_X
    if hand_org is not None:
        X[..., :3] -= hand_org
    return X, aug_rotmat, aug_trans

def augmentation_joints(
    X, 
    r_rot=[5, 20, 5], r_trans=[0.0, 0.0, 0.0],
    aug_rotmat=None, 
    aug_trans=None, 
):
    """
        x: x_obj or x_hand (nframes, 9 or 99)
        r_rot: range of random rotation
        r_trans: range of random translation
    """
    nframes, njoints = X.shape[:2]

    hand_org = X[:, :1].copy()
    X_root_aligned = X-hand_org
    homo = np.ones([1])[np.newaxis, np.newaxis]
    homo = np.tile(homo, (nframes, njoints, 1))
    X_homo = np.concatenate([X_root_aligned, homo], axis=2)

    if aug_rotmat is None:
        aug_rotmat = get_augm_rot(*r_rot)
    if aug_trans is None:
        aug_trans = get_augm_trans(*r_trans)
    aug_rotmat = proc_numpy(aug_rotmat)
    aug_trans = proc_numpy(aug_trans)
    aug_extmat = np.concatenate([aug_rotmat, aug_trans], axis=1)
    aug_homo = np.array([0, 0, 0, 1])[np.newaxis]
    aug_extmat = np.concatenate([aug_extmat, aug_homo], axis=0)

    augmented_X_homo = np.einsum("ij,fkj->fki", aug_extmat, X_homo)
    augmented_X_root_aligned = augmented_X_homo[..., :3]
    augmented_X = augmented_X_root_aligned + hand_org
    return augmented_X, aug_rotmat, aug_trans

def get_augm_rot(r_x_rot, r_y_rot, r_z_rot):
    if r_x_rot != 0:
        x_angle = np.random.randint(-r_x_rot, r_x_rot)
    else:
        x_angle = 0
    if r_y_rot != 0:
        y_angle = np.random.randint(-r_y_rot, r_y_rot)
    else:
        y_angle = 0
    if r_z_rot != 0:
        z_angle = np.random.randint(-r_z_rot, r_z_rot)
    else:
        z_angle = 0

    x_radians = np.pi*(x_angle/180)
    x_rotmat = torch.FloatTensor(get_rotmat_x(x_radians))

    y_radians = np.pi*(y_angle/180)
    y_rotmat = torch.FloatTensor(get_rotmat_y(y_radians))

    z_radians = np.pi*(z_angle/180)
    z_rotmat = torch.FloatTensor(get_rotmat_z(z_radians))
    aug_rotmat = torch.matmul(torch.matmul(z_rotmat, y_rotmat), x_rotmat)
    return aug_rotmat

def get_augm_trans(x_trans, y_trans, z_trans):
    aug_x_trans = 2*x_trans*torch.rand(1)-x_trans # [-x_trans, x_trans]
    aug_y_trans = 2*y_trans*torch.rand(1)-y_trans # [-y_trans, y_trans]
    aug_z_trans = 2*z_trans*torch.rand(1)-z_trans # [-z_trans, z_trans]
    aug_trans = torch.stack([aug_x_trans, aug_y_trans, aug_z_trans])
    return aug_trans

def get_augm_scale(scale=0.2):
    aug_scale = 2*scale*torch.rand(1)+(1-scale) # 1-scale ~ 1+scale
    return aug_scale