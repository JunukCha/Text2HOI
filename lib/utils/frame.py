import numpy as np

import torch


def get_valid_mask(is_lhand, is_rhand, nframes, valid_nframes):
    valid_mask_lhand = np.zeros((nframes))
    valid_mask_rhand = np.zeros((nframes))
    valid_mask_obj = np.zeros((nframes))
    valid_mask_lhand[:valid_nframes] = 1
    valid_mask_rhand[:valid_nframes] = 1
    valid_mask_obj[:valid_nframes] = 1
    if not is_lhand:
        valid_mask_lhand[:] = 0
    if not is_rhand:
        valid_mask_rhand[:] = 0
    return (
        valid_mask_lhand.astype(np.bool), 
        valid_mask_rhand.astype(np.bool), 
        valid_mask_obj.astype(np.bool)
    )

def get_frame_align_data_format(key, ndata, max_nframes):
    if "beta" in key:
        final_data = np.zeros((ndata, max_nframes, 10), dtype=np.float32)
    elif key == "x_lhand" or key == "x_rhand":
        final_data = np.zeros((ndata, max_nframes, 3+16*6), dtype=np.float32)
    elif key == "j_lhand" or key == "j_rhand":
        final_data = np.zeros((ndata, max_nframes, 21, 3), dtype=np.float32)
    elif key == "obj_alpha":
        final_data = np.zeros((ndata, max_nframes), dtype=np.float32)
    elif key == "x_obj":
        final_data = np.zeros((ndata, max_nframes, 3+6), dtype=np.float32)
    elif key == "x_obj_angle":
        final_data = np.zeros((ndata, max_nframes, 1), dtype=np.float32)
    elif "org" in key:
        final_data = np.zeros((ndata, max_nframes, 3), dtype=np.float32)
    elif "idx" in key or "dist" in key:
        final_data = np.zeros((ndata, max_nframes), dtype=np.float32)
    return final_data

def align_frame(total_dict):
    max_nframes = 0
    value = next(iter(total_dict.values())) # ndata, frames, _
    for _value in value:
        nframes = len(_value)
        if nframes > max_nframes:
            max_nframes = nframes
    ndata = len(value)

    final_dict = {}
    for key, value in total_dict.items():
        final_data = get_frame_align_data_format(key, ndata, max_nframes)    
        for i, data in enumerate(value):
            nframes = len(data)
            if nframes == 0:
                continue
            final_data[i, :nframes] = data
        final_dict[key] = final_data
    return final_dict

def sample_with_window_size(pred, targ, valid_mask, window_size=20, window_step=1):
    ### data: B, T, C
    valid_mask_max_inx = get_mask_max_idx(valid_mask)
    valid_mask_max_inx_w = valid_mask_max_inx-(window_size*window_step)
    valid_mask_max_inx_w[valid_mask_max_inx_w<0] = 0

    sampled_f_idx = sample_frame_index(valid_mask_max_inx_w, window_size, window_step)
    sampled_pred = torch.gather(pred, 1, sampled_f_idx.unsqueeze(2).expand(-1, -1, pred.shape[2]))
    sampled_targ = torch.gather(targ, 1, sampled_f_idx.unsqueeze(2).expand(-1, -1, targ.shape[2]))
    return sampled_pred, sampled_targ
    
def sample_frame_index(frame_indices, window_size, window_step):
    sampled_indices = torch.rand(len(frame_indices), device=frame_indices.device)*frame_indices
    sampled_indices = sampled_indices.int()
    sampled_indices = sampled_indices.reshape(-1, 1)
    sampled_indices = torch.arange(
            start=0,
            end=window_step*window_size, 
            step=window_step,
                device=sampled_indices.device
        ).unsqueeze(0).expand(sampled_indices.shape[0], -1) \
        + sampled_indices
    return sampled_indices

def get_mask_max_idx(mask):
    '''
    Find the indices of the last occurrences of True in a boolean mask.

    Parameters:
        mask (torch.Tensor): A boolean tensor representing a mask.

    Returns:
        torch.Tensor: A 1-dimensional tensor containing the indices of the last occurrences of True in the input mask.
    '''
    
    # Compute the transitions from True to False in the mask
    true_to_false_transitions = mask.int().diff(dim=1)
    
    # Find the indices of the last True occurrences in each row
    first_false_indices = true_to_false_transitions.argmin(dim=1)

    # Set indices to the maximum frame idx 
    # if there are no False occurrences in a row
    first_false_indices[
        ~true_to_false_transitions.any(dim=1)
        ] = mask.size(1)-1
    return first_false_indices
