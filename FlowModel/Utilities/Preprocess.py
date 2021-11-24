import torch
import numpy as np

def min_max_scaling(x, device):
    '''
    :param x: [bs, t, x_grid, y_grid, dim]
    :return: normalized_x [bs, t, x_grid, y_grid, dim]
    '''
    normalized_x = torch.zeros_like(x).to(device)
    for channel_i in range(x.shape[-1]):
        chl_min = torch.min(x[:,:,:,:,channel_i])
        chl_max = torch.max(x[:,:,:,:,channel_i])
        normalized_x[:,:,:,:,channel_i] = (x[:,:,:,:,channel_i] - chl_min) / (chl_max-chl_min)

    return normalized_x


def seg_traj(si, ti, lookahead, batch_skip):
    """Cut the trajectory and the coresponding time into segments"""
    bs, n_time, n_x_grid, n_y_grid, n_state = si.size()  # _ is the place holder for the number of segments
    seg_list_bs = []
    init_cond_list_bs = []
    for batch_i in range(bs):
        # In order to make the segmentation for each trajectory,
        # we need to split each traj first.
        sii = si[batch_i]  # [n_time, n_agent, n_state]
        tii = ti[batch_i]  # [n_time]
        # save segments into list
        seg_list = []
        init_cond_list = []
        time_seg = tii[0:0+lookahead]  # Data sampling frequency should be the same different batches.
        for j in range(0, n_time-lookahead+1, batch_skip):
            # j = j + i % batch_skip
            seg_traj = sii[j:j+lookahead][:, np.newaxis, :, :, :]
            init_cond = seg_traj[0:1]
            # put the seg info into the list
            seg_list.append(seg_traj)
            init_cond_list.append(init_cond)
        # put all segs together for each batch
        segs_ii = torch.cat(seg_list, dim=1)  # [lookahead, n_segs, n_x_grid, n_y_grid, n_dim]
        seg_list_bs.append(segs_ii)
        init_conds_ii = torch.cat(init_cond_list, dim=1)  # [1(init_cond), n_segs, n_x_grid, n_y_grid, n_dim]
        init_cond_list_bs.append(init_conds_ii)
    # stack segs in all batches
    segs_i = torch.cat(seg_list_bs, dim=1)
    init_conds_i = torch.cat(init_cond_list_bs, dim=1)
    return segs_i, init_conds_i, time_seg