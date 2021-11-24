import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import torch
from matplotlib import animation
from matplotlib.lines import Line2D

def vis_vf_large(vf, drifter_grid, name='Drifter_Flow', anim_folder='./VFAnim'):
    '''

    :param vf: [t_vis, n_x_grid, n_y_grid, 4(U,V)]
    :param drifter_grid: [t_vis, drifter_id, 2]
    :return:
    '''
    trailing_buf_size = 144
    # convert the tensor to numpy array
    gt_vf = vf
    # gt U&V, pred U&V
    gt_U = gt_vf[:, :, :, 2]
    gt_V = gt_vf[:, :, :, 3]
    x = drifter_grid[:,:,0]
    y = drifter_grid[:,:,1]
    AGENT_COUNT = x.shape[1]

    ## Animate
    fig = plt.figure()
    fig.set_size_inches(30, 10)
    ax1 = plt.subplot(111)
    # put on a meshgrid
    xx, yy = np.meshgrid(np.linspace(0,500, 11), np.linspace(0,1000,11))
    ii, jj = np.meshgrid(np.linspace(0,1000,11), np.linspace(0,500, 11))

    # fig, ax = plt.subplots()
    # ax1 = plt.subplot()
    ## init the animation frame
    gt_U0, gt_V0 = gt_U[0, :, :], gt_V[0, :, :]
    drifter0 = drifter_grid[0,:,:]
    gt_Q = ax1.imshow(gt_U0, cmap='ocean', interpolation='nearest', origin='lower')

    # drifter id
    for j in range(AGENT_COUNT):
        ax1.text(drifter0[j,1], drifter0[j,0], str(j), color='yellow')

    drifter = ax1.plot(drifter0[:,1], drifter0[:,0], marker='o', markersize=5, linestyle='None', color='red')

    trail_p = [Line2D([x[0][0]], [y[0][0]]) for i in range(AGENT_COUNT)]
    for j in range(AGENT_COUNT):
        trail_p[j], = ax1.plot([y[0,j]],
                               [x[0,j]],
                               c='red', marker=None, lw=1)


    meshgrid1 = ax1.plot(yy,xx, color="green", lw=3)
    meshgrid2 = ax1.plot(ii,jj, color="green", lw=3)



    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend()
    plt.title(name)
    # plt.savefig("test.png")

    def update_quiver(ti):
        # print(ti)
        gt_Ui, gt_Vi = gt_U[ti, :, :], gt_V[ti, :, :]
        drifteri = drifter_grid[ti, :, :]
        gt_Q.set_data(gt_Ui)
        drifter[0].set_data(drifteri[:,1], drifteri[:,0])
        trail_offset = trailing_buf_size if ti > trailing_buf_size else ti
        for j in range(AGENT_COUNT):
            trail_p[j].set_data(y[ti - trail_offset:ti, j], x[ti - trail_offset:ti, j])
        return gt_Q

    # you need to set blit=False, or the first set of arrows never gets
    # cleared on subsequent frames
    anim = animation.FuncAnimation(fig, update_quiver, frames=gt_U.shape[0])
    save_name = os.path.join(anim_folder, name+'.mp4')
    anim.save(save_name, writer=animation.FFMpegWriter(fps=10))


if __name__ == '__main__':

    ## get the training data
    train_data_folder = './data/train'
    eval_data_folder = './data/eval'
    forecast_data_folder = './data/forecast'

    # load flow
    flow_data_eval = np.load(os.path.join(eval_data_folder, "noaa_nowcast_data_nov_2_to_nov_19.npy"))[-96:-24]
    flow_data_eval[flow_data_eval > 10e10] = 0
    # load drifter
    drifter_data_day14 = np.load("./drifter_data/processed_data/day14_pos.npz", allow_pickle=True)['pos']
    drifter_data_day15 = np.load("./drifter_data/processed_data/day15_pos.npz", allow_pickle=True)['pos']
    drifter_data_day16 = np.load("./drifter_data/processed_data/day16_pos.npz", allow_pickle=True)['pos']
    drifter_data = np.concatenate((drifter_data_day14,
                                   drifter_data_day15,
                                   drifter_data_day16), axis=0)
    # filter None to nan
    drifter_data[drifter_data==None] = 0
    # filter missing data
    search_len = 144
    for t in tqdm(range(drifter_data.shape[0])):
        for k in range(drifter_data.shape[1]):
            if drifter_data[t,k,0] == 0:
                for search_i in range(1, search_len):
                    if drifter_data[t+search_i,k,0] != 0:
                        drifter_data[t, k, 0] = drifter_data[t+search_i,k,0]
                        break
                    elif drifter_data[t-search_i,k,0] != 0:
                        drifter_data[t, k, 0] = drifter_data[t-search_i,k,0]
                        break

            if drifter_data[t, k, 1] == 0:
                for search_i in range(1, search_len):
                    if drifter_data[t + search_i, k, 1] != 0:
                        drifter_data[t, k, 1] = drifter_data[t + search_i, k, 1]
                        break
                    elif drifter_data[t - search_i, k, 1] != 0:
                        drifter_data[t, k, 1] = drifter_data[t - search_i, k, 1]
                        break

    # select step2
    drifter_data = drifter_data[0::2]

    # lat&lon list
    lat_list = flow_data_eval[0,:,0,0]  # [500,]
    lon_list = flow_data_eval[0,0,:,1]  # [800,]

    # drift lat&lon convert to grid
    t, n, _ = drifter_data.shape
    drifter_flat_lat = drifter_data.reshape(-1, 2)[:, 0]  # [t*n,]
    drifter_flat_lon = drifter_data.reshape(-1, 2)[:, 1] + 360  # [t*n,], with 360 offset


    # lat index
    lat_diff = (drifter_flat_lat[:,np.newaxis] - lat_list[np.newaxis, :])
    lat_diff[lat_diff>0] = 1
    lat_diff[lat_diff<0] = 0
    lat_index = torch.argmin(torch.tensor(lat_diff.astype(float)), dim=1)
    # lon index
    lon_diff = (drifter_flat_lon[:, np.newaxis] - lon_list[np.newaxis, :])
    lon_diff[lon_diff>0] = 1
    lon_diff[lon_diff<0] = 0
    lon_index = torch.argmin(torch.tensor(lon_diff.astype(float)), dim=1)

    drifter_grid = torch.cat([lat_index.unsqueeze(-1), lon_index.unsqueeze(-1)], dim=-1).numpy()
    drifter_grid = drifter_grid.reshape(t,n,2)

    vis_vf_large(flow_data_eval, drifter_grid)


print()
