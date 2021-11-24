import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib import animation
import os


def flow_animation(xx, yy, U, V, save_name='test'):
    fig, ax = plt.subplots()
    # Animation
    ## init the animation frame
    U0, V0 = U[:, :, 0], V[:, :, 0]
    Q = ax.quiver(xx, yy, U0, V0)
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.title(save_name)
    def update_quiver(ti):
        # print(ti)
        Ui, Vi = U[:, :, ti], V[:, :, ti]
        Q.set_UVC(Ui, Vi)
        return Q

    # you need to set blit=False, or the first set of arrows never gets
    # cleared on subsequent frames
    anim = animation.FuncAnimation(fig, update_quiver, frames=U.shape[-1])
    save_name = save_name + '.mp4'
    anim.save(save_name, writer=animation.FFMpegWriter(fps=10))



def data2FlowAnim(mat_file, save_name='test'):
    # Visualize the 2D vector field
    U = mat_file['U']  # x velocity on each grid over time [x_grid, y_grid, t]
    V = mat_file['V']  # y velocity on each grid over time [x_grid, y_grid, t]
    xx = mat_file['x']  # coordinate of x-meshgrid
    yy = mat_file['y']  # coordinate of y-meshgrid
    t_len = U.shape[-1]  # total time steps --> int

    # flow animation
    flow_animation(xx, yy, U, V, save_name)


# Auto-load data mat files
load_folder = './data'
anim_save_folder = './VFAnim'
# .mat file list
mat_files = [mat_file for mat_file in os.listdir(load_folder) if mat_file.endswith(".mat")]

# loop the mat file and animate
for mat_file in mat_files:
    print('Animate' + mat_file + 'start...')
    load_path = os.path.join(load_folder, mat_file)
    data_name = mat_file.split('.')[0]
    save_path = os.path.join(anim_save_folder, data_name)
    # load the mat file
    mat = scipy.io.loadmat(load_path)
    # Animate the flow vector field.
    data2FlowAnim(mat, save_path)

# # # load mat file
# SB_mat = scipy.io.loadmat('./data/SB_data_1_17_2016_to_1_18_2016.mat')
# # MB_mat = scipy.io.loadmat('./data/MB_data_12122014.mat')
# # NP_mat = scipy.io.loadmat('./data/NP_data_06_1_2016_to_08_31_2016.mat')
# #
# # # Animate the flow vector field.
# data2FlowAnim(SB_mat, 'SB Flow Vecter Field')
# # data2FlowAnim(MB_mat, 'MB Flow Vecter Field')
# # data2FlowAnim(NP_mat, 'NP Flow Vecter Field')
#
# # format dataset [t, 1, n_agent, dim]
