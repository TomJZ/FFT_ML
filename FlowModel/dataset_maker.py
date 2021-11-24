# Format the dataset. [t, bs(1), n_agents, dim]
import numpy as np
import scipy.io
import os

def mat2npy(load_folder_name, save_folder_name, mat_file_name):
    file_name = mat_file_name.split('.')[0]
    mat_file_name = os.path.join(load_folder_name, mat_file_name)
    mat_file = scipy.io.loadmat(mat_file_name)
    U = mat_file['U']  # x velocity on each grid over time [x_grid, y_grid, t]
    V = mat_file['V']  # y velocity on each grid over time [x_grid, y_grid, t]
    xx = mat_file['x']  # coordinate of x-meshgrid [x_grid, y_grid]
    yy = mat_file['y']  # coordinate of y-meshgrid [x_grid, y_grid]
    t = mat_file['t_vect'].flatten()
    t_len = len(t)
    xx_rep = np.repeat(xx[:, :, np.newaxis], t_len, axis=-1)  # [x_grid, y_grid, t]
    yy_rep = np.repeat(yy[:, :, np.newaxis], t_len, axis=-1)  # [x_grid, y_grid, t]

    # flat the grids for U,V,xx,yy
    U_flat = np.transpose(U, (2, 0, 1))[:, :, :, np.newaxis]
    V_flat = np.transpose(V, (2, 0, 1))[:, :, :, np.newaxis]
    xx_flat = np.transpose(xx_rep, (2, 0, 1))[:, :, :, np.newaxis]
    yy_flat = np.transpose(yy_rep, (2, 0, 1))[:, :, :, np.newaxis]

    # format the final dataset for 2D grid
    X = np.concatenate((U_flat, V_flat, xx_flat, yy_flat), axis=-1)
    # save npz file
    np.savez(os.path.join(save_folder_name, file_name), t=t, s=X)


if __name__ == '__main__':
    # Auto-load data mat files
    load_folder = './data'
    save_folder = './data/npz_data'
    # .mat file list
    mat_files = [mat_file for mat_file in os.listdir(load_folder) if mat_file.endswith(".mat")]

    # format the dataset and save
    for mat_file in mat_files:
        mat2npy(load_folder, save_folder, mat_file)


