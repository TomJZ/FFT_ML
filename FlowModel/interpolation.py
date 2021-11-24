import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

data_folder = './data/inference'
s = np.load(os.path.join(data_folder, "forecast_data_from_nov_22.npy"))
s_lat_lon = s[:,:,:,:2]
s_uv = s[:,:,:,2:]
t = np.load(os.path.join(data_folder, "forecast_times_from_nov_22.npy"))

# interpolate the uv
s_u = s[:, :, :, 2]
s_v = s[:, :, :, 3]
t_len, x_grid, y_grid, uv_dim = s_uv.shape

t_interp = np.arange(193)
s_uv_interp = np.zeros((len(t_interp), x_grid, y_grid, 2))

for i in tqdm(range(x_grid)):
    for j in range(y_grid):
        s_ij_u = s_uv[:, i, j, 0]
        s_ij_v = s_uv[:, i, j, 1]

        s_ij_u_interp = np.interp(t_interp, t, s_ij_u)[:,np.newaxis]
        s_ij_v_interp = np.interp(t_interp, t, s_ij_v)[:,np.newaxis]
        s_ij_uv_interp = np.concatenate((s_ij_u_interp, s_ij_v_interp), axis=1)
        # fill in
        s_uv_interp[:, i, j, :] = s_ij_uv_interp

        # visualize
        # plt.plot(t, s_ij_u, "o", label="original")
        # plt.plot(t_interp, s_ij_u_interp, "x", label='interp')
        # plt.legend()
        # plt.show()

# fill in the lat, lon
s_lat_lon_repeat = np.repeat(s_lat_lon[0:1, :, :, :2], len(t_interp), axis=0)
s_interp = np.concatenate((s_lat_lon_repeat, s_uv_interp), axis=-1)
np.save(os.path.join(data_folder, "forecast_data_from_nov_22_interp.npy"), s_interp.astype(np.float32))
np.save(os.path.join(data_folder, "forecast_times_from_nov_22_interp.npy"), s_interp.astype(np.int64))

print()