from read_data import *
from Utilities.Plotters import *
from knn_pca import *
from fft_integrator import *

import numpy as np
import geopy
import tqdm


"""
Loading Ocean Flow Data
"""
flow_data_path = "Data/flow/forecast_data_from_nov_2.npy"
flow_time_path = "Data/flow/forecast_times_from_nov_2.npy"
flow_data_all = read_data(flow_data_path)
flow_time = read_data(flow_time_path)

flow_data = flow_data_all
flow_data_all[flow_data_all > 1e20] = 0
flow_data[flow_data == 0] = None
template = np.copy(flow_data)

print("Flow data has size: ", flow_data.shape)
print(np.max(flow_data))

"""
Loading Drifter Location Data
"""
drifter_data_all = read_all_drifter_data()

"""
KNN PCA
"""
drifter_id = 11
flow_snapshot = 2
k_neighbor = 10
drifter_ts = drifter_data_all[:, drifter_id, :]  # the time series path of a single drifter
drifter_latlon = drifter_data_all[flow_snapshot, drifter_id, :].reshape([1, -1])  # the location of a single drifter at one snapshot
site_lat = drifter_latlon[:, 0]
site_lon = drifter_latlon[:, 1]

ind_flat, ind_2d = knn_on_flow(drifter_latlon, flow_data[flow_snapshot, :, :, 0:2], k_neighbor)
recon_vel = knn_then_pca(drifter_latlon, flow_data[flow_snapshot], k_neighbor)
print("reconstructed flow is: ", recon_vel)

# visualize_flow(flow_data, 2)
visualize_all_drifter(drifter_data_all, flow_data)
# visualize_pca(flow_data, ind_flat, flow_snapshot, recon_vel, drifter_ts=drifter_ts)


drifter_curr_pos = drifter_data_all[0, drifter_id, :]
drifter_traj = [drifter_curr_pos]
step_size = 3600  # every hour is 3600 seconds, corresponding to the flow field sampling freq
for ss in tqdm.tqdm(range(73)):
    print(drifter_curr_pos.shape)
    recon_vel = knn_then_pca(drifter_curr_pos, flow_data[ss], k_neighbor)
    u_vel = recon_vel[0]
    v_vel = recon_vel[1]
    u_vel_in_deg = geopy.units.degrees(arcminutes=geopy.units.nautical(meters=u_vel))
    v_vel_in_deg = geopy.units.degrees(arcminutes=geopy.units.nautical(meters=v_vel))
    drifter_curr_pos = drifter_curr_pos + np.array([u_vel_in_deg, v_vel_in_deg]) * step_size
    drifter_traj.append(drifter_curr_pos)

with open('Data/predictions/drifter_11_pred_traj.npy', 'wb') as f:
    np.save(f, np.array(drifter_traj))




