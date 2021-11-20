from Utilities.Plotters import *
from read_data import *

"""
Loading Drifter Prediction Data
"""
with open('Data/predictions/drifter_11_pred_traj.npy', 'rb') as f:
    pred_traj = np.load(f)

print("Prediction shape is: ", pred_traj.shape)

"""
Loading Drifter Location Data
"""
drifter_data_all = read_all_drifter_data()
drifter_id = 11
drifter_ts = drifter_data_all[:48 * 3, drifter_id, :]

compare_trajs(pred_traj, drifter_ts)
