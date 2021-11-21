import numpy as np


def read_data(file_path):
    flow_data = np.load(file_path)
    return flow_data


def read_all_drifter_data():
    """
    Loading Drifter Location Data
    """
    drifter_data_all = []
    drifter_time_all = []
    root_path = 'Data/drifter/day'
    for i in range(17):
        path = root_path + str(i + 1) + '_pos.npz'
        with np.load(path, allow_pickle=True) as data:
            a = data['pos']
            b = data['timestamp']
        drifter_data_all.append(a)
        drifter_time_all.append(b)

    drifter_data_all = np.array(drifter_data_all).astype(np.float64)
    drifter_time_all = np.array(drifter_time_all)
    drifter_data_all[:, :, :, 1][drifter_data_all[:, :, :, 1] is not None] = drifter_data_all[:, :, :, 1][
                                                                                 drifter_data_all[:, :, :,
                                                                                 1] is not None] + 360  # need to offset by 360 to match the coordinates of the flow
    drifter_data_all = np.concatenate(drifter_data_all, 0)
    print("drifter data size: ", drifter_data_all.shape)
    return drifter_data_all
