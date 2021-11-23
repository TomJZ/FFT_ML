from data_processing import *
from Utilities.Plotters import *
from knn_pca import *
from fft_integrator import *

import numpy as np
import torch
import tqdm


def make_prediction_three_fields(field_mode, partial_file_name, drifter_id, start_day, k_neighbor):
    """
    :param field_mode:
    :param partial_file_name:
    :param drifter_id:
    :param start_day:
    :param k_neighbor:
    """
    if field_mode == "transformer":
        field_path = 'Data/flow/from_torch/predict' + partial_file_name
    elif field_mode == "forecast":
        field_path = 'Data/flow/from_torch/forecast' + partial_file_name
    elif field_mode == "nowcast":
        field_path = 'Data/flow/from_torch/eval' + partial_file_name
    else:
        raise NameError("No such field mode!")

    flow_data = torch.load(field_path, map_location='cpu').detach().numpy()
    flow_time = np.arange(len(flow_data))
    print("flow data shape: ", flow_data.shape)

    """
    Loading All Drifter Location Data
    """
    drifter_data_all = read_all_drifter_data()
    print("Drifter data size: ", drifter_data_all.shape)

    """
    Drifter Trajectory Prediction
    """

    initial_condition = drifter_data_all[(start_day - 1) * 48, drifter_id, :]
    save_name = "Data/predictions/drifter_" + str(drifter_id) + "_" + field_mode + "_pred_traj.npy"
    traj_pred = field_integrator(initial_condition, flow_data, flow_time, k_neighbor, save_name=save_name)
    print(traj_pred.shape)


def predict_with_whole_field(drifter_id, k_neighbor, start_day, flow_data_path, flow_time_path):
    """
    :param drifter_id: int, the id of the drifter, starting from 0
    :param k_neighbor: int, k nearest neighbor
    :param start_day: int, the day to be used as the initial condition
    :param flow_data_path: string, the path where the flow is loaded, the flow is [time_len, lat_size, lon_size, 4]
    :param flow_time_path: string, the path where the time is loaded, 1D array
    :return drifter_ts: true trajectory, sampled every half hour
    :return pred_traj: predicted trajectory
    """
    """
    Loading All Drifter Location Data
    """
    drifter_data_all = read_all_drifter_data()
    print("Drifter data size: ", drifter_data_all.shape)
    drifter_ts = drifter_data_all[(start_day - 1) * 48:, drifter_id, :]  # the time series path of a single drifter
    drifter_ic = drifter_ts[0].reshape([1, -1])  # the location of a single drifter at one snapshot
    max_lat = np.nanmax(drifter_ts[:, 0]) + 3
    min_lat = np.nanmin(drifter_ts[:, 0]) - 3
    max_lon = np.nanmax(drifter_ts[:, 1]) + 3
    min_lon = np.nanmin(drifter_ts[:, 1]) - 3
    # min_lat = max_lat - 7
    # min_lon = max_lon - 7

    """
    Loading Ocean Flow Data
    """
    flow_data_all = read_flow_data(flow_data_path)
    flow_time = read_flow_data(flow_time_path)

    flow_data = flow_data_all
    flow_data_all[flow_data_all > 1e20] = 0
    flow_data[flow_data == 0] = None

    max_lat_idx = np.argmin(flow_data_all[0, :, 0, 0] < max_lat)
    min_lat_idx = np.argmax(flow_data_all[0, :, 0, 0] > min_lat)
    max_lon_idx = np.argmin(flow_data_all[0, 0, :, 1] < max_lon)
    min_lon_idx = np.argmax(flow_data_all[0, 0, :, 1] > min_lon)
    if max_lat_idx == 0:
        max_lat_idx = len(flow_data_all[0, :, 0, 0]) - 1  # if max lat is larger than all flow lat
    if max_lon_idx == 0:
        max_lon_idx = len(flow_data_all[0, 0, :, 1]) - 1  # if max lat is larger than all flow lat
    # cropping flow field
    flow_data = flow_data[:, min_lat_idx:max_lat_idx, min_lon_idx:max_lon_idx, :]
    print("Flow data has size: ", flow_data.shape)

    """
    KNN PCA
    """
    ind_flat, ind_2d = knn_on_flow(drifter_ic, flow_data[(start_day - 1) * 24, :, :, 0:2],
                                   k_neighbor)  # only for plotting
    recon_vel = knn_then_pca(drifter_ic, flow_data[(start_day - 1) * 24], k_neighbor)
    print("reconstructed flow is: ", recon_vel)

    # visualize_flow(flow_data, 2)
    # visualize_all_drifter(drifter_data_all, flow_data)
    # visualize_pca(flow_data, ind_flat, (start_day-1)*24, recon_vel, drifter_ts=drifter_ts)
    pred_traj = field_integrator(drifter_ic, flow_data, flow_time, k_neighbor, save_name=None).reshape([-1, 2])
    plot_lim = [min_lon, max_lon, min_lat, max_lat]  # setting plot limit for comparisons
    compare_trajs(drifter_ts, pred_traj1=pred_traj, pred_traj1_label="nowcast", plot_lim=None)

    return drifter_ts, pred_traj


def generate_training_data_1day(drifter_id, k_neighbor, flow_data_path, flow_time_path):
    """
    :param drifter_id: int, the id of the drifter, starting from 0
    :param k_neighbor: int, k nearest neighbor
    :param flow_data_path: string, the path where the flow is loaded, the flow is [time_len, lat_size, lon_size, 4]
    :param flow_time_path: string, the path where the time is loaded, 1D array
    :return: training data
    """
    """
    Loading All Drifter Location Data
    """
    drifter_data_all = read_all_drifter_data()
    drifter_ts = drifter_data_all[::2, drifter_id, :]  # the time series path of a single drifter

    print("Drifter time series size: ", drifter_ts.shape)
    max_lat = np.nanmax(drifter_ts[:, 0]) + 3
    min_lat = np.nanmin(drifter_ts[:, 0]) - 3
    max_lon = np.nanmax(drifter_ts[:, 1]) + 3
    min_lon = np.nanmin(drifter_ts[:, 1]) - 3

    """
    Loading Ocean Flow Data
    """
    flow_data_all = read_flow_data(flow_data_path)
    flow_time = read_flow_data(flow_time_path)

    flow_data = flow_data_all
    flow_data_all[flow_data_all > 1e20] = 1e5
    # flow_data[flow_data == 0] = 1e10

    max_lat_idx = np.argmin(flow_data_all[0, :, 0, 0] < max_lat)
    min_lat_idx = np.argmax(flow_data_all[0, :, 0, 0] > min_lat)
    max_lon_idx = np.argmin(flow_data_all[0, 0, :, 1] < max_lon)
    min_lon_idx = np.argmax(flow_data_all[0, 0, :, 1] > min_lon)
    if max_lat_idx == 0:
        max_lat_idx = len(flow_data_all[0, :, 0, 0]) - 1  # if max lat is larger than all flow lat
    if max_lon_idx == 0:
        max_lon_idx = len(flow_data_all[0, 0, :, 1]) - 1  # if max lat is larger than all flow lat
    # cropping flow field
    flow_data = flow_data[:, min_lat_idx:max_lat_idx, min_lon_idx:max_lon_idx, :]
    print("Flow data has size: ", flow_data.shape)

    """
    KNN PCA
    """
    training_data = []
    last_good_loc = np.array([[0, 0]]).T
    print("Running KNN-PCA and Generating Training Data: ")
    for i in tqdm.tqdm(range(len(flow_data))):
        try:
            drifter_location = drifter_ts[i].reshape([1, -1])  # the drifter location is sampled every half hour
        except IndexError:
            print("max index in drifter ts is: ", i)
            break
        # if not np.any(np.isnan(drifter_location)):
        #    last_good_loc = drifter_location
        # if np.any(np.isnan(drifter_location)):
        #    print("nan in drifter location detected at index: ", i)
        #    drifter_location = last_good_loc

        # print(drifter_location)
        # np.any(np.isnan(flow_data[i]))
        # np.all(np.isfinite(flow_data[i]))
        recon_vel = knn_then_pca(drifter_location, flow_data[i+24], k_neighbor)
        training_point = np.concatenate([recon_vel, drifter_location.reshape(-1)])
        training_data.append(training_point)

    training_data = np.array(training_data)
    save_name = "Data/training_data/submission_1day/train_data_nowcast_drifter_" + str(drifter_id) + "_knn_" + str(
        k_neighbor) + ".npy"
    with open(save_name, 'wb') as f:
        np.save(f, training_data)

    return training_data


def generate_training_data_3day(drifter_id, k_neighbor, flow_data_path, flow_time_path):
    """
    :param drifter_id: int, the id of the drifter, starting from 0
    :param k_neighbor: int, k nearest neighbor
    :param flow_data_path: string, the path where the flow is loaded, the flow is [time_len, lat_size, lon_size, 4]
    :param flow_time_path: string, the path where the time is loaded, 1D array
    :return: training data
    """
    """
    Loading All Drifter Location Data
    """
    drifter_data_all = read_all_drifter_data()
    drifter_ts = drifter_data_all[::2, drifter_id, :]  # the time series path of a single drifter

    print("Drifter time series size: ", drifter_ts.shape)
    max_lat = np.nanmax(drifter_ts[:, 0]) + 3
    min_lat = np.nanmin(drifter_ts[:, 0]) - 3
    max_lon = np.nanmax(drifter_ts[:, 1]) + 3
    min_lon = np.nanmin(drifter_ts[:, 1]) - 3

    """
    Loading Ocean Flow Data
    """
    flow_data_all = read_flow_data(flow_data_path)

    flow_data = flow_data_all
    flow_data_all[flow_data_all > 1e20] = 1e5
    # flow_data[flow_data == 0] = 1e10

    max_lat_idx = np.argmin(flow_data_all[0, :, 0, 0] < max_lat)
    min_lat_idx = np.argmax(flow_data_all[0, :, 0, 0] > min_lat)
    max_lon_idx = np.argmin(flow_data_all[0, 0, :, 1] < max_lon)
    min_lon_idx = np.argmax(flow_data_all[0, 0, :, 1] > min_lon)
    if max_lat_idx == 0:
        max_lat_idx = len(flow_data_all[0, :, 0, 0]) - 1  # if max lat is larger than all flow lat
    if max_lon_idx == 0:
        max_lon_idx = len(flow_data_all[0, 0, :, 1]) - 1  # if max lat is larger than all flow lat
    # cropping flow field
    flow_data = flow_data[:, min_lat_idx:max_lat_idx, min_lon_idx:max_lon_idx, :]
    print("\nFlow data has size: ", flow_data.shape)

    """
    KNN PCA
    """
    training_data = []
    last_good_loc = np.array([[0, 0]]).T
    print("Running KNN-PCA and Generating Training Data: ")
    for i in tqdm.tqdm(range(len(flow_data))):
        try:
            drifter_location = drifter_ts[i].reshape([1, -1])
        except IndexError:
            print("max index in drifter ts is: ", i)
            break

        recon_vel0 = knn_then_pca(drifter_location, flow_data[i], k_neighbor)
        recon_vel1 = knn_then_pca(drifter_location, flow_data[i], k_neighbor) if i == 0 else knn_then_pca(
            drifter_location, flow_data[i - 1], k_neighbor)
        recon_vel2 = knn_then_pca(drifter_location, flow_data[i], k_neighbor) if i == len(
            flow_data) - 1 else knn_then_pca(drifter_location, flow_data[i + 1], k_neighbor)
        training_point = np.concatenate([recon_vel0, recon_vel1, recon_vel2, drifter_location.reshape(-1)])
        training_data.append(training_point)

    training_data = np.array(training_data)

    save_name = "Data/training_data/submission_1day/train_data_nowcast_drifter_" + str(drifter_id) + "_knn_" + str(
        k_neighbor) + ".npy"
    with open(save_name, 'wb') as f:
        np.save(f, training_data)

    return training_data


if __name__ == "__main__":
    '''
    # choose from "transformer", "forecast", "nowcast"
    field_modes = ["transformer", "forecast", "nowcast"]
    drifter_id = 41
    start_day = 15
    k_neighbor = 10
    partial_file_names = ['_lat_200_250_lon_0_80.pth',  # 41, 52, 59
                          '_lat_200_250_lon_160_240.pth',  # 1, 70
                          '_lat_250_300_lon_240_320.pth',  # 42, 87, 72
                          '_lat_300_350_lon_160_240.pth',  # 55
                          '_lat_300_350_lon_240_320.pth']  # 7, 67

    for _, field_mode in enumerate(field_modes):
        for _, drifter_id in enumerate([7, 67]):
            make_prediction_three_field(field_mode, partial_file_names[4], drifter_id, start_day, k_neighbor)
    '''
    drifter_id = 0
    k_neighbor = 10
    start_day = 1
    flow_data_path = "Data/flow/noaa_nowcast_data_nov_2_to_nov_22.npy"
    flow_time_path = "Data/flow/noaa_nowcast_times_nov_2_to_nov_22.npy"
    drifter_ids = np.arange(93)  # all drifter ids
    for _, drifter_id in enumerate(drifter_ids):
        print("processing drifter:", drifter_id)
        try:
            training_data = generate_training_data_1day(drifter_id, k_neighbor, flow_data_path, flow_time_path)
        except:
            continue
    # true_traj, pred_traj = predict_with_whole_field(drifter_id, k_neighbor, start_day, flow_data_path, flow_time_path)
