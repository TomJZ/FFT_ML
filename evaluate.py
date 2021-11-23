from Utilities.Plotters import *
from data_processing import *
from sklearn.metrics import mean_squared_error as mse
from torch import Tensor
import torch
import tqdm
from knn_pca import *
import geopy.distance  # for converting lon lat to distances


def latlon_to_km(latlon1, latlon2):
    """
    :param latlon1: [2,] latitude and longitude of the first location
    :param latlon2: [2,] latitude and longitude of the second location
    """
    dist_in_km = geopy.distance.geodesic((latlon1[0], latlon1[1]), (latlon2[0], latlon2[1])).km
    return dist_in_km


def compute_traj_loss(drifter_id, plot_traj=False, model_path=None, true_traj=None):
    """
    high level evaluation function, not general
    """
    """
    Loading Drifter Prediction Data
    """
    pred_traj1_label = "nowcast_integrator"
    pred_traj2_label = "forecast_integrator"
    pred_traj3_label = "transformer_integrator"
    with open('Data/predictions/drifter_' + str(drifter_id) + '_nowcast_pred_traj.npy', 'rb') as f:
        pred_traj1 = np.load(f)

    with open('Data/predictions/drifter_' + str(drifter_id) + '_forecast_pred_traj.npy', 'rb') as f:
        pred_traj2 = np.load(f)

    with open('Data/predictions/drifter_' + str(drifter_id) + '_transformer_pred_traj.npy', 'rb') as f:
        pred_traj3 = np.load(f)

    print("Prediction shape is: ", pred_traj1.shape)

    """
    Loading Drifter Location Data
    """
    drifter_data_all = read_all_drifter_data()

    start_day = 15  # the day start from which prediction is to be plotted
    duration = 3  # how many days ahead to plot
    drifter_ts = drifter_data_all[(start_day - 1) * 48:(start_day - 1) * 48 + duration * 48, drifter_id, :]
    if plot_traj:  # plotting all trajs
        traj_ls = [drifter_data_all[:, drifter_id, :], pred_traj1, pred_traj2, pred_traj3]
        traj_label_ls = ["full true path", pred_traj1_label, pred_traj2_label, pred_traj3_label]
        if model_path is not None and true_traj is not None:
            node_traj, _, _ = eval_node_model(model_path, true_traj)
            traj_ls.append(node_traj)
            traj_label_ls.append("KNODE")
        # plotting
        compare_trajs(drifter_ts, traj_ls, traj_label_ls, title="Drifter " + str(drifter_id) + " Day 15-17")

    nowcast_error = mse(drifter_ts[::2], pred_traj1)
    forecast_error = mse(drifter_ts[::2], pred_traj2)
    transformer_error = mse(drifter_ts[::2], pred_traj3)

    nowcast_dist = traj_deviation_in_km(drifter_ts[::2], pred_traj1)
    forecast_dist = traj_deviation_in_km(drifter_ts[::2], pred_traj2)
    transformer_dist = traj_deviation_in_km(drifter_ts[::2], pred_traj3)
    node_dist = traj_deviation_in_km(drifter_ts[::2], node_traj)
    print(nowcast_dist.shape)

    fig2 = plt.figure(figsize=(10, 5))
    ax2 = fig2.add_subplot(1, 1, 1)
    ax2.plot(nowcast_dist, label=pred_traj1_label)
    ax2.plot(forecast_dist, label=pred_traj2_label)
    ax2.plot(transformer_dist, label=pred_traj3_label)
    ax2.plot(node_dist, label="KNODE")
    ax2.set_title("Distance error plot")
    plt.legend()
    plt.show()

    return nowcast_error, forecast_error, transformer_error


def error_plot():
    pass


def traj_deviation_in_km(true_traj, pred_traj):
    """
    :param true_traj: [time_len, 2], the true trajectory of
    :param pred_traj: [time_len, 2], the predicted trajectory
    :return: [time_len,], an array of distances between the two trajectories
    """
    dist_arr = []
    for i in range(len(true_traj)):
        distance_in_km = latlon_to_km(true_traj[i], pred_traj[i])
        dist_arr.append(distance_in_km)
    dist_arr = np.array(dist_arr)
    return dist_arr


def eval_node_model_using_training_data(model_path, true_traj):
    """
    :param model_path:
    :param true_traj:
    :return:
    """
    node_model = torch.load(model_path)['ode_train']
    step_skip = 1
    val_set = Tensor(true_traj[:, :]).detach().unsqueeze(1)
    val_len, _, state_dim = val_set.shape
    with torch.no_grad():
        curr_state = val_set[0]
        curr_latlon = curr_state[:, -2:]
        pred_traj = [curr_latlon]
        for j in range(val_len - 1):
            next_state = node_model(curr_state, Tensor([0., 1.]), return_whole_sequence=True).squeeze()
            next_latlon = next_state[-1, -2:].unsqueeze(0)
            pred_traj.append(next_latlon)
            curr_state = torch.cat([val_set[j + 1, :, :-2], next_latlon], 1)
        pred_traj = torch.cat(pred_traj)
        pred_traj = pred_traj.detach().numpy()
        dist_err_arr = traj_deviation_in_km(pred_traj, true_traj[:, -2:])
        first_failure_idx = np.argmax(dist_err_arr > 32.)
    return pred_traj, dist_err_arr, first_failure_idx


def eval_node_model_using_flow_field(model_path, initial_condition, flow_data_path, true_traj=None):
    """
    :param model_path:
    :param true_traj:
    :return:
    """
    flow_data = read_flow_data(flow_data_path)
    print("flow shape: ", flow_data.shape, "ic shape: ", initial_condition.shape)
    flow_data[flow_data > 1e20] = 0

    max_lat = initial_condition[2] + 5
    min_lat = initial_condition[2] - 5
    max_lon = initial_condition[3] + 5
    min_lon = initial_condition[3] - 5

    max_lat_idx = np.argmin(flow_data[0, :, 0, 0] < max_lat)
    min_lat_idx = np.argmax(flow_data[0, :, 0, 0] > min_lat)
    max_lon_idx = np.argmin(flow_data[0, 0, :, 1] < max_lon)
    min_lon_idx = np.argmax(flow_data[0, 0, :, 1] > min_lon)
    if max_lat_idx == 0:
        max_lat_idx = len(flow_data[0, :, 0, 0]) - 1  # if max lat is larger than all flow lat
    if max_lon_idx == 0:
        max_lon_idx = len(flow_data[0, 0, :, 1]) - 1  # if max lat is larger than all flow lat
    # cropping flow field
    flow_data = flow_data[:, min_lat_idx:max_lat_idx, min_lon_idx:max_lon_idx, :]

    node_model = torch.load(model_path)['ode_train']

    with torch.no_grad():
        curr_state = Tensor(initial_condition).view(1, -1)
        curr_latlon = curr_state[:, -2:]
        pred_traj = [curr_latlon]
        for j in tqdm.tqdm(range(112)):
            next_state = node_model(curr_state, Tensor([0., 1.]), return_whole_sequence=True).squeeze()
            next_latlon = next_state[-1, -2:].unsqueeze(0)
            pred_traj.append(next_latlon)
            flow_knn_pca = knn_then_pca(next_latlon.detach().numpy(), flow_data[j], 10)
            curr_state = torch.cat([Tensor(flow_knn_pca).view(1, -1), next_latlon], 1)
        pred_traj = torch.cat(pred_traj)
        pred_traj = pred_traj.detach().numpy()
        if true_traj is not None:
            dist_err_arr = traj_deviation_in_km(pred_traj, true_traj[:len(pred_traj), -2:])
            if any(dist_err_arr > 32.):
                first_failure_idx = np.argmax(dist_err_arr > 32.)
            else:
                first_failure_idx = np.nan
        else:
            dist_err_arr = None
            first_failure_idx = None
    return pred_traj, dist_err_arr, first_failure_idx


if __name__ == '__main__':
    """
    drifter_id = 7
    pred_start_idx = 336  # 336
    with_day = 3  # number of days of flow field in training data
    # model_path = 'NODE/saved_models/drifter_' + str(drifter_id) + '_tanh.pth'
    # true_traj_path = 'Data/training_data/train_data_nowcast_drifter_' + str(drifter_id) + '_knn_10.npy'
    model_path = 'NODE/saved_models/drifter_' + str(drifter_id) + '_tanh_' + str(with_day) + 'day.pth'
    true_traj_path = 'Data/training_data/with_' + str(with_day) + 'days_flow/train_data_nowcast_drifter_' + str(
        drifter_id) + '_knn_10.npy'
    true_traj = np.load(true_traj_path)[pred_start_idx:]

    # Computing traj loss from three different field approaches
    n_err_total = 0
    f_err_total = 0
    t_err_total = 0
    # drifter_ids = [41, 52, 59, 1, 70, 42, 87, 72, 7, 67, 55]
    drifter_ids = [drifter_id]
    plot_traj = True
    for _, drifter_id in enumerate(drifter_ids):
        n_err, f_err, t_err = compute_traj_loss(drifter_id,
                                                plot_traj=plot_traj,
                                                model_path=model_path,
                                                true_traj=true_traj)
        n_err_total += n_err
        f_err_total += f_err
        t_err_total += t_err
    print(n_err_total, f_err_total, t_err_total)
    """
    """
    drifter_id = 59
    model_path = 'NODE/saved_models/drifter_' + str(drifter_id) + '_tanh_3day.pth'
    true_traj_path = 'Data/training_data/from_nov2_to_nov19_with_3days_flow/train_data_nowcast_drifter_' + str(
        drifter_id) + '_knn_10.npy'
    true_traj = np.load(true_traj_path)[336:]
    pred_traj, _, first_hour = eval_node_model_using_training_data(model_path, true_traj)
    print("first hour to diverge: ", first_hour)
    compare_trajs(true_traj[:, -2:], [pred_traj], ["KNODE"])

    """
    drifter_id = 3
    model_path = 'NODE/submission_models/drifter_' + str(drifter_id) + '_tanh_1day.pth'
    true_traj_path = 'Data/training_data/submission_1day/train_data_nowcast_drifter_' + \
                     str(drifter_id) + '_knn_10.npy'
    true_traj = np.load(true_traj_path)
    initial_condition = true_traj[-1]
    flow_path = 'Data/flow/noaa_forecast_data_from_nov_22.npy'
    pred_traj, _, first_hour = eval_node_model_using_flow_field(model_path, initial_condition,
                                                                flow_path)
    print("first hour to diverge from 32km: ", first_hour)
    compare_trajs(true_traj[:, -2:], [pred_traj], ["KNODE"])
