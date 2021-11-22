from Utilities.Plotters import *
from data_processing import *
from sklearn.metrics import mean_squared_error as mse
from torch import Tensor
import torch


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
        traj_label_ls = ["full path", pred_traj1_label, pred_traj2_label, pred_traj3_label]
        if model_path is not None and true_traj is not None:
            node_traj = eval_node_model(model_path, true_traj)
            traj_ls.append(node_traj)
            traj_label_ls.append("KNODE")
        # plotting
        compare_trajs(drifter_ts, traj_ls, traj_label_ls, title="Drifter " + str(drifter_id) + " Day 15-17")

    nowcast_error = mse(drifter_ts[::2], pred_traj1)
    forecast_error = mse(drifter_ts[::2], pred_traj2)
    transformer_error = mse(drifter_ts[::2], pred_traj3)
    return nowcast_error, forecast_error, transformer_error


def error_plot():
    pass


def mse_in_km(true_traj, pred_traj):
    pass


def eval_node_model(model_path, true_traj):
    node_model = torch.load(model_path)['ode_train']
    step_skip = 1
    val_set = Tensor(true_traj[:, :]).detach().unsqueeze(1)
    val_len, _, state_dim = val_set.shape
    with torch.no_grad():
        curr_state = val_set[0]
        curr_latlon = curr_state[:, -2:]
        pred_traj = [curr_latlon]
        for j in range(val_len-1):
            next_state = node_model(curr_state, Tensor([0., 1.]), return_whole_sequence=True).squeeze()
            next_latlon = next_state[-1, -2:].unsqueeze(0)
            pred_traj.append(next_latlon)
            curr_state = torch.cat([val_set[j+1, :, :-2], next_latlon], 1)
        pred_traj = torch.cat(pred_traj)
    return pred_traj.detach().numpy()


if __name__ == '__main__':
    drifter_id = 1
    pred_start_idx = 336  # 336
    with_day = 3  # number of days of flow field in training data
    # model_path = 'NODE/saved_models/drifter_' + str(drifter_id) + '_tanh.pth'
    # true_traj_path = 'Data/training_data/train_data_nowcast_drifter_' + str(drifter_id) + '_knn_10.npy'
    model_path = 'NODE/saved_models/drifter_' + str(drifter_id) + '_tanh_' + str(with_day) +'day.pth'
    true_traj_path = 'Data/training_data/with_' + str(with_day) + 'days_flow/train_data_nowcast_drifter_' + str(drifter_id) + '_knn_10.npy'
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
    model_path = 'NODE/saved_models/drifter_tanh.pth'
    true_traj_path = 'Data/training_data/train_data_nowcast_drifter_59_knn_10.npy'
    true_traj = np.load(true_traj_path)[336:]
    pred_traj = eval_node_model(model_path, true_traj)
    compare_trajs(true_traj[:, 2:], pred_traj1=pred_traj, pred_traj1_label="KNODE")
    """