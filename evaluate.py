from Utilities.Plotters import *
from data_processing import *
from sklearn.metrics import mean_squared_error as mse


def compute_traj_loss(drifter_id, plot_traj=False):
    """
    Loading Drifter Prediction Data
    """
    pred_traj1_label = "nowcast"
    pred_traj2_label = "forecast"
    pred_traj3_label = "transformer"
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
    if plot_traj:
        compare_trajs(drifter_ts,
                      pred_traj1=pred_traj1, pred_traj1_label=pred_traj1_label,
                      pred_traj2=pred_traj2, pred_traj2_label=pred_traj2_label,
                      pred_traj3=pred_traj3, pred_traj3_label=pred_traj3_label)

    nowcast_error = mse(drifter_ts[::2], pred_traj1)
    forecast_error = mse(drifter_ts[::2], pred_traj2)
    transformer_error = mse(drifter_ts[::2], pred_traj3)
    return nowcast_error, forecast_error, transformer_error

def error_plot():
    pass

def mse_in_km(true_traj, pred_traj):




if __name__ == '__main__':
    n_err_total = 0
    f_err_total = 0
    t_err_total = 0
    # drifter_ids = [41, 52, 59, 1, 70, 42, 87, 72, 7, 67, 55]
    drifter_ids = [1]
    plot_traj = True
    for _, drifter_id in enumerate(drifter_ids):
        n_err, f_err, t_err = compute_traj_loss(drifter_id, plot_traj=plot_traj)
        n_err_total += n_err
        f_err_total += f_err
        t_err_total += t_err
    print(n_err_total, f_err_total, t_err_total)
