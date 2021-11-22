import matplotlib.pyplot as plt
from data_processing import *
import os


def visualize_flow(flow_data, snapshot):
    """
    :param flow_data: must have size [time_len, lat_size, lon_size, 4], the first two dims are lat and lon
    :param snapshot: integer-valued time index to be plotted
    :return:
    """
    fig = plt.figure(figsize=(10, 20))
    ax1 = fig.add_subplot(2, 1, 1)
    # note that for plotting, latitude and longitude are reversed
    ax1.quiver(flow_data[snapshot, :, :, 1], flow_data[snapshot, :, :, 0],
               flow_data[snapshot, :, :, 2], flow_data[snapshot, :, :, 3])  # plot the flow as a vector field

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.imshow(flow_data[0, :, :, 2], cmap='ocean', interpolation='nearest')  # plot u_vel as a color map
    plt.gca().invert_yaxis()
    plt.show()


def visualize_pca(flow_data, ind_flat, snapshot, recon_vel, drifter_ts=None):
    """
    :param ind_flat: [k,], where k is the number of nearest neighbors
    :param flow_data: [time_len, lat_size, lon_size, 4], the first two dims are lat and lon
    :param snapshot: integer-valued time index to be plotted
    :param recon_vel: [2,], u_vel and v_vel respectively
    :param drifter_ts: [time_len, 2], the time series path of a drifter
    :return:
    """
    all_coors = flow_data[snapshot, :, :, 0:2].reshape([-1, 2])
    neighbor_coors = all_coors[ind_flat]

    fig2 = plt.figure(figsize=(10, 5))
    ax2 = fig2.add_subplot(1, 1, 1)
    ax2.quiver(flow_data[snapshot, :, :, 1], flow_data[snapshot, :, :, 0],
               flow_data[snapshot, :, :, 2], flow_data[snapshot, :, :, 3])  # entire flow field
    ax2.scatter(neighbor_coors[:, 1], neighbor_coors[:, 0], s=50, label='neighbors', marker='o')  # neighbors

    if drifter_ts is not None:
        site_lat = drifter_ts[:, 0]
        site_lon = drifter_ts[:, 1]
        plt.xlim([np.nanmin(site_lon) - 1, np.nanmax(site_lon) + 1])
        plt.ylim([np.nanmin(site_lat) - 1, np.nanmax(site_lat) + 1])
        ax2.quiver(drifter_ts[0, 1], drifter_ts[0, 0],
                   recon_vel[0], recon_vel[1], color='red',
                   label='reconstructed vel')  # recon vel plotted for a drifter
        ax2.plot(site_lon, site_lat, label='true traj', marker='.', c='green')  # true time series path of a drifter


    plt.legend()
    plt.show()


def visualize_all_drifter(drifter_data_all, flow_data):
    """
    :param drifter_data_all: [time_len, drifter_count, 2], time series path of all drifters
    :param flow_data: [time_len, lat_size, lon_size, 4], the first two dims are lat and lon
    :return:
    """
    snapshot = 0
    drifter_id = 50
    use_ax_lim = False
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(1, 1, 1)
    # note that for plotting, latitude and longitude are reversed
    ax.pcolormesh(flow_data[snapshot, :, :, 1], flow_data[snapshot, :, :, 0], flow_data[snapshot, :, :, 2])

    for drifter_i in range(90):
        site_lat = drifter_data_all[:, drifter_i, 0]
        site_lon = drifter_data_all[:, drifter_i, 1]
        ax.plot(site_lon, site_lat)
        plt.annotate("d_" + str(drifter_i), (site_lon[0], site_lat[1]), c='yellow')

    plt.show()


def compare_trajs(true_traj, pred_traj_ls=None, pred_label_ls=None, plot_lim=None, title=None):
    """
    Compares multiple trajectories if provided
    :param pred_traj: [time_len, 2], the predicted trajectory
    :param true_traj: [time_len, 2], the true trajectory
    :return:
    """
    assert(len(pred_traj_ls) == len(pred_label_ls))
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(true_traj[:, 1], true_traj[:, 0], label='true traj',
            c='black', marker='x')  # true time series path of a drifter
    if pred_traj_ls is not None:
        for i, pred_traj in enumerate(pred_traj_ls):
            ax.plot(pred_traj[:, 1], pred_traj[:, 0],
                    label=pred_label_ls[i], marker='.')  # pred time series path of a drifter

    if title is not None:
        ax.set_title(title)

    if plot_lim is not None:
        min_lon = plot_lim[0]
        max_lon = plot_lim[1]
        min_lat = plot_lim[2]
        max_lat = plot_lim[3]
        plt.xlim([min_lon - 1, max_lon + 1])
        plt.ylim([min_lat - 1, max_lat + 1])
    plt.legend()
    plt.show()


def visualize_training_data(training_traj, flow_data_path):
    """
    :param training_traj: [time_len, 4]
    """
    fig2 = plt.figure(figsize=(10, 5))
    ax2 = fig2.add_subplot(1, 1, 1)
    ax2.plot(training_traj[:, 3], training_traj[:, 2], label='true traj', linewidth=3)
    ax2.quiver(training_traj[:, 3], training_traj[:, 2],
               training_traj[:, 1], training_traj[:, 0], label='pca flow')  # entire flow field

    """
    Loading Ocean Flow Data
    """
    flow_data_all = read_flow_data(flow_data_path)
    flow_data_all[flow_data_all > 1e20] = 0

    min_lat = np.min(training_traj[:, 2])
    max_lat = np.max(training_traj[:, 2])
    min_lon = np.min(training_traj[:, 3])
    max_lon = np.max(training_traj[:, 3])
    max_lat_idx = np.argmin(flow_data_all[0, :, 0, 0] < max_lat)
    min_lat_idx = np.argmax(flow_data_all[0, :, 0, 0] > min_lat)
    max_lon_idx = np.argmin(flow_data_all[0, 0, :, 1] < max_lon)
    min_lon_idx = np.argmax(flow_data_all[0, 0, :, 1] > min_lon)
    if max_lat_idx == 0:
        max_lat_idx = len(flow_data_all[0, :, 0, 0]) - 1  # if max lat is larger than all flow lat
    if max_lon_idx == 0:
        max_lon_idx = len(flow_data_all[0, 0, :, 1]) - 1  # if max lat is larger than all flow lat

    snapshot = 0
    flow_snapshot = flow_data_all[snapshot, min_lat_idx:max_lat_idx, min_lon_idx:max_lon_idx, :]
    ax2.quiver(flow_snapshot[:, :, 1], flow_snapshot[:, :, 0],
               flow_snapshot[:, :, 3], flow_snapshot[:, :, 2], label='flow field', color='grey')  # entire flow field

    plt.legend()
    plt.show()


if __name__ == '__main__':
    training_data = np.load('../Data/training_data/train_data_nowcast_drifter_1_knn_10.npy')
    flow_data_path = "../Data/flow/noaa_nowcast_data_nov_2_to_nov_19.npy"
    visualize_training_data(training_data, flow_data_path)
    print(training_data.shape)

    """
    cwd = os.getcwd()
    os.chdir(os.path.abspath(os.path.join(cwd, os.pardir)))

    drifter_data_all = read_all_drifter_data()

    drifter_id = 67
    path1 = 'Data/predictions/drifter_' + str(drifter_id) + '_forecast_pred_traj.npy'
    path2 = 'Data/predictions/drifter_' + str(drifter_id) + '_nowcast_pred_traj.npy'
    path3 = 'Data/predictions/drifter_' + str(drifter_id) + '_transformer_pred_traj.npy'
    label1 = "forecast"
    label2 = "nowcast"
    label3 = "transformer"
    traj1 = np.load(path1)
    traj2 = np.load(path2)
    traj3 = np.load(path3)

    compare_trajs(drifter_data_all[14*48:48*17, drifter_id],
                  [traj1, traj2, traj3],
                  [label1, label2, label3])
    """


    '''
    """
    Loading Ocean Flow Data
    """
    flow_data_path = 'Data/flow/noaa_nowcast_data_nov_2_to_nov_19.npy'
    flow_data_all = read_flow_data(flow_data_path)

    flow_data = flow_data_all
    flow_data_all[flow_data_all > 1e20] = 0
    flow_data[flow_data == 0] = None
    visualize_all_drifter(drifter_data_all, flow_data)
    '''