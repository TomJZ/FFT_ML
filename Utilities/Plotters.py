import matplotlib.pyplot as plt
import numpy as np


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
        ax2.plot(site_lon, site_lat, label='true traj', marker='x', c='green')  # true time series path of a drifter

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


def compare_trajs(pred_traj, true_traj):
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(pred_traj[:, 0], pred_traj[:, 1], label='pred traj',
            c='green')  # pred time series path of a drifter
    ax.plot(true_traj[:, 0], true_traj[:, 1], label='true traj',
            c='red')  # true time series path of a drifter
    plt.legend()
    plt.show()