from knn_pca import *
import geopy
import tqdm


def field_integrator(initial_condition, flow_data, flow_time, k_neighbor, save_name=None):
    """
    :param initial_condition: [2,] the initial position the drifter starts from
    :param flow_data: [time_len, lat_size, lon_size, 4], first two dimensions are lat and lon
    :param flow_time: 1D array of time indices
    :param k_neighbor: int, k nearest neighbor
    :param save_name: path to save the resulting trajectory to
    :return:
    """
    drifter_traj = [initial_condition]
    step_size = 3600  # every hour is 3600 seconds, corresponding to the flow field sampling freq
    total_pred_step = len(flow_time)
    drifter_curr_pos = initial_condition

    for ss in tqdm.tqdm(range(total_pred_step-1)):
        dt = flow_time[ss+1] - flow_time[ss]
        recon_vel = knn_then_pca(drifter_curr_pos, flow_data[ss], k_neighbor)
        u_vel = recon_vel[0]
        v_vel = recon_vel[1]
        u_vel_in_deg = geopy.units.degrees(arcminutes=geopy.units.nautical(meters=u_vel))
        v_vel_in_deg = geopy.units.degrees(arcminutes=geopy.units.nautical(meters=v_vel))
        drifter_curr_pos = drifter_curr_pos + np.array([u_vel_in_deg, v_vel_in_deg]) * step_size * dt
        drifter_traj.append(drifter_curr_pos)

    if save_name is not None:
        with open(save_name, 'wb') as f:
            np.save(f, np.array(drifter_traj))

    return np.array(drifter_traj)