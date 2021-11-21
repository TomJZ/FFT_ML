import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA

def knn_on_flow(drifter_latlon, flow_latlon, k):
    """
    gives the k nearest neighboring grids' latitude and longitude for a drifter in a flow
    :param drifter_latlon: the latitude and longitude of the drifter, [1, 2]
    :param flow_latlon: the latitudes and longitudes of the flow field, [H, W, 2]
    :param k: k nearest neighbors
    :return indices_2d: the list of indices of the k nearest neighbors, [k, 2]
    """
    h, w, _ = flow_latlon.shape
    latlon = flow_latlon.reshape(-1, 2)  # flattening the array of flow latlons to be [H*W,2]
    latlon_append = np.concatenate([drifter_latlon, latlon], 0)

    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(latlon_append)
    distances, indices = nbrs.kneighbors(latlon_append)
    ind_flat = indices[0, 1:]-1
    ind_2d = np.array(np.unravel_index(ind_flat, (h, w))).T

    return ind_flat, ind_2d


def pca_on_flow(flow_uv):
    """
    :param flow_field: the u_vel and v_vel of the flow field, [len, 2]
    :return [u_pca, v_pca]: the reconstructed u_vel and v_vel
    """
    pca = PCA(n_components=1, svd_solver='full')
    pca.fit(flow_uv)
    flow_pca = pca.transform(flow_uv)
    reconstructed_flow = pca.inverse_transform(flow_pca)

    return reconstructed_flow[0, :]

def knn_then_pca(drifter_latlon, flow_field_snapshot, k):
    """
    :param drifter_latlon: the latitude and longitude of the drifter, [1, 2]
    :param flow_field_snapshot: a snapshot of the flow field including latlon and u v vels, [H, W, 4]
    :param k: k nearest neighbors

    :return: reconstructed_flow: the reconstructed u_vel and v_vel, [2,]
    """
    drifter_latlon = drifter_latlon.reshape([1,2])
    flow_field_latlon = flow_field_snapshot[:, :, 0:2]
    flow_field_uv = flow_field_snapshot[:, :, 2:4]
    ind_flat, ind_2d = knn_on_flow(drifter_latlon, flow_field_latlon, k)
    flow_field_uv_flat = flow_field_uv.reshape([-1, 2])
    neighbor_flow = flow_field_uv_flat[ind_flat]
    reconstructed_flow = pca_on_flow(neighbor_flow)
    return reconstructed_flow
