import numpy as np
import torch
import os
import numpy as np
class Dataset_builder(torch.utils.data.Dataset):
    def __init__(self, data_folder, data_type, lat_min, lat_max, lon_min, lon_max, transform=None):

        self.s_data = []
        self.t_data = []
        if data_type == "train":
            self.s_data.append(torch.tensor(np.load(os.path.join(data_folder, "noaa_nowcast_data_nov_2_to_nov_19.npy")),
                                       dtype=torch.float)[:360, lat_min:lat_max, lon_min:lon_max,])  # get the first 4 day's states as training set (8,9,10,11|)
            self.t_data.append(torch.tensor(np.load(os.path.join(data_folder, "noaa_nowcast_times_nov_2_to_nov_19.npy")),
                                       dtype=torch.float)[:360])  # get the first 4 day's times as training set (1-14|)

        elif data_type == "eval":
            self.s_data.append(torch.tensor(np.load(os.path.join(data_folder, "noaa_nowcast_data_nov_2_to_nov_19.npy")),
                                            dtype=torch.float)[360:360+72, lat_min:lat_max, lon_min:lon_max,])  # get the last 4 day's states as training set (|15-17)
            self.t_data.append(torch.tensor(np.load(os.path.join(data_folder, "noaa_nowcast_times_nov_2_to_nov_19.npy")),
                                            dtype=torch.float)[360:360+72])  # get the last 4 day's times as training set (|15-17)

        elif data_type == "forecast":
            self.s_data.append(torch.tensor(np.load(os.path.join(data_folder, "forecast_data_from_nov_16.npy")),
                                            dtype=torch.float)[:72, lat_min:lat_max, lon_min:lon_max,])  # get the forecast for (|15-17)
            self.t_data.append(torch.tensor(np.load(os.path.join(data_folder, "forecast_times_from_nov_16.npy")),
                                            dtype=torch.float)[:72])  # get the forecast for (|15-17)

        elif data_type == "inference":
            self.s_data.append(torch.tensor(np.load(os.path.join(data_folder, "forecast_data_from_nov_22_interp.npy")),
                                            dtype=torch.float)[:, lat_min:lat_max, lon_min:lon_max,])  # get the forecast for (|15-17)
            self.t_data.append(torch.tensor(np.load(os.path.join(data_folder, "forecast_times_from_nov_22_interp.npy")),
                                            dtype=torch.float)[:])  # get the forecast for (|22-02)


        # # parse all dataset in the given folder
        # # .npz file list
        # npz_files = [os.path.join(data_folder, npz_file) for npz_file in os.listdir(data_folder) if npz_file.endswith(".npz")]
        # # init data list
        # self.s_data = []
        # self.t_data = []
        # # loop the file list and parse the dataset
        # for data_name in npz_files:
        #     data = np.load(data_name)
        #     self.s_data.append(torch.tensor(data['s'], dtype=torch.float))
        #     self.t_data.append(torch.tensor(data['t'].astype(np.int), dtype=torch.float))

        # load transform if existed.
        self.transform = transform

    def __getitem__(self, index):
        s_data = self.s_data[index]
        t_data = self.t_data[index]
        # if transformation needed.
        if self.transform:
            pass

        return s_data, t_data

    def __len__(self):
        return len(self.s_data)