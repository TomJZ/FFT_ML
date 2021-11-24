import numpy as np
import torch
import os
import numpy as np
import re
import netCDF4

class Dataset_builder(torch.utils.data.Dataset):
    def __init__(self, data_folder, data_type="n",transform=None):
        self.data_folder = data_folder
        self.data_type = data_type
        # parse all dataset in the given folder
        # .npz file list
        nc_files = [nc_file for nc_file in os.listdir(data_folder) if nc_file.endswith(".nc")]

        regex_n = re.compile(r'n\d+')
        regex_f = re.compile(r'f\d+')


        # create dataset list
        nowcast_list = []
        forecast_list = []
        # readin the "nXXX" and "fXXX"
        for f in nc_files:
            n_index = regex_n.findall(f)
            f_index = regex_f.findall(f)
            if len(n_index) > 0:
                nowcast_list.append(int(n_index[0][1:]))
            if len(f_index) > 0:
                forecast_list.append(int(f_index[0][1:]))

        # sort the index
        nowcast_list = sorted(nowcast_list)
        forecast_list = sorted(forecast_list)

        self.data_index = nowcast_list if self.data_type == 'n' else forecast_list

        # load transform if existed.
        self.transform = transform

        # crop region
        self.x_min = 1700
        self.x_max = self.x_min + 500
        self.y_min = 2600
        self.y_max = self.y_min + 800

    def __getitem__(self, index):
        # compose the file name
        f_name = "rtofs_glo_2ds_"+self.data_type+"%03d" % self.data_index[index] +"_prog.nc"
        f_name = os.path.join(self.data_folder, f_name)
        # read data
        data = netCDF4.Dataset(f_name)
        lat = torch.tensor(data.variables['Latitude'][self.x_min:self.x_max, self.y_min:self.y_max], dtype=torch.float)
        lon = torch.tensor(data.variables['Longitude'][self.x_min:self.x_max, self.y_min:self.y_max], dtype=torch.float)
        u_vel = torch.tensor(data.variables['u_velocity'][0, 0][self.x_min:self.x_max, self.y_min:self.y_max].filled(0), dtype=torch.float)
        v_vel = torch.tensor(data.variables['v_velocity'][0, 0][self.x_min:self.x_max, self.y_min:self.y_max].filled(0), dtype=torch.float)

        return torch.cat([u_vel.unsqueeze(-1), v_vel.unsqueeze(-1),
                          lat.unsqueeze(-1), lon.unsqueeze(-1)], dim=-1)

    def __len__(self):
        return len(self.data_index)