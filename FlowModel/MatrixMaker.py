import numpy as np
import sys
import torch
from tqdm import tqdm
from Utilities.Animation import *

# file_name = './PredictionTensor/WholeFlowInfer/forecast_8days_transformer_60hrs_predict.pth'
#
# # load partial
# sub_matrix = torch.load(file_name)
# vis_vf_large(1, sub_matrix[:,:,:,2:], sub_matrix[:,:,:,2:], sub_matrix[:,:,:,2:], './test')

FlowInferTensor = torch.zeros(253,500,1000,4)

for lat_min in [0,50,100,150,200,250,300,350,400,450]:
    for lon_min in [0, 100, 200, 300, 400, 500, 600, 700, 800, 900]:
        lat_max = lat_min + 50
        lon_max = lon_min + 100

        save_name = 'lat_' + str(lat_min) + '_' + str(lat_max) \
                        + '_lon_' + str(lon_min) + '_' + str(lon_max)
        file_name = './PredictionTensor/fore_plus_trans/forecast_8days_transformer_60hrs_predict_'+save_name+'.pth'

        # load partial
        sub_matrix = torch.load(file_name)

        FlowInferTensor[:,lat_min:lat_max,lon_min:lon_max,:] = sub_matrix

torch.save(FlowInferTensor, './PredictionTensor/WholeFlowInfer/forecast_8days_transformer_60hrs_predict.pth')
print()


