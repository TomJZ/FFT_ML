import numpy as np

def read_data(file_path):
    flow_data = np.load(file_path)
    return flow_data