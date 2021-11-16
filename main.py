from read_data import *

flow_data_path = "Data/forecast_data_from_nov_9.npy"
flow_time_path = "Data/forecast_times_from_nov_9.npy"
flow_data = read_data(flow_data_path)
flow_time = read_data(flow_time_path)

print("Flow data has size: ", flow_data.shape)
print(np.max(flow_data))
