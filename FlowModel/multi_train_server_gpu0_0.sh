for lat_min in 100 150
do
	for lon_min in 0 100 200 300 400 500 600 700 800 900
	do
		python train_flow_transformer_server_gpu0.py $lat_min $lon_min
	done
done
