for lat_min in 300 350
do
	for lon_min in 0 100 200 300 400 500 600 700 800 900
	do
		python train_flow_transformer_server_gpu1.py $lat_min $lon_min
	done
done
