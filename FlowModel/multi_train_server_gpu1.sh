for lat_min in 200 250
do
	for lon_min in 0 100 200 300 400 500 600 700 800 900
	do
		python train_flow_transformer.py $lat_min $lon_min
	done
done
