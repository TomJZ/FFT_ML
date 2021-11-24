for lat_min in 0 50 100 150 200 250 300 350 400 450
do
	for lon_min in 0 100 200 300 400 500 600 700 800 900
	do
		python infer_flow_transformer.py $lat_min $lon_min
	done
done
