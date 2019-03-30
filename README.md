These are GRPC Clients for GRPC TF Serving compatible open source models like Resnet 50 , Retinanet etc


Server-
----------
docker run  --net=host --runtime=nvidia  -it --rm -p 8900:8500 -p 8901:8501 -v /home/alex/coding/IPython_neuralnet/models:/models -e MODEL_NAME=retinanet tensorflow/serving:latest-gpu --rest_api_port=0  --enable_batching=true  --model_config_file=/models/model_configs/retinanet.json

--other models
ssd_inception_v2_coco.json

Client 
----------------------------
docker run -it --runtime=nvidia  --net=host   -v /home/alex/coding/IPython_neuralnet:/coding --rm alexcpn/tfserving-keras-retinanet-dev-gpu
To run TF Client
cd /coding/tfserving_client
unset http_proxy
unset https_proxy
root@drone-OMEN:/coding/tfserving_client#python retinanet_client.py -num_tests=1 -server=127.0.0.1:8500 -batch_size=1 -img_path='../examples/google1.jpg'
other clients ssd_client.py

-Dev contianer with retinanet - alexcpn/tfserving-keras-retinanet-dev-gpu

