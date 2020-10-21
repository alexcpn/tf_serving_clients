This is a Generic GRPC Clients for Tensor Flow Serving compatible open source models like Resnet 50 , Retinanet, SSD etc

Full documentation - [Generic TensorFlow Client](https://medium.com/data-science-engineering/productising-tensorflow-keras-models-via-tensorflow-serving-69e191cb1f37)

## Server


**With GPU**

```console
docker run  --net=host --runtime=nvidia  -it --rm -p 8900:8500 -p 8901:8501 -v /home/alex/coding/Prototypes/models:/models  tensorflow/serving:latest-gpu --rest_api_port=0  --enable_batching=true   --model_config_file=/models/model_configs/ssd_mobilenet_v1_0.75.json
```

Note - Please see this post regarding preparing your environment for GPU https://medium.com/data-science-engineering/install-the-right-nvidia-driver-for-cuda-in-ubuntu-2d9ade437dec

**With CPU**

```console
docker run  --net=host -it --rm -p 8900:8500 -p 8901:8501 -v /home/alex/coding/Prototypes/models:/models  tensorflow/serving:latest --rest_api_port=0  --enable_batching=true   --model_config_file=/models/model_configs/ssd_mobilenet_v1_0.75.json
```

## Client

**With GPU**

```console
docker run --entrypoint=/bin/bash   --runtime=nvidia  -it --rm  -v /home/alex/:/alex --net=host tensorflow/tensorflow:2.3.1-gpu-jupyter
```

**With CPU**

```console
docker run --entrypoint=/bin/bash -it --rm  -v /home/alex/:/alex --net=host tensorflow/tensorflow:2.3.1-jupyter
```

Install all the following after `cd`ing to tf_serving_client folder

```console
pip install pipenv
pipenv install
pipenv install --dev
pipenv shell
make all
./install.sh
```

After this you are ready to run the program

```console
tf_serving_clients# python test.py -i image.jpg -o out.jpg
```

Default threshold is .5 and can be set by `-th` option in command line

Here are the other options

```
TF Serving Client Test
usage: test.py [-h] [-i IMAGE] [-c CLASS] [-r RESIZE]
               [-th DETECTION_THRESHOLD] [-ip DETECTOR_IP] [-m MODEL]
               [-o OUT_FILE]

optional arguments:
  -h, --help            show this help message and exit
  -i IMAGE, --image IMAGE
                        Path to input image file
  -c CLASS, --class CLASS
                        Class to detect
  -r RESIZE, --resize RESIZE
                        Resize the image
  -th DETECTION_THRESHOLD, --detection_threshold DETECTION_THRESHOLD
                        Threshold of Detection -between 0.1 and 0.9
  -ip DETECTOR_IP, --detector_ip DETECTOR_IP
                        TF Serving Server IP:PORT
  -m MODEL, --model MODEL
                        TF Serving CNN Model name to map to example SSD
  -o OUT_FILE, --out_file OUT_FILE
                        Path to output image file

```
And here is the result - Input Image

![input img](https://i.imgur.com/AieSIA4.jpg)

Result Output

![output image](https://i.imgur.com/nVHXKvP.jpg)


