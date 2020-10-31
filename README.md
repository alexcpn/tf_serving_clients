This is a Generic GRPC Clients for Tensor Flow Serving compatible open source models like Resnet 50 , Retinanet, SSD etc

Full documentation - [Generic TensorFlow Client](https://medium.com/data-science-engineering/productising-tensorflow-keras-models-via-tensorflow-serving-69e191cb1f37)

## TensorFlow Server

We are using the models downloaded from here https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md

You can also convert to a TFServing compatible model (I will put the link here)

**With GPU**

```console
docker run  --net=host --runtime=nvidia  -it --rm -p 8900:8500 -p 8901:8501 -v /home/alex/coding/Prototypes/models:/models  tensorflow/serving:latest-gpu --rest_api_port=0  --enable_batching=true   --model_config_file=/models/model_configs/ssd_mobilenet_v1_0.75.json
```

Note - Please see this post regarding preparing your environment for GPU https://medium.com/data-science-engineering/install-the-right-nvidia-driver-for-cuda-in-ubuntu-2d9ade437dec

**With CPU**

```console
tf_serving_clients$ docker run  --net=host -it --rm -p 8900:8500 -p 8901:8501 -v$(pwd)/models:/models  tensorflow/serving:latest --rest_api_port=0  --enable_batching=true   --model_config_file=/models/model_configs/ssd_mobilenet_v1_0.75.json
```
 
 Note - If you get the error that `No versions of servable mask_rcnn_inception_resnet_v2 found under base path /models/mask_rcnn/mask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8/` this just means that TF Serving expects the saved model to be versioned, under a folder named with a version number of the model ex we put `1` for the SSD in this repo and `2` for MaskRCNN


Now on to the TF Client part. We will see how this client can work with any saved model

## TF Client

We will install and run the Client in the Tensorflow Jupyter notebook. We will use the volume mapping to compile inside this container.

**With GPU**

```console
tf_serving_clients$ docker run --entrypoint=/bin/bash   --runtime=nvidia  -it --rm  -v $(pwd):/tf_serving_clients --net=host tensorflow/tensorflow:2.3.1-gpu-jupyter
```

**With CPU**

```console
tf_serving_clients$ docker run --entrypoint=/bin/bash -it --rm  -v $(pwd):/tf_serving_clients --net=host tensorflow/tensorflow:2.3.1-jupyter
```

Install all the following after `cd`ing to tf_serving_client folder

```console
cd /tf_serving_clients
./setup.sh
./makeinstall.sh
```

After this you are ready to run the program. You can add new model parameters in the TFClient.py for now and do `makeinstall.sh` to rebuild.

(I will move this to yaml later)

```console
tf_serving_clients# python test.py -i image.jpg -o out.jpg -m ssd
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
                        TF Serving CNN Model name to map [ssd,maskrcnn]

  -o OUT_FILE, --out_file OUT_FILE
                        Path to output image file

```
And here is the result - Input Image

![input img](https://i.imgur.com/AieSIA4.jpg)

Result Output

![output image](https://i.imgur.com/nVHXKvP.jpg)


# Part 2

## Adding any Saved Model (TFServing compatible model) to the TF Client

Every model would have a different input and ouput interface. Let us use the MaskRCNN model and see how to add this to the TF Client

### TF Server

Let's use the MaskRCNN model from http://download.tensorflow.org/models/object_detection/tf2/20200711/mask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8.tar.gz 
( For making it TFServing compatible I just rename the SavedModel folder as `2` as TF Serving needs versioned folder)

```console
tf_serving_clients$ docker run  --net=host -it --rm -p 8900:8500 -p 8901:8501 -v $(pwd)/models:/models  tensorflow/serving:latest --rest_api_port=0  --enable_batching=false   --model_config_file=/models/model_configs/mask_rcnn_inception_resnet_v2.json
```
**Note** - I have set `enable_batching=false` here than `true` for the previous models. I was getting an error with output tensor not matching input tensor's 0th dimension here



## Modifying the TF Client for MaskRCNN

We will use the same development environment as above; That is the tensorflow jupyter docker container.


Step 1 is to find the model params. We can use the `saved_model_cli show` command to get the details of the model. We can run this in the TF Client container


```console
tf_serving_clients$ docker run --entrypoint=/bin/bash -it --rm  -v $(pwd):/tf_serving_clients --net=host tensorflow/tensorflow:2.3.1-jupyter

cd \tf_serving_clients

./setup.sh

tf_serving_clients# saved_model_cli show --dir ./models/mask_rcnn/mask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8/2 --all

MetaGraphDef with tag-set: 'serve' contains the following SignatureDefs:

signature_def['__saved_model_init_op']:
  The given SavedModel SignatureDef contains the following output(s):
    outputs['__saved_model_init_op'] tensor_info:
        dtype: DT_INVALID
        shape: unknown_rank
        name: NoOp
  Method name is: 

signature_def['serving_default']:
  The given SavedModel SignatureDef contains the following input(s):
    inputs['input_tensor'] tensor_info:
        dtype: DT_UINT8
        shape: (1, -1, -1, 3)
        name: serving_default_input_tensor:0
  The given SavedModel SignatureDef contains the following output(s):
    outputs['anchors'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1, 4)
        name: StatefulPartitionedCall:0
    outputs['box_classifier_features'] tensor_info:
        dtype: DT_FLOAT
        shape: (300, 9, 9, 1536)
        name: StatefulPartitionedCall:1
    outputs['class_predictions_with_background'] tensor_info:
        dtype: DT_FLOAT
        shape: (300, 91)
        name: StatefulPartitionedCall:2
    outputs['detection_anchor_indices'] tensor_info:
        dtype: DT_FLOAT
        shape: (1, 100)
        name: StatefulPartitionedCall:3
    outputs['detection_boxes'] tensor_info:
        dtype: DT_FLOAT
        shape: (1, 100, 4)
        name: StatefulPartitionedCall:4
    outputs['detection_classes'] tensor_info:
        dtype: DT_FLOAT
        shape: (1, 100)
        name: StatefulPartitionedCall:5
    outputs['detection_masks'] tensor_info:
        dtype: DT_FLOAT
        shape: (1, 100, 33, 33)
        name: StatefulPartitionedCall:6
    outputs['detection_multiclass_scores'] tensor_info:
        dtype: DT_FLOAT
        shape: (1, 100, 91)
        name: StatefulPartitionedCall:7
    outputs['detection_scores'] tensor_info:
        dtype: DT_FLOAT
        shape: (1, 100)
        name: StatefulPartitionedCall:8
...
Defined Functions:
  Function Name: '__call__'
    Option #1
      Callable with:
        Argument #1
          input_tensor: TensorSpec(shape=(1, None, None, 3), dtype=tf.uint8, name='input_tensor')
```

For object detection the following are important

```
  inputs['input_tensor'] tensor_info:
        dtype: DT_UINT8
        shape: (1, -1, -1, 3)
        name: serving_default_input_tensor:0

    outputs['detection_boxes'] tensor_info:
        dtype: DT_FLOAT
        shape: (1, 100, 4)
        name: StatefulPartitionedCall:4

    outputs['detection_classes'] tensor_info:
        dtype: DT_FLOAT
        shape: (1, 100)

    outputs['num_detections'] tensor_info:
        dtype: DT_FLOAT
        shape: (1)

There is of course the mask predictions - which the client as of now is not handling 

      outputs['mask_predictions'] tensor_info:
        dtype: DT_FLOAT
        shape: (100, 90, 33, 33)
        name: StatefulPartitionedCall:11
```

This is very similar to the params we have for SSD in tf_client.py.

For SSD

```
(tf_serving_clients) root@alex-swift:/tf_serving_clients# saved_model_cli show --dir ./models/ssd/ssd_mobilenet_v1_0.75_depth_300x300_coco14_sync_2018_07_03/1/

MetaGraphDef with tag-set: 'serve' contains the following SignatureDefs:

signature_def['serving_default']:
  The given SavedModel SignatureDef contains the following input(s):
    inputs['inputs'] tensor_info:
        dtype: DT_UINT8
        shape: (-1, -1, -1, 3)
        name: image_tensor:0
  The given SavedModel SignatureDef contains the following output(s):
    outputs['detection_boxes'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1, 100, 4)
        name: detection_boxes:0
    outputs['detection_classes'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1, 100)
        name: detection_classes:0
    outputs['detection_scores'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1, 100)
        name: detection_scores:0
    outputs['num_detections'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1)
        name: num_detections:0
```

 Lets use that as a base

```
        ssd_model_params = {
            "model": "SSD",
            "model_name": "ssd_inception_v1_coco",
            "signature": "serving_default",
            "input": "inputs",
            "input_dtype" : "DT_UINT8",
            "boxes": "detection_boxes",
            "scores": "detection_scores",
            "classes": "detection_classes",
            "num_detections": "num_detections",
            "IMAGENET_MEAN": (0, 0, 0),
            "swapRB": True,
            "label_map": _label_map_ssd
        }
```

And adapt to this new nodel.The model and model name are what we give in the model configuration json. The only change is for the input name which is `input_tensor` here.

The other two important ones are `label_map` and `IMAGENET_MEAN`.
As these models are trained for COCO 2017 dataset it should be same as our SSD model


```
maskrcnn_model_params = {
            "model": "mask_rcnn_inception",
            "model_name": "mask_rcnn_inception_resnet_v2",
            "signature": "serving_default",
            "input": "input_tensor",
            "boxes": "detection_boxes",
            "scores": "detection_scores",
            "classes": "detection_classes",
            "num_detections": "num_detections",
            "IMAGENET_MEAN": (0, 0, 0),
            "swapRB": True,
            "label_map": _label_map_ssd
        }
```

We can add this to `model_map` in `tf_client.py`. The only other change is done in the display for box co-ordinates in test.py

```
 # For certain models we need to convert box to normal
        if ( (model_name == 'maskrcnn') or (model_name == 'ssd') ):
            box = detectObject.box_normal_to_pixel(box, image.shape)
```

Now we just have to compile and run

```console
./make_install.sh
tf_serving_clients# python test.py -i image.jpg -o out.jpg -m maskrcnn
```

And here is the output

![maskrcnnoutput](https://imgur.com/IBIs3Mn.jpg)

