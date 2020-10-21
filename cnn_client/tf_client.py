""" GRPC Client for TF Serving Model"""

__author__ = "Alex Punnen"
__date__ = "March 2019"

from __future__ import division

from google.protobuf import text_format
import cnn_client.cnncli_generated.string_int_label_map_pb2 as labelmap
import cv2
from cnn_client.image_preprocessor import decode_image_opencv
from tensorflow_serving.apis import prediction_service_pb2_grpc
from tensorflow_serving.apis import predict_pb2
import queue
from timeit import default_timer as timer
import grpc
import numpy
import tensorflow as tf
import time
import os
import numpy as np

# protoc --proto-path=. --python_out=. string_int_label_map.proto
# load label to names mapping for visualization purposes
# https://github.com/tensorflow/models/blob/master/research/object_detection/protos/string_int_label_map.proto

# labels are in mscoco_complete_label_map.pbtx
# https://github.com/tensorflow/models/tree/master/research/object_detection/data
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('.', 'mscoco_label_map.pbtxt')
# lets load the labels to the protobuf
templates_dir = os.path.join(os.path.dirname(__file__), 'resources')

class DetectObject:

    def __init__(self, server,model_name):
        self.server = server
        self._response_awaiting = True
        file_path = os.path.join(templates_dir, \
            'mscoco_complete_label_map.pbtxt')
        s = open(file_path, 'r').read()
        mymap = labelmap.StringIntLabelMap()
        _label_map_ssd = text_format.Parse(s, mymap)

        file_path = os.path.join(templates_dir, \
            'retinanet_complete_label_map.pbtxt')
        s = open(file_path, 'r').read()
        mymap = labelmap.StringIntLabelMap()
        _label_map_retinanet = text_format.Parse(s, mymap)


        self.channel = grpc.insecure_channel(self.server)
        self.stub = prediction_service_pb2_grpc.PredictionServiceStub(self.channel)
        
        self.Q = queue.Queue()
        ssd_model_params = {
            "model": "SSD",
            "model_name": "ssd_inception_v2_coco",
            "signature": "serving_default",
            "input": "inputs",
            "boxes": "detection_boxes",
            "scores": "detection_scores",
            "classes": "detection_classes",
            "num_detections": "num_detections",
            "IMAGENET_MEAN": (0, 0, 0),
            "swapRB": True,
            "label_map": _label_map_ssd
        }

        retinanet_model_params = {
            "model": "RETINANET",
            "model_name": "retinanet",
            "signature": "serving_default",
            "input": "input_image",
            "boxes": "filtered_detections/map/TensorArrayStack/TensorArrayGatherV3:0",
            "scores": "filtered_detections/map/TensorArrayStack_1/TensorArrayGatherV3:0",
            "classes": "filtered_detections/map/TensorArrayStack_2/TensorArrayGatherV3:0",
            #"IMAGENET_MEAN": (103.939, 116.779, 123.68),
            "IMAGENET_MEAN": (0, 0, 0),
            "swapRB": True,
            "label_map": _label_map_retinanet
        }

        self.model_params = ssd_model_params
        self.model_name = model_name.lower()
        
        if(self.model_name == "ssd"):
          self.model_params = ssd_model_params
        elif(self.model_name == "retinanet"):
          self.model_params = retinanet_model_params
        

    def get_label(self, index):
        return self.model_params["label_map"].item[index].display_name

    def _callback(self, result_future):
        """Callback function.
        Calculates the statistics for the prediction result.
        Args:
          result_future: Result future of the RPC.
        """
        #print ("Something came - Hee haw")

        exception = result_future.exception()
        if exception:
            print("Got an error", exception)
            self._response_awaiting = False
        boxes, scores, labels, num_detections = \
            self.parse_result(result_future.result())
        self.Q.put(boxes, scores, labels, num_detections)

    def parse_result(self, result_future):
        """Callback function.
        Calculates the statistics for the prediction result.
        Args:
          result_future: Result future of the RPC.

        """

        boxes = result_future.\
            outputs[self.model_params["boxes"]]
        scores = result_future.\
            outputs[self.model_params["scores"]]
        labels = result_future.\
            outputs[self.model_params["classes"]]
        if (self.model_params["model"] == 'SSD'):
            num_detections = result_future.\
                outputs[self.model_params["num_detections"]]
            num_detections = tf.make_ndarray(num_detections)
        else:
            num_detections = None
        
        boxes = tf.make_ndarray(boxes)
        scores = tf.make_ndarray(scores)
        labels = tf.make_ndarray(labels)

        # For 100 labels/classes you get something like
        # print("boxes output",(boxes).shape)
        #print("scores output",(scores).shape)
        #print("labels output",(labels).shape)
        #print('num_detections',num_detections[0])
        # this 
        # 'boxes output',  #(1, 100, 4))
        #('scores output', (1, 100))
        #('labels output', (1, 100))

        return boxes, scores, labels, num_detections

    def visualize_results(self, image, box, score, class_label):
        b = box
        cv2.putText(image, class_label + "-"+str(round(score, 2)), (b[0]+2, b[1]+8),
                    cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 0, 255))
        cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), (0, 100, 255), 1)
        cv2.imshow("Frame", image)

    def output_results(self, image, box, score, class_label):
        b = box
        cv2.putText(image, class_label + " " +str(round(score, 2)), (b[0]+2, b[1]+8),
                    cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 0, 255))
        cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), (0, 100, 255), 1)
        cv2.putText(image,class_label + " " +str(round(score,2)), (b[0]+2,b[1]+8),\
           cv2.FONT_HERSHEY_SIMPLEX, .45, (0,0,255))
        return image


    def do_inference(self, image, height=800, batch_size=1):

        request = predict_pb2.PredictRequest()
        request.model_spec.name = self.model_params["model_name"]
        request.model_spec.signature_name = self.model_params["signature"]
        
        # post process the image
        IMAGENET_MEAN = self.model_params["IMAGENET_MEAN"]
        swapRB = self.model_params["swapRB"]
        image, org = decode_image_opencv(image, max_height=height, swapRB=swapRB,
                                         imagenet_mean=IMAGENET_MEAN)
        if self.model_params["model"] == "SSD":
          image = image.astype(np.uint8)
        #retinanet needs as float
        self._draw = org.copy()
        #print ("in image shape",image.shape)

        self._start = time.time()
        self._response_awaiting = True
        #print("Going to send the request")
        # batching
        # If using anything other than decode_opencv uncomment line below
        #input = np.expand_dims(image, axis=0)
        #('Input shape=', (1, 480, 640, 3))
        input = image  # comment this if using anything other than retinanet
        #print("Input shape=",input.shape )
        inputs = input
        for _ in range(batch_size-1):
            inputs = np.append(inputs, input, axis=0)

        #print("Input-s shape",inputs.shape)
        # note - original is inputs, for V100 I saved modesl with 'input_image'
        # Pre TF 2.0
        #request.inputs[self.model_params["input"]].CopyFrom(tf.contrib.util.make_tensor_proto
        #                                  (inputs, shape=inputs.shape))

        request.inputs[self.model_params["input"]].CopyFrom(tf.make_tensor_proto
                                          (inputs, shape=inputs.shape))

        # call back way - this is faster
        # result_future = stub.Predict.future(request, 60.25)  # Intial takes time
        # result_future.add_done_callback(self._callback)
        # self._response_awaiting = True

        # request reponse way - this is slower
        result = self.stub.Predict(request, 10.25)  # seconds
        boxes, scores, labels, num_detections = self.parse_result(result)
        #print("No of detections=",num_detections)
        return boxes, scores, labels, num_detections

    def box_normal_to_pixel(self, box, dim, scalefactor=1):
        # https://github.com/tensorflow/models/blob/master/research/object_detection/utils/visualization_utils.py
        height, width = dim[0], dim[1]
        ymin = int(box[0]*height*scalefactor)
        xmin = int(box[1]*width*scalefactor)
        ymax = int(box[2]*height*scalefactor)
        xmax = int(box[3]*width*scalefactor)
        return np.array([xmin, ymin, xmax, ymax])
