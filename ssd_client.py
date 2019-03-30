""" GRPC Client for SSD TF Serving Model"""

from __future__ import division
__author__ = "Alex Punnen"
__date__  = "March 2019"

import grpc
import numpy
import tensorflow as tf
import time
import os
import numpy as np 
from timeit import default_timer as timer

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from image_preprocessor import decode_image_opencv

import cv2
#protoc --proto-path=. --python_out=. string_int_label_map.proto
# load label to names mapping for visualization purposes
#https://github.com/tensorflow/models/blob/master/research/object_detection/protos/string_int_label_map.proto

import string_int_label_map_pb2 as labelmap
from google.protobuf import text_format

tf.app.flags.DEFINE_integer('num_tests', 1, 'Number of test images')
tf.app.flags.DEFINE_string('server', '127.0.0.1:8500', 'PredictionService host:port')
tf.app.flags.DEFINE_integer('batch_size', 1, 'No of images to batch -32,64,138  ')
tf.app.flags.DEFINE_string('img_path', '', 'realtive/fullpath to jpegfile')
FLAGS = tf.app.flags.FLAGS

_counter = 0
_start = 0
_draw = None
_label_map = None

# labels are in mscoco_complete_label_map.pbtx
#https://github.com/tensorflow/models/tree/master/research/object_detection/data
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('.', 'mscoco_label_map.pbtxt')
# lets load the labels to the protobuf

_response_awaiting = True

"""
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
    Method name is: tensorflow/serving/predict
   
"""

def box_normal_to_pixel(box, dim,scalefactor=1):
    # https://github.com/tensorflow/models/blob/master/research/object_detection/utils/visualization_utils.py
    height, width = dim[0], dim[1]
    ymin = int(box[0]*height*scalefactor)
    xmin = int(box[1]*width*scalefactor)

    ymax = int(box[2]*height*scalefactor)
    xmax= int(box[3]*width*scalefactor)
    return np.array([xmin,ymin,xmax,ymax])   

def get_label(index):
  global _label_map
  return _label_map.item[index].display_name

def _callback(result_future):
    """Callback function.
    Calculates the statistics for the prediction result.
    Args:
      result_future: Result future of the RPC.
    """
    #print ("Something came - Hee haw")
    global _counter 
    global _start
    global _response_awaiting
  
    exception = result_future.exception()
    if exception:
      print(exception)
  

    parse_result(result_future.result())


    
def parse_result(result_future):
    """Callback function.
    Calculates the statistics for the prediction result.
    Args:
      result_future: Result future of the RPC.

    """
    global _counter 
    global _start
    global _response_awaiting
    global _draw

    boxes = result_future.\
          outputs['detection_boxes']
    scores = result_future.\
          outputs['detection_scores']
    labels = result_future.\
          outputs['detection_classes']
    num_detections= result_future.\
          outputs['num_detections']

    boxes= tf.make_ndarray(boxes)
    scores= tf.make_ndarray(scores)
    labels= tf.make_ndarray(labels)
    num_detections= tf.make_ndarray(num_detections)

    print("result no",_counter)
    print("boxes output",(boxes).shape)
    print("scores output",(scores).shape)
    print("labels output",(labels).shape)
    print('num_detections',num_detections[0])

    # visualize detections hints from 
    # # https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb

    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        # scores are sorted so we can break
        if score < 0.3:
            break
        #dim = image.shape[0:2]
        dim = _draw.shape
        #print("Label-raw",labels_to_names[label-1]," at ",box," Score ",score)
        box = box_normal_to_pixel(box, dim)
        b = box.astype(int)
        class_label = get_label(int(label))
        print("Label",class_label ," at ",b," Score ",score)
        # draw the image and write out
        cv2.rectangle(_draw,(b[0],b[1]),(b[2],b[3]),(0,0,255),1)
        cv2.putText(_draw,class_label + "-"+str(round(score,2)), (b[0]+2,b[1]+8),\
           cv2.FONT_HERSHEY_SIMPLEX, .45, (0,0,255))
        

    _counter += 1
    #if( (_counter % 1) ==0):#print every 100
    #  print("[", _counter,"] From Callback Predicted Result is ", prediction,"confidence= ",response[prediction])
    if (_counter == FLAGS.num_tests):
        end = time.time()
        print("Time for ",FLAGS.num_tests," is ",end -_start)
        _response_awaiting = False
        cv2.imwrite("/coding/out.png", _draw)

    
def do_inference(server, batch_size, num_tests,img_path):    

    channel = grpc.insecure_channel(server)            
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)      
    request = predict_pb2.PredictRequest()                      
    request.model_spec.name = 'ssd_inception_v2_coco'                        
    request.model_spec.signature_name = 'serving_default'
    print("Image path",img_path)
    #post process the image
    image,org= decode_image_opencv(img_path,max_height=800)
    #image,org = decode_image_tf_reader(img_path,max_height=800)
    image = image.astype(np.uint8)
    global _draw
    _draw = org.copy()
    print ("in image shape",image.shape)
    
    
    global _start
    _start = time.time()
    global _response_awaiting
    _response_awaiting =True
    for i in range(num_tests):
      #print("Going to send the request")
      # batching  
      # If using anything other than decode_opencv uncomment line below   
      #input = np.expand_dims(image, axis=0)
      #('Input shape=', (1, 480, 640, 3))
      input = image # comment this if using anything other than retinanet
      print("Input shape=",input.shape )
      inputs = input
      for _ in range(batch_size-1):
        inputs = np.append(inputs, input, axis=0)
      
      print("Input-s shape",inputs.shape)
      request.inputs['inputs'].CopyFrom(tf.contrib.util.make_tensor_proto
            (inputs, shape=inputs.shape))
      
      # call back way - this is faster
      result_future = stub.Predict.future(request, 60.25)  # Intial takes time  
      result_future.add_done_callback(_callback)

      # request reponse way - this is slower
      # result = stub.Predict(request, 10.25)  #  seconds  
      # parse_result(result)
      _response_awaiting = True

      #print("Send the request")
      # End for loop

    while(_response_awaiting):
      time.sleep(.000010)
    print("Response Received Exiting")


def main(_):

  if not FLAGS.img_path:
      img_path = FLAGS.img_path
  if not FLAGS.num_tests:
      print('Please specify num_tests -num_tests=n')
      return
  if not FLAGS.server:
    print('please specify server -server host:port')
    return
  print("Number of test=",FLAGS.num_tests)
  s = open('mscoco_complete_label_map.pbtxt','r').read()
  mymap =labelmap.StringIntLabelMap()
  global _label_map 
  _label_map = text_format.Parse(s,mymap)
  

  do_inference(FLAGS.server, FLAGS.batch_size,
                             FLAGS.num_tests,FLAGS.img_path) 
          
if __name__ == '__main__':
    """
    Server 
     docker run  --net=host --runtime=nvidia  -it --rm -p 8900:8500 -p 8901:8501 
      -v /home/alex/coding/IPython_neuralnet/models:/models -e MODEL_NAME=ssd
       tensorflow/serving:latest-gpu --rest_api_port=0  --enable_batching=true
         --model_config_file=/models/model_configs/ssd_inception_v2_coco.json
    """
    print ("SSD TFServing Client  < -num_tests=1 -server=127.0.0.1:8500 -batch_size=2>")
    print ("Override these default values by command line args")
    tf.app.run()