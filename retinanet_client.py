""" GRPC Client for Retinanet TF Serving Model"""
__author__ = "Alex Punnen"
__date__  = "March 2019"

import grpc
import numpy
import tensorflow as tf
import time
import numpy as np 
from timeit import default_timer as timer


from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

from image_preprocessor import decode_image_opencv
from keras_retinanet.utils.visualization import draw_box,label_color, draw_caption
import cv2

tf.app.flags.DEFINE_integer('num_tests', 1, 'Number of test images')
tf.app.flags.DEFINE_string('server', '127.0.0.1:8500', 'PredictionService host:port')
tf.app.flags.DEFINE_integer('batch_size', 1, 'No of images to batch -32,64,138  ')
tf.app.flags.DEFINE_string('img_path', '', 'realtive/fullpath to jpegfile')
FLAGS = tf.app.flags.FLAGS

_counter = 0
_start = 0
_draw = None
# load label to names mapping for visualization purposes
labels_to_names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}
_response_awaiting = True


""" 
  MetaGraphDef with tag-set: 'serve' contains the following SignatureDefs:

  signature_def['serving_default']:
  The given SavedModel SignatureDef contains the following input(s):
    inputs['input_image'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1, -1, -1, 3)
        name: input_1_2:0
  The given SavedModel SignatureDef contains the following output(s):
    outputs['filtered_detections/map/TensorArrayStack/TensorArrayGatherV3:0'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1, 300, 4)
        name: filtered_detections/map/TensorArrayStack/TensorArrayGatherV3:0
    outputs['filtered_detections/map/TensorArrayStack_1/TensorArrayGatherV3:0'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1, 300)
        name: filtered_detections/map/TensorArrayStack_1/TensorArrayGatherV3:0
    outputs['filtered_detections/map/TensorArrayStack_2/TensorArrayGatherV3:0'] tensor_info:
        dtype: DT_INT32
        shape: (-1, 300)
        name: filtered_detections/map/TensorArrayStack_2/TensorArrayGatherV3:0

  From retina-net
  In general, inference of the network works as follows:
  boxes, scores, labels = model.predict_on_batch(inputs)
  Where `boxes` are shaped `(None, None, 4)` (for `(x1, y1, x2, y2)`), scores is shaped `(None, None)` (classification score) and labels is shaped `(None, None)` (label corresponding to the score). In all three outputs, the first dimension represents the shape and the second dimension indexes the list of detections.
"""

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
          outputs['filtered_detections/map/TensorArrayStack/TensorArrayGatherV3:0']
    scores = result_future.\
          outputs['filtered_detections/map/TensorArrayStack_1/TensorArrayGatherV3:0']
    labels = result_future.\
          outputs['filtered_detections/map/TensorArrayStack_2/TensorArrayGatherV3:0']
    boxes= tf.make_ndarray(boxes)
    scores= tf.make_ndarray(scores)
    labels= tf.make_ndarray(labels)

    print("result no",_counter)
    print("boxes output",(boxes).shape)
    print("scores output",(scores).shape)
    print("labels output",(labels).shape)

    # visualize detections
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        # scores are sorted so we can break
        if score < 0.3:
            break
        b = box.astype(int)
        print("Label",labels_to_names[label]," at ",b," Score ",score)
        # draw the image and write out
        color = label_color(label)
        draw_box(_draw, b, color=color)
        caption = "{} {:.3f}".format(labels_to_names[label], score)
        draw_caption(_draw, b, caption)
        

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
    request.model_spec.name = 'retinanet'                        
    request.model_spec.signature_name = 'serving_default'
    print("Image path",img_path)
    #post process the image
    image,org= decode_image_opencv(img_path,max_height=800)
    #image,org = decode_image_retinanet(img_path,max_height=800)
    #image,org = decode_image_tf_reader(img_path)
    global _draw
    _draw = org.copy()
    print ("in image shape",image.shape)
    #('in image shape', (480, 640, 3))

    
    global _start
    _start = time.time()
    global _response_awaiting
    _response_awaiting =True
    for i in range(num_tests):
      #print("Going to send the request")
      # batching  
      # If using anything other than decode_opencv uncomment line below   
     # input = np.expand_dims(image, axis=0)
      #('Input shape=', (1, 480, 640, 3))
      input = image # comment this if using anything other than retinanet
      print("Input shape=",input.shape )
      inputs = input
      for _ in range(batch_size-1):
        inputs = np.append(inputs, input, axis=0)
      
      print("in tf shape",inputs.shape)
      request.inputs['input_image'].CopyFrom(tf.contrib.util.make_tensor_proto
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
  error_rate = do_inference(FLAGS.server, FLAGS.batch_size,
                             FLAGS.num_tests,FLAGS.img_path) 
          
if __name__ == '__main__':
    """
    Server 
     docker run  --net=host --runtime=nvidia  -it --rm -p 8900:8500 -p 8901:8501 
      -v /home/alex/coding/IPython_neuralnet/models:/models -e MODEL_NAME=ssd
       tensorflow/serving:latest-gpu --rest_api_port=0  --enable_batching=true
         --model_config_file=//models/model_configs/retinanet.json
    """
    print ("Retinanet TFServing Client  < -num_tests=1 -server=127.0.0.1:8500 -batch_size=2>")
    print ("Override these default values by command line args")
    tf.app.run()