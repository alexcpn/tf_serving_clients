""" GRPC Client for Retinanet TF Serving Model"""
__author__ = Alex Punnem

import grpc
import numpy
import tensorflow as tf

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from keras.preprocessing import image
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
import time
import cv2
import numpy as np 


tf.app.flags.DEFINE_integer('num_tests', 1, 'Number of test images')
tf.app.flags.DEFINE_string('server', '127.0.0.1:8500', 'PredictionService host:port')
tf.app.flags.DEFINE_integer('batch_size', 2, 'No of images to batch -32,64,138  ')
FLAGS = tf.app.flags.FLAGS

_counter = 0
_start = 0
# load label to names mapping for visualization purposes
labels_to_names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}
_response_awaiting = True

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
  
      """
        Querying the save model gives
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
      Method name is: tensorflow/serving/predict
      ---
      From retina-net
      In general, inference of the network works as follows:
      boxes, scores, labels = model.predict_on_batch(inputs)
      Where `boxes` are shaped `(None, None, 4)` (for `(x1, y1, x2, y2)`), scores is shaped `(None, None)` (classification score) and labels is shaped `(None, None)` (label corresponding to the score). In all three outputs, the first dimension represents the shape and the second dimension indexes the list of detections.
    """
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
        if score < 0.5:
            break
        b = box.astype(int)
        print("Label",labels_to_names[label]," at ",b," Score ",score)
    _counter += 1
    #if( (_counter % 1) ==0):#print every 100
    #  print("[", _counter,"] From Callback Predicted Result is ", prediction,"confidence= ",response[prediction])
    if (_counter == FLAGS.num_tests):
        end = time.time()
        print("Time for ",FLAGS.num_tests," is ",end -_start)
        _response_awaiting = False

    
def do_inference(server, batch_size, num_tests):    
    channel = grpc.insecure_channel(server)            
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)      
    request = predict_pb2.PredictRequest()                      
    request.model_spec.name = 'retinanet'                        
    request.model_spec.signature_name = 'serving_default'
  
    # Going to read the image
    image = read_image_bgr('../examples/000000008021.jpg')
    # copy to draw on
    draw = image.copy()
    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

    # preprocess image for network
    image = preprocess_image(image)
    image, scale = resize_image(image)
    
    #image= numpy.random.rand(400, 567, 3).astype('f')
    # trying with real image
    print ("in image shape",image.shape)
    #input = np.expand_dims(image, axis=0)
    #request.inputs['input_image'].CopyFrom(tf.contrib.util.make_tensor_proto \
    #  (input, shape=[1, 800, 1067, 3]))  # (input, shape=[1, 500, 567, 3]))
    
    global _start
    _start = time.time()
    global _response_awaiting
    _response_awaiting =True
    for i in range(num_tests):
      #print("Going to send the request")
      # try batching      
      input = np.expand_dims(image, axis=0)
      inputs = np.append(input, input, axis=0)
      for _ in range(batch_size-2):
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

    while(_response_awaiting):
      time.sleep(.000010)
    print("Response Received Exiting")




def main(_):
  if not FLAGS.num_tests:
      print('Please specify num_tests -num_tests=n')
      return
  if not FLAGS.server:
    print('please specify server -server host:port')
    return
  print("Number of test=",FLAGS.num_tests)
  error_rate = do_inference(FLAGS.server, FLAGS.work_dir,
                            FLAGS.concurrency, FLAGS.num_tests) 
          
if __name__ == '__main__':
    print ("Retinanet TFServing Client  < -num_tests=1 -server=127.0.0.1:8500 -batch_size=2>")
    print ("Override these default values by command line args")
    tf.app.run()
