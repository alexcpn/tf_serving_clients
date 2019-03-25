""" GRPC Client for Resnet TF Serving Model
MODEL = https://github.com/tensorflow/models/tree/master/official/resnet
FP32 = http://download.tensorflow.org/models/official/20181001_resnet/savedmodels/resnet_v2_fp32_savedmodel_NCHW.tar.gz
FP16 =  http://download.tensorflow.org/models/official/20181001_resnet/savedmodels/resnet_v2_fp16_savedmodel_NCHW.tar.gz
Model Extracted and RUN
docker run  --net=host --runtime=nvidia  -it --rm -p 8900:8500 -p 8901:8501 
 -v /home/alex/coding/IPython_neuralnet/:/models/ tensorflow/serving:latesgpu 
  --model_config_file=/models/resnet50_fp32.json or resnet50_fp16.json
Results =

"""
__author__ = "Alex Punnen"
__date__  = "March 2019"

import grpc
import numpy
import tensorflow as tf

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
import time
import cv2
import numpy as np 

tf.app.flags.DEFINE_integer('num_tests', 1, 'Number of test images')
tf.app.flags.DEFINE_string('server', '127.0.0.1:8500', 'PredictionService host:port')
tf.app.flags.DEFINE_integer('batch_size', 2, 'No of images to batch -32,64,138  ')
tf.app.flags.DEFINE_string('model_name', 'resnet_32', 'Either resnet_32 or resnet_16')

FLAGS = tf.app.flags.FLAGS

_counter = 0
_start = 0
# load label to names mapping for visualization purposes
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
        MetaGraphDef with tag-set: 'serve' contains the following SignatureDefs:

        signature_def['predict']:
        The given SavedModel SignatureDef contains the following input(s):
            inputs['input'] tensor_info:
                dtype: DT_FLOAT
                shape: (64, 224, 224, 3)
                name: input_tensor:0
        The given SavedModel SignatureDef contains the following output(s):
            outputs['classes'] tensor_info:
                dtype: DT_INT64
                shape: (64)
                name: ArgMax:0
            outputs['probabilities'] tensor_info:
                dtype: DT_FLOAT
                shape: (64, 1001)
                name: softmax_tensor:0
        Method name is: tensorflow/serving/predict

        signature_def['serving_default']:
        The given SavedModel SignatureDef contains the following input(s):
            inputs['input'] tensor_info:
                dtype: DT_FLOAT
                shape: (64, 224, 224, 3)
                name: input_tensor:0
        The given SavedModel SignatureDef contains the following output(s):
            outputs['classes'] tensor_info:
                dtype: DT_INT64
                shape: (64)
                name: ArgMax:0
            outputs['probabilities'] tensor_info:
                dtype: DT_FLOAT
                shape: (64, 1001)
                name: softmax_tensor:0
        Method name is: tensorflow/serving/predict
            ---
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

    scores = result_future.\
          outputs['probabilities']
    labels = result_future.\
          outputs['classes']
    scores= tf.make_ndarray(scores)
    labels= tf.make_ndarray(labels)

    print("result no",_counter)
    print("scores output",(scores).shape)
    print("labels output",(labels).shape)

    # visualize detections
    for   label,score in zip(labels,scores):
        # scores are sorted so we can break
        #print("Label",label," Score ",score)
        print("Label",label)
    _counter += 1
    #if( (_counter % 1) ==0):#print every 100
    #  print("[", _counter,"] From Callback Predicted Result is ", prediction,"confidence= ",response[prediction])
    if (_counter == FLAGS.num_tests):
        end = time.time()
        print("Time for ",FLAGS.num_tests," is ",end -_start)
        _response_awaiting = False

    
def do_inference(server, model_name, batch_size, num_tests):    
    channel = grpc.insecure_channel(server)            
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)      
    request = predict_pb2.PredictRequest()                      
    request.model_spec.name =  model_name                        
    request.model_spec.signature_name = 'serving_default'
  
    # Going to read the image
    #image = read_image_bgr('../examples/000000008021.jpg')
    # copy to draw on
    #draw = image.copy()
    #draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

    
    image= numpy.random.rand(224, 224, 3).astype('f')
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
      request.inputs['input'].CopyFrom(tf.contrib.util.make_tensor_proto
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
  error_rate = do_inference(FLAGS.server, FLAGS.model_name,
                            FLAGS.batch_size, FLAGS.num_tests) 
          
if __name__ == '__main__':
    print ("TFServing ResNet Client  < -num_tests=1 -server=127.0.0.1:8500 -batch_size=2 -model-name=resnet_32 (or16)>")
    print ("Override these default values by command line args")
    # ex python resnet50_32_client.py -model_name=resnet_16 -batch_size=64
    # or python resnet50_32_client.py -model_name=resnet_32 -batch_size=64
    tf.app.run()
