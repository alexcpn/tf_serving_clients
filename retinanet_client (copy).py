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

# for reitnanet image processing
from keras.preprocessing import image
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
import cv2
from PIL import Image

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
    parse_result(result_future.result())

def parse_result(result_future):
    """Callback function.
    Calculates the statistics for the prediction result.
    Args:
      result_future: Result future of the RPC.

    For the  example image the results would come like using retinanet pre-processing
    'in image shape', (800, 1067, 3))
    ('Input shape=', (1, 800, 1067, 3))
    ('in tf shape', (2, 800, 1067, 3))
    ('result no', 0)
    ('boxes output', (1, 300, 4))
    ('scores output', (1, 300))
    ('labels output', (1, 300))
    ('Label', 'person', ' at ', array([409, 167, 728, 603]), ' Score ', 0.9681119)
    ('Label', 'person', ' at ', array([  0, 426, 512, 785]), ' Score ', 0.8355836)
    ('Label', 'person', ' at ', array([ 723,  475, 1067,  791]), ' Score ', 0.72344124)
    ('Label', 'tie', ' at ', array([527, 335, 569, 505]), ' Score ', 0.525432)
    ('Time for ', 1, ' is ', 0.7898709774017334) in 1070 8Gb
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
    #image,org= decode_image_opencv(img_path)
    image,org = decode_image_retinanet(img_path)
    #image,org = decode_image_tf_reader(img_path)
    global _draw
    _draw = org.copy()
    print ("in image shape",image.shape)
    #('in image shape', (480, 640, 3))
    
    #input = np.expand_dims(image, axis=0)
    # do this once as tf. make_tensor proto is slow on first use
    #request.inputs['input_image'].CopyFrom(tf.contrib.util.make_tensor_proto \
    #  (input, shape=[1, 800, 1067, 3]))  # (input, shape=[1, 500, 567, 3]))
    
    global _start
    _start = time.time()
    global _response_awaiting
    _response_awaiting =True
    for i in range(num_tests):
      #print("Going to send the request")
      # batching  
      # If using anything other than decode_opencv uncomment line below   
      input = np.expand_dims(image, axis=0)
      #('Input shape=', (1, 480, 640, 3))
      #input = image # comment this if using anything other than retinanet
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


################################################################################
#  Helper functions  for image preoricessing                                   #
################################################################################
# Going to read the image via TF helper

  img_raw = tf.read_file(img_path)
  start = timer()
  img_tensor = tf.image.decode_jpeg(img_raw, channels=0,
        dct_method="INTEGER_FAST") #not much effect here decode time 30ms
  print("img_tensor.shape=",img_tensor.shape)
  i
def decode_image_retinanet(img_path):
  ## Going to read the image via retinanet helper- the original way
  start = timer()
  image = read_image_bgr(img_path)
  # preprocess image for network
  image = preprocess_image(image)
  image, scale = resize_image(image)
  end = timer()
  print("decode time=",end - start)
  #('decode time=', 0.028119802474975586)
  # these are the best scores
  #('Label', 'person', ' at ', array([409, 167, 728, 603]), ' Score ', 0.9681119)  
  #('Label', 'person', ' at ', array([  0, 426, 512, 785]), ' Score ', 0.8355836)
  #('Label', 'person', ' at ', array([ 723,  475, 1067,  791]), ' Score ', 0.72344124)
  #('Label', 'tie', ' at ', array([527, 335, 569, 505]), ' Score ', 0.525432)
  return image,image

def decode_image_tf_reader(img_path,max_height=480.0):
  #mage = tf.cast(img_tensor, tf.float32)
  
  #image = tf.image.resize_images(img_tensor, [800,1067])
  
  smallest_side = max_height # will losse some info
  height, width = tf.shape(image)[0], tf.shape(image)[1]
  height = tf.to_float(height)
  width = tf.to_float(width)
  
  scale = tf.cond(tf.greater(height, width),
                          lambda: smallest_side / width,
                          lambda: smallest_side / height)
  new_height = tf.to_int32(height * scale)
  new_width = tf.to_int32(width * scale)

  image = tf.image.resize_images(image, [new_height, new_width])

  #image = tf.image.resize_images(image, [800,1200])

  #https://forums.fast.ai/t/how-is-vgg16-mean-calculated/4577/19
  VGG_MEAN = [123.68, 116.78, 103.94] # This is R-G-B for Imagenet
  #means = tf.reshape(tf.constant(VGG_MEAN), [1, 1, 3])
  image = image - means 
  # swap to BGR    
  img_channel_swap = image[..., ::-1]
  image = tf.reverse(image, axis=[-1])
  #without the above preprocessing there is a miss of detection and change in weight
  image = tf.Session().run(image)
  #image = image[:, :, [2,1,0]] # swap channel from RGB to BGR
  end = timer()
  print("decode time=",end - start)
  #('decode time=', 0.032473087310791016)
  #('Label', 'person', ' at ', array([ 0, 252, 306, 470]), ' Score ', 0.8479492)
  #('Label', 'person', ' at ', array([241, 97, 435, 365]), ' Score ', 0.7183717)
  #('Label', 'person', ' at ', array([429, 287, 635, 475]), ' Score ', 0.67711633)
  #('Label', 'tie', ' at ', array([313, 199, 340, 311]), ' Score ', 0.5820301
  return image,image

def decode_image_opencv(img_path):
  ### Going to create image vector via OpenCV
  #todo https://docs.nvidia.com/deeplearning/sdk/dali-developer-guide/docs/examples/getting%20started.html
  start = timer()
  image = cv2.imread(img_path,1)
  image = image_resize(image,height=800)
  org  = image
  image = cv2.dnn.blobFromImage(image, scalefactor=1.0,mean=(103.939, 116.779, 123.68), swapRB=True)
  # this gives   shape as  (1, 3, 480, 640))
  # we need it as ('Input shape=', (1, 480, 640, 3))
  image = np.transpose(image, (0, 2, 3, 1))
  #image = image.astype('f')
  #image = image -127.5
  end = timer()
  #VGG_MEAN = [103.939, 116.779, 123.68]
  #out = np.copy(image) 
  #out = out[:, :, [2,1,0]] # swap channel from RGB to BGR
  print("decode time=",end - start)
  #'decode time=', 0.007803916931152344)
  return image,org

#https://stackoverflow.com/a/44659589/429476
def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized
  
def decode_image_pillow(img_path):
  ### Going to create image vector via PIL
  ## does not work
  start = timer()
  image = np.asarray(Image.open(img_path).convert('RGB'))
  image = image.astype('f')
  end = timer()
  print("decode time=",end - start)
  #('decode time=', 0.023459911346435547)
  # and without conversion as in 
  # https://github.com/fizyr/keras-retinanet/blob/master/keras_retinanet/utils/image.py
  # it is not able to detect one person object
  #Label', 'person', ' at ', array([  0, 258, 308, 473]), ' Score ', 0.8666027)
  #('Label', 'person', ' at ', array([239,  98, 427, 378]), ' Score ', 0.745802)
  #('Label', 'tie', ' at ', array([312, 199, 340, 314]), ' Score ', 0.7037207)
  return image

def create_dummy_image(width=1067,height=800,channels=3):
    #Create a random numpy array
    return numpy.random.rand(height,width, channels).astype('f')

def main(_):
  #img_path ='../examples/000000008021.jpg'
  #img_path ='../examples/business_people_highdefinition_picture_hd_pictures.jpg'
  #img_path ='../examples/Emeraude-Toubia-Hd-Photos-At-People-Choice-Awards-2018-2.jpg'
  #img_path ='../examples/google1.jpg'
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
    print ("Retinanet TFServing Client  < -num_tests=1 -server=127.0.0.1:8500 -batch_size=2>")
    print ("Override these default values by command line args")
    tf.app.run()