################################################################################
#  Helper functions  for image preoricessing                                   #
################################################################################
__author__ = "Alex Punnen"
__date__  = "March 2019"

from timeit import default_timer as timer
import numpy as np 
import cv2

def decode_image_opencv(img_path,max_height=800):
  ### Going to create image vector via OpenCV
  #todo https://docs.nvidia.com/deeplearning/sdk/dali-developer-guide/docs/examples/getting%20started.html
  start = timer()
  image = cv2.imread(img_path,1)
  print("original image shape=",image.shape)
  (h, w) = image.shape[:2]
  print("Scale factor=", h/max_height) #we are currenly drawing in original so this is not relevant
  image = image_resize(image,height=max_height)
  org  = image
  #VGG_MEAN = [103.939, 116.779, 123.68]
  image = cv2.dnn.blobFromImage(image, scalefactor=1.0,mean=(103.939, 116.779, 123.68), swapRB=True)
  # this gives   shape as  (1, 3, 480, 640))
  image = np.transpose(image, (0, 2, 3, 1))
  # we get it after transpose as ('Input shape=', (1, 480, 640, 3))
  print("resized image shape=",image.shape)
  # for original image we take the first image, (the first dim is number of images)
  #org = image[0,:,:,:] 
  #print("Draw shape=",org.shape)
  end = timer()
  print("decode time=",end - start)
  return image,org

#https://stackoverflow.com/a/44659589/429476
# It is important to resize without loosing the aspect ratio for 
# good detection
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
  

def create_dummy_image(width=1067,height=800,channels=3):
    #Create a random numpy array
    return np.random.rand(height,width, channels).astype('f')

################################################################################
# Other functions

from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image

def decode_image_retinanet(img_path,max_height=None):
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

################################################################################

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
