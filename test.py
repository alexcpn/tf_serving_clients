__author__ = "Alex Punnen"
__date__ = "October 20 2020"

from cnn_client.tf_client import DetectObject
#from imutils.video import VideoStream, FileVideoStream
#from imutils.video import FPS
import argparse
import imutils
import cv2

if __name__ == "__main__":

    print("TF Serving Client Test")

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", type=str,
                    help="Path to input image file")
    ap.add_argument("-c", "--class", type=str, default="person",
                    help="Class to detect")
    ap.add_argument("-r", "--resize", type=bool, default=True,
                    help="Resize the image")
    ap.add_argument("-th", "--detection_threshold", type=float, default=0.50,
                    help="Threshold of Detection -between 0.1 and 0.9")
    ap.add_argument("-ip", "--detector_ip", type=str, default="127.0.0.1:8500",
                    help="TF Serving Server IP:PORT")
    ap.add_argument("-m", "--model", type=str, default="ssd",
                    help="TF Serving CNN Model name to map to example SSD")
    ap.add_argument("-o", "--out_file", type=str,default="out.jpg",
                    help="Path to output image file")                
    args = vars(ap.parse_args())

    print("Args=", args)
    model_name = args["model"].lower()
    label_to_detect = args["class"].lower()
    detection_threshold = args["detection_threshold"]
    out_file = args["out_file"]

    detectObject = DetectObject(args["detector_ip"], model_name)
    original_frame = cv2.imread(args["image"],cv2.IMREAD_COLOR)
    #if args["resize"]:
    #    frame = imutils.resize( original_frame, width=640)
    
    org_size = original_frame.shape[:2]  # H,W
    #new_size = frame.shape[:2]  # H,W
    print("original size = ",org_size,"org_size[0] = ",org_size[0] )

    boxes, scores, labels, num_detections = detectObject.do_inference(
                original_frame, org_size[0])
    if(num_detections == 0):
        print("Nothing detected")
    
    image = original_frame

    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        print("Score is ",score)
        # scores are sorted so we can break
        if score < detection_threshold:
            break
        # only if SSD we need to convert box to normal
        if (model_name == 'ssd'):
            box = detectObject.box_normal_to_pixel(box, image.shape)
        b = box.astype(int)
        class_label = detectObject.get_label(int(label))
        print("getObject Label ", class_label, " at ", b, " Score ", score)
        if(label_to_detect in class_label):
            print("Object detected")
            image =detectObject.output_results(
                image, b, score, class_label)
        cv2.imwrite(out_file, image)
        
        
