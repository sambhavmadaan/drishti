# USAGE
# python pi_object_detection.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel

# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
from multiprocessing import Process
from multiprocessing import Queue
import numpy as np
import argparse
import imutils
import time
import cv2
import os
import RPi.GPIO as GPIO
import threading
import time
from gtts import gTTS

GPIO.setwarnings(False)

def sonic_one():
    GPIO.setmode(GPIO.BCM)

    TRIG = 23

    ECHO = 24

    #print ("Distance 1 Measurement In Progress")

    GPIO.setup(TRIG,GPIO.OUT)

    GPIO.setup(ECHO,GPIO.IN)

    GPIO.output(TRIG, False)

    #print ("Waiting For Sensor1 To Settle")

    time.sleep(1)

    GPIO.output(TRIG, True)

    time.sleep(0.00001)

    GPIO.output(TRIG, False)

    while GPIO.input(ECHO)==0:

      pulse_start = time.time()

    while GPIO.input(ECHO)==1:

      pulse_end = time.time()
      
    pulse_duration = pulse_end - pulse_start

    distance = pulse_duration * 17150

    distance = round(distance, 2)

    #print ("Distance1:",distance,"cm")

    GPIO.cleanup()
    return distance

    
def sonic_two():
    GPIO.setmode(GPIO.BCM)

    TRIG2 = 17

    ECHO2 = 18

    #print ("Distance2 Measurement In Progress")

    GPIO.setup(TRIG2,GPIO.OUT)

    GPIO.setup(ECHO2,GPIO.IN)

    GPIO.output(TRIG2, False)

    #print ("Waiting For Sensor 2To Settle")

    time.sleep(1)

    GPIO.output(TRIG2, True)

    time.sleep(0.00001)

    GPIO.output(TRIG2, False)

    while GPIO.input(ECHO2)==0:

      pulse_start2 = time.time()

    while GPIO.input(ECHO2)==1:

      pulse_end2 = time.time()
      
    pulse_duration2 = pulse_end2 - pulse_start2

    distance2 = pulse_duration2 * 17150

    distance2 = round(distance2, 2)

    #print ("Distance2:",distance2,"cm")

    GPIO.cleanup()
    return distance2

def sonic_three():
    
    GPIO.setmode(GPIO.BCM)

    TRIG3= 25

    ECHO3 = 9

    #print ("Distance 3 Measurement In Progress")

    GPIO.setup(TRIG3,GPIO.OUT)

    GPIO.setup(ECHO3,GPIO.IN)

    GPIO.output(TRIG3, False)

    #print ("Waiting For Sensor 3 To Settle")

    time.sleep(2)

    GPIO.output(TRIG3, True)

    time.sleep(0.00001)

    GPIO.output(TRIG3, False)

    while GPIO.input(ECHO3)==0:

      pulse_start3 = time.time()

    while GPIO.input(ECHO3)==1:

      pulse_end3 = time.time()
      
    pulse_duration3 = pulse_end3 - pulse_start3

    distance3 = pulse_duration3 * 17150

    distance3 = round(distance3, 2)

    #print ("Distance 3 :",distance3,"cm")

    GPIO.cleanup()
    return distance3

def classify_frame(net, inputQueue, outputQueue):
    # keep looping
    while True:
        # check to see if there is a frame in our input queue
        if not inputQueue.empty():
            # grab the frame from the input queue, resize it, and
            # construct a blob from it
            frame = inputQueue.get()
            frame = cv2.resize(frame, (900, 900))
            blob = cv2.dnn.blobFromImage(frame, 0.007843,
                (300, 300), 127.5)

            # set the blob as input to our deep learning object
            # detector and obtain the detections
            net.setInput(blob)
            detections = net.forward()

            # write the detections to the output queue
            outputQueue.put(detections)

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
    help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
    help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
    help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# initialize the input queue (frames), output queue (detections),
# and the list of actual detections returned by the child process
inputQueue = Queue(maxsize=1)
outputQueue = Queue(maxsize=1)
detections = None

# construct a child process *indepedent* from our main process of
# execution
print("[INFO] starting process...")
p = Process(target=classify_frame, args=(net, inputQueue,
    outputQueue,))
p.daemon = True
p.start()

# initialize the video stream, allow the cammera sensor to warmup,
# and initialize the FPS counter
print("[INFO] starting video stream...")
#vs = VideoStream(src=0).start()
vs = VideoStream(usePiCamera=True).start()
time.sleep(2.0)
fps = FPS().start()

# loop over the frames from the video stream
while True:
    
    d1 = sonic_one()
    d2 = sonic_two()
    d3= sonic_three()
    m = min(d1,d2,d3)
    # grab the frame from the threaded video stream, resize it, and
    # grab its imensions
    frame = vs.read()
    frame = imutils.resize(frame, width=400)
    (fH, fW) = frame.shape[:2]

    # if the input queue *is* empty, give the current frame to
    # classify
    if inputQueue.empty():
        inputQueue.put(frame)

    # if the output queue *is not* empty, grab the detections
    if not outputQueue.empty():
        detections = outputQueue.get()

    # check to see if our detectios are not None (and if so, we'll
    # draw the detections on the frame)
    text = ""
    if m < 50:
        if m==d1:
            text = " Object in the front. Please wait. "
            print("front")
    
        if m==d2:
            text = "object in the right. Please be careful. "
            print("right")
            
        if m==d3:
            text="object in the left. Please be careful."
            print("left")
            
        myobj = gTTS(text=text,lang="en")
        myobj.save('welcome.mp3')
        os.system("omxplayer welcome.mp3")
          
    text= ""
    if detections is not None and m==d1:
        # loop over the detections
        print("trying to detecct")
        for i in np.arange(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated
            # with the prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the `confidence`
            # is greater than the minimum confidence
            if confidence < args["confidence"]:
                continue

            # otherwise, extract the index of the class label from
            # the `detections`, then compute the (x, y)-coordinates
            # of the bounding box for the object
            idx = int(detections[0, 0, i, 1])
            dims = np.array([fW, fH, fW, fH])
            box = detections[0, 0, i, 3:7] * dims
            (startX, startY, endX, endY) = box.astype("int")

            # draw the prediction on the frame
            label = "{}: {:.2f}%".format(CLASSES[idx],
                confidence * 100)
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
            print(CLASSES[idx])
            
            text="There is a "+ CLASSES[idx]+"ahead."
            myobj = gTTS(text=text,lang="en")
            myobj.save('welcome.mp3')
            os.system("omxplayer welcome.mp3")

    # show the output frame
    
    #cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

    # update the FPS counter
    fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()