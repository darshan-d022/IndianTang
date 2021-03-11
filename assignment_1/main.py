import numpy as np
import imutils
import cv2, datetime

p =0

class person:
 def __init__(self):
  self.p =0
 
 def crop(self):
  two = cv2.imread('original1.jpg')
  (h,w) = two.shape[:2]
  print(h,w)
  starty =0
  endy =int(h/2)
  for i in range(0,2):
   startx =0
   endx =int(w/4)
   for j in range(0,4):
    print(starty,endy,startx,endx)
    init_crop = two[starty:endy,startx:endx]
    init_crop = cv2.resize(init_crop,(int(w/4),int(h/2)))
    frame =self.run(init_crop)
    cv2.imwrite('crop{}.{}.jpg'.format(i,j),frame)
    startx += int(w/4)
    endx += int(w/4)
    
   starty += int(h/2)
   endy += int(h/2)
  
 def run(self,frame):

  # construct the argument parse and parse the arguments
  proto ='mobilenet_ssd\MobileNetSSD_deploy.prototxt'
  model = 'mobilenet_ssd/MobileNetSSD_deploy.caffemodel'
  # confidence default 0.4
  conf =0.29

  # initialize the list of class labels MobileNet SSD was trained to detect
  CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
   "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
   "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
   "sofa", "train", "tvmonitor"]

  # load our serialized model from disk
  net = cv2.dnn.readNetFromCaffe(proto, model)
  # frame = imutils.resize(frame, width = 00)
  rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  W =None
  H=None
  # If the frame dimensions are empty, set them
  if W is None or H is None:
   (H, W) = frame.shape[:2]
  rects = []
  # convert the frame to a blob and pass the blob through the network and obtain the detections
  blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
  net.setInput(blob)
  detections = net.forward()
  count = 0
  # loop over the detections
  for i in np.arange(0, detections.shape[2]):
   # extract the confidence (i.e., probability) associated with the prediction
   confidence = detections[0, 0, i, 2]
   # filter out weak detections by requiring a minimum confidence
   if confidence > conf:
    idx = int(detections[0, 0, i, 1])
    if CLASSES[idx] != "person":
     continue
    count=count+1
    self.p +=1
    print(self.p)
    box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
    (startX, startY, endX, endY) = box.astype("int")
    image = cv2.putText(frame, 'person{} :{:.2f}'.format(self.p,confidence), (startX-20,startY-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255),2, cv2.LINE_AA) 
    cv2.rectangle(frame,(startX,startY),(endX,endY),(0, 0, 255),2)


  cv2.imshow("Real-Time Monitoring/Analysis Window", frame)
  cv2.waitKey(0)
   # close any open windows
  cv2.destroyAllWindows()
  return frame

#funtion to join the images back
 def join(self):
  for i in range(0,2):
   frame1 = cv2.imread('crop{}.0.jpg'.format(i))
   for j in range(0,3):
    frame2 =cv2.imread('crop{}.{}.jpg'.format(i,j+1))
    print(frame1.shape,frame2.shape)
    frame1 = cv2.hconcat([frame1,frame2])
   cv2.imwrite('crop_{}.jpg'.format(i+2),frame1)
  
  final = cv2.imread('crop_1.jpg')
  for i in range(2,4):
   j = cv2.imread('crop_{}.jpg'.format(i))
   print(final.shape,j.shape)
   final = cv2.vconcat([final,j])
  cv2.imwrite('final.jpg',final)
    
people = person()
people.crop()
people.join()
