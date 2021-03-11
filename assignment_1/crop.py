import cv2
import numpy as np
import imutils

#reading the original image
frame = cv2.imread('original.jpg')
(H,W) = frame.shape[:2]
print(H,W)
init_crop = frame[1000:H,0:W]
print(init_crop.shape)

cv2.imwrite('original1.jpg',init_crop)

