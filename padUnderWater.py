__author__ = 'moon'

import numpy as np
import cv2
import time

##------------------------
## simple
#flags=[i for i in dir(cv2) if i.startswith('COLOR_')]
#print flags

##----------------------
## object tracking
cap = cv2.VideoCapture('../Images/light1.MOV')
result=open('mask.txt','w')

frameNum= 0
points= []
start= time.time()
while(1):

    # Take each frame
    _, frame = cap.read()
    frame= frame[450:1000, 300:1600]
   # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define range of blue color in HSV
    lower_blue = np.array([80,50,50])
    upper_blue = np.array([130,255,255])


    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_blue, upper_blue)


    res = cv2.bitwise_and(frame,frame, mask= mask)

    imgArray= np.array(res)
    points.append(np.transpose(np.nonzero(imgArray)))


    # result.write('\n')
    # result.write('\n')
    # result.write(mask)


    cv2.imshow('mask',mask)

    end= time.time()
    if end - start > 2:
        break

    frameNum= frameNum + 1
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

##how to find hsv values to track?
result.close()

print points[0].shape


cv2.destroyAllWindows()
