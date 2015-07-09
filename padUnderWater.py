__author__ = 'moon'

import numpy as np
import cv2
import time
from matplotlib import pyplot as plt

##------------------------
## simple
#flags=[i for i in dir(cv2) if i.startswith('COLOR_')]
#print flags

##----------------------
## object tracking
cap = cv2.VideoCapture('../Images/light1.MOV')

frameNum= 0
points= []
start= time.time()
while(1):

    # Take each frame
    _, frame = cap.read()
    frame= frame[450:1000, 300:1600]
    frame= cv2.pyrDown(frame)
    # frame= cv2.pyrDown(frame)
    # frame= cv2.pyrDown(frame)

   # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define range of blue color in HSV
    lower_blue = np.array([80,50,50])
    upper_blue = np.array([130,255,255])


    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_blue, upper_blue)


    res = cv2.bitwise_and(frame,frame, mask= mask)

    imgArray= np.array(mask)
    points.append(np.transpose(np.nonzero(imgArray)))

    cv2.imshow('mask',mask)

    end= time.time()
    if end - start > 3.5:
        break

    frameNum= frameNum + 1
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()


xs = np.arange(0,650,1)
for i in range(frameNum/2,frameNum):
    x= points[i][:,0]
    y= points[i][:,1]

    z= np.polyfit(x,y,3)
    polynomial= np.poly1d(z)

    ys= polynomial(xs)

    plt.plot(xs,ys,'r')




plt.show()
