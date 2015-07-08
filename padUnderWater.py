__author__ = 'moon'

import numpy as np
import cv2



##------------------------
## simple
#flags=[i for i in dir(cv2) if i.startswith('COLOR_')]
#print flags

##----------------------
## object tracking
cap = cv2.VideoCapture('../Images/light1.MOV')
result=open('mask.txt','w')

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
#    kernel = np.ones((19,19),np.uint8)
#    gradient = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel)
#    closing = cv2.morphologyEx(gradient, cv2.MORPH_CLOSE, kernel)
#    kernel = np.ones((25,25),np.uint8)
#    erosion = cv2.erode(closing,kernel,iterations = 1)
    # Bitwise-AND mask and original image
    #print mask[600:601,300:301]
    res = cv2.bitwise_and(frame,frame, mask= mask)
    cv2.imshow('mask',mask)
 #list

#    contours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
   # Draw contours
#    cv2.drawContours(frame, contours, -1, (0,255,0), 3)


    #print contours[0]
   # result.write('\n')
    #result.write('\n')
   # result.write(mask)
 ##   cv2.imshow('frame',frame)
   # cv2.imshow('mask',mask)
    #cv2.imshow('res',res)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
##how to find hsv values to track?
result.close()

green=np.uint8([[[0,0,128]]])
hsv_green=cv2.cvtColor(green,cv2.COLOR_BGR2HSV)


print hsv_green

cv2.destroyAllWindows()
