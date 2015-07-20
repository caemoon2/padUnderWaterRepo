__author__ = 'moon'

import numpy as np
import cv2
import time
import heapq
from matplotlib import pyplot as plt
import math

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
overlap=0
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

    cv2.imshow('mask',res)
    overlap=overlap+mask

    end= time.time()
    if end - start > 3:
        break

    frameNum= frameNum + 1

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
    cv2.imshow('overlap',overlap)

xs = np.arange(0,650,1)
bottomxright=0
topxright=0
bottomxleft=500
topxleft=1000

for i in range(1,4*frameNum/4):
    x= points[i][:,0]
    y= points[i][:,1]
    xval1=np.max(y)
    xval2=np.min(y)

    z= np.polyfit(x,y,3)

    polynomial= np.poly1d(z)
    ys= polynomial(xs)
    polyderi=np.polyder(polynomial)

    xmaxs=heapq.nlargest(3,x)
    xmins=heapq.nsmallest(3,x)
#    print xmaxs[0]
    if bottomxright<xval1:
        bottomxright=xval1
        topxright=min(y)
        right=i

    if bottomxleft>xval2:
        bottomxleft=xval2
        topxleft=max(y)
        left=i
    plt.figure(0)
    plt.plot(y,-x,'o')
    plt.plot(ys,-xs,'r')
    plt.ylim((-300,0))
    plt.xlim(((-150,750)))

    ytop=polynomial(np.average(xmins))
    ybtm=polynomial(np.average(xmaxs))
    slope1=0
    slope2=0
    for k in range(0,3):
        slope11=math.degrees(math.atan(-1/polyderi(xmins[k])))
        # print 'slope11',slope11
        slope22=math.degrees(math.atan(-1/polyderi(xmaxs[k])))
        slope1=slope1+slope11
        slope2=slope2+slope22
    slope1=slope1/3
    slope2=slope2/3
    slope3=math.degrees(math.atan(-(np.average(xmins)-np.average(xmaxs))/(ytop-ybtm)))

    # xmin=min(x)
    # xmax=max(x)

    # ytop=polynomial(xmin)
    # ybtm=polynomial(xmax)
    #
    # slope111=math.degrees(math.atan(-1/polyderi(xmin)))
    # print 'slope111',slope111
    # slope2=math.degrees(math.atan(-1/polyderi(xmax)))
    # slope3=math.degrees(math.atan(-(xmin-xmax)/(ytop-ybtm)))
    #difference=np.abs(slope1-slope2)
    difference=slope1-slope2
    if difference<-100 :
        difference=180+difference
    elif difference>100 :
        difference=180-difference
    plt.figure(4)
    plt.title('angle difference')
    plt.plot(i,difference,'o')
    plt.figure(1)
    plt.title('start slope')
    plt.plot(i,slope1,'o')
    plt.figure(3)
    plt.title('average slope')
    plt.plot(i,slope3,'o')
    plt.figure(2)
    plt.title('final slope')
    plt.plot(i,slope2,'o')

print topxright, bottomxright
print topxleft,bottomxleft
plt.show()
cv2.waitKey()
cv2.destroyAllWindows()