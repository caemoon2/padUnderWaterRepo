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
#    cv2.imshow('overlap',overlap)

xs = np.arange(0,650,1)
bottomxright=0
topxright=0
bottomxleft=500
topxleft=1000
delxright=0
delxleft=0
for i in range(frameNum/2,frameNum):
    x= points[i][:,0]
    y= points[i][:,1]
    xval1=np.max(y)
    xval2=np.min(y)

    z= np.polyfit(x,y,3)

    polynomial= np.poly1d(z)

    ys= polynomial(xs)

    if bottomxright<xval1:
        bottomxright=xval1
        topxright=min(y)
        right=i

    if bottomxleft>xval2:
        bottomxleft=xval2
        topxleft=max(y)
        left=i
    if (i%2)==1:
        plt.figure(0)
        plt.plot(y,-x,'o')
        plt.plot(ys,-xs,'r')
        plt.ylim((-300,0))
        plt.xlim(((-150,750)))

    delright=bottomxright-topxright
    if topxright<bottomxright:
        if delright>delxright:
            delxleft=delright
            ctright=topxright
            cbright=bottomxright
            curvright=i
    print 'ctright',curvright

    delleft=topxleft-bottomxleft
    curvleft=0
    if topxleft>bottomxleft:
        if delleft>delxleft:
            ctleft=topxleft
            cbleft=bottomxleft
            curvleft=i
    print 'ctleft',curvleft
print ctright,cbright,curvright,ctleft,cbleft,curvleft
for i in range(frameNum/2,frameNum):
    x= points[i][:,0]
    y= points[i][:,1]

    if i==right or i==left:
        plt.figure(1)
        plt.plot(y,-x,'k')
        plt.ylim((-300,0))
        plt.xlim(((-150,750)))
        plt.text(topxright+20, -50, topxright, fontsize=12)
        plt.text(bottomxright+20, -220, bottomxright, fontsize=12)
        plt.text(topxleft-60, -20, topxleft, fontsize=12)
        plt.text(bottomxleft-20, -180, bottomxleft, fontsize=12)
    if i==curvleft or i==curvright:
        plt.figure(2)
        plt.plot(y,-x,'k')
        plt.ylim((-300,0))
        plt.xlim(((-150,750)))
        plt.text(ctright+20, -40, ctright, fontsize=12)
        plt.text(cbright+20, -220, cbright, fontsize=12)
        plt.text(ctleft-20, -20, ctleft, fontsize=12)
        plt.text(cbleft-20, -200, cbleft, fontsize=12)
#plt.Figure.
print topxright, bottomxright
print topxleft,bottomxleft
plt.show()
cv2.waitKey()
cv2.destroyAllWindows()