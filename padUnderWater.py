__author__ = 'guille'


import cv2
import numpy as np

cap= cv2.VideoCapture('../Images/light1.MOV')
# cap= cv2.VideoCapture(0)

# Small change to git Hub
g= 7

while(True):
    ret, frame= cap.read()
    # frame= cv2.bilateralFilter(frame,5,75,75)
    frame= frame[300:1000, 300:1600]
    if ret:
        gray= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # gray= cv2.bilateralFilter(gray,9,75,75)  # filters taking into account color
        # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.medianBlur(gray,9)
      # define range of blue color in HSV
      #   lower = np.array([40,100,100])
      #   upper = np.array([70,255,255])

    # Threshold the HSV image to get only blue colors
    #     imgray = cv2.inRange(hsv, lower, upper)
    #     ret2,thresh = cv2.threshold(gray,100,255,0)
        gray = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,19,7)  # threshold
        kernel= np.ones((21,21),np.uint8)
        gray= cv2.morphologyEx(gray,cv2.MORPH_OPEN, kernel)




        # cv2.imshow('image',gray)
        contours, hierarchy = cv2.findContours(gray,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

        length= len(contours)
        i= 0
        while i < length:
            arcLength= cv2.arcLength(contours[i],True)
            if arcLength > 3000 or arcLength < 500:
                del contours[i]
                i= i-1
                length= length - 1
            i= i+1

    # Draw contours
        cv2.drawContours(frame, contours, -1, (0,255,0), 3)
        cv2.imshow('contours', frame)


    else:
        print 'video not read properly'

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
