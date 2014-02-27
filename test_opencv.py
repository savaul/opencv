import cv,cv2 
import numpy as np
from matplotlib import pyplot as plt
import time

cv.NamedWindow("camera", 1)
capture = cv.CreateCameraCapture(0)

width = None #leave None for auto-detection
height = None #leave None for auto-detection

if width is None:
    width = int(cv.GetCaptureProperty(capture, cv.CV_CAP_PROP_FRAME_WIDTH))
else:
	cv.SetCaptureProperty(capture,cv.CV_CAP_PROP_FRAME_WIDTH,width)    

if height is None:
	height = int(cv.GetCaptureProperty(capture, cv.CV_CAP_PROP_FRAME_HEIGHT))
else:
	cv.SetCaptureProperty(capture,cv.CV_CAP_PROP_FRAME_HEIGHT,height) 

while True:
        img = cv.QueryFrame(capture)
    	im_gray  = cv.CreateImage(cv.GetSize(img),cv.IPL_DEPTH_8U,1)
    	cv.CvtColor(img,im_gray,cv.CV_RGB2GRAY)
#	img=cv2.imread(img, 0)
#    cv.ShowImage("camera", img)
	#ret, pict = cv2.threshold(im_gray,127,255,cv2.THRESH_BINARY)
	#ret,thresh2 = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
	#ret,thresh3 = cv2.threshold(img,127,255,cv2.THRESH_TRUNC)
	#ret,thresh4 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO)
	ret,thresh5 = cv2.adaptiveThreshold(im_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)

	#titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
	#images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]

	#for i in xrange(6):
	#    plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
	#    plt.title(titles[i])
	#    plt.xticks([]),plt.yticks([])
	cv.ShowImage("camera", im_gray)
	#plt.show()
	k = cv.WaitKey(10);
        if k == 'f':
            break
