import cv2
import numpy as np

def nothing(x):
    pass
cv2.namedWindow('image')
cap = cv2.VideoCapture(0)
ret=cap.set(3, 320)
ret=cap.set(4, 240)
lines=[[[1,2,100,200]]]
cv2.createTrackbar('R','image',427,1000,nothing)
cv2.createTrackbar('G','image',65,1000,nothing)
cv2.createTrackbar('B','image',20,50,nothing)
cv2.createTrackbar('T','image',17,50,nothing)
cv2.createTrackbar('U','image',2,10,nothing)
while(1):
    minLineLength = 100
    maxLineGap = 10

    r = cv2.getTrackbarPos('R','image')
    g = cv2.getTrackbarPos('G','image')
    b = cv2.getTrackbarPos('B','image')
    t = cv2.getTrackbarPos('T','image')
    u = cv2.getTrackbarPos('U','image')
	
    # Take each frame
    _, frame = cap.read()
#    frame = cv2.imread('gradient2.jpg')
    # Convert BGR to HSV
#    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#    th3 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    gray = cv2.GaussianBlur(gray,(3,3),0)
    equ = cv2.equalizeHist(gray)

    edges = cv2.Canny(equ,r,g)
    #edges=255-th3
    # define range of blue color in HSV
#    lower_blue = np.array([110,30,30])
#    upper_blue = np.array([130,255,255])

    # Threshold the HSV image to get only blue colors
#    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Bitwise-AND mask and original image
#    res = cv2.bitwise_and(frame,(255-frame), mask= mask)
    lines = cv2.HoughLinesP(edges,1,np.pi/180,b, minLineLength = t, maxLineGap = u)
    print "o iteratie "
#    if lines[0]:	
    try:
        for x1,y1,x2,y2 in lines[0]:
            cv2.line(frame,(x1,y1),(x2,y2),(0,255,0),2)
#    cv2.imshow('houghlines',hsv)

    except:
        print "eroare"
    cv2.imshow('frame',edges)
    cv2.imshow('mask',frame)
    #cv2.imshow('res',th3)
    k = cv2.waitKey(5)
    if k == ord('q'):
        break

cv2.destroyAllWindows()

