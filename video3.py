import cv2 
import numpy as np 
def nothing (x):
	pass

cv2.namedWindow('image')

cap = cv2.VideoCapture(0)
ret=cap.set(3, 320)
ret=cap.set(4, 240)
cv2.createTrackbar('canny1','image',456,1000,nothing)
cv2.createTrackbar('canny2','image',24,1000,nothing)
#cv2.createTrackbar('B','image',2,255,nothing)
#cv2.createTrackbar('T','image',10,100,nothing)
#cv2.createTrackbar('U','image',10,100,nothing)
while(1):

    r = cv2.getTrackbarPos('canny1','image')
    g = cv2.getTrackbarPos('canny2','image')
#    b = cv2.getTrackbarPos('B','image')
#    t = cv2.getTrackbarPos('T','image')
#    u = cv2.getTrackbarPos('U','image')
	
    # Take each frame
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(5,5),0)
    equ = cv2.equalizeHist(gray)
    edges = cv2.Canny(equ,r,g)

#    lines = cv2.HoughLinesP(edges,1,np.pi/180,b, minLineLength = t, maxLineGap = u)
#    print "o iteratie "
#    try:
#        for x1,y1,x2,y2 in lines[0]:
#            cv2.line(frame,(x1,y1),(x2,y2),(0,255,0),2)

#    except:
#        print "eroare"
    cv2.imshow('frame',frame)
    cv2.imshow('mask',equ)
    cv2.imshow('res',edges)
    k = cv2.waitKey(5)
    if k == ord('q'):
        break

cv2.destroyAllWindows()

