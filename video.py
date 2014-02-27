import numpy as np
import cv2, random, time
t0=0
font=cv2.FONT_HERSHEY_SIMPLEX
cap = cv2.VideoCapture(0)
ret = cap.set(3,320)
ret = cap.set(4,240)
def draw_circle(event,x,y,flags,param):
    global bmp	
    if event == cv2.EVENT_MOUSEMOVE:
        print "mouse event "+"x= "+str(x)+"y= "+str(y)
	cv2.putText(bmp, str(vit), (10, 50), font, 1, (255, 255, 255), 2)
        cv2.circle(bmp,(x,y),10,(255,0,0),-1)
#bmp = np.zeros((512,512,3), np.uint8)
cv2.namedWindow('frame')

while(True):
    # Capture frame-by-frame
#    ret = cap.set(3,320)
 #   ret = cap.set(4,240)	
    ret, frame = cap.read()
    t1=time.time()
    vit=round(1/(t1-t0), 2)
    t0=t1
    # Our operations on the frame come here
#    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#    ret, bmp=cv2.treshhold(gray, 127, 255, cv2.TRESH_TOZERO)
#    bmp=cv2.flip(frame,0)	
#    bmp=cv2.flip(bmp,0)
    bmp=(255-frame) 
#    bmp2=cv2.line(bmp,(random.randrange(cap.get(3)), random.randrange(cap.get(4))),(random.randrange(cap.get(3)), random.randrange(cap.get(4))),(0,255,0),2)
    # Display the resulting frame
#    cv2.setMouseCallback('frame',draw_circle,0)
    frame[120,160]=[0,255,0]
    y=random.randrange(cap.get(4)-20)
    x=random.randrange(cap.get(3)-50)
    frame[50:70, 100:150]=frame[y:y+20, x:x+50]
    cv2.rectangle(frame, (x,y), (x+50, y+20), (50,50,50), 1)
#    frame2=cv2.split(frame)[0]
#    bmp=frame[0]-frame[1]
    cv2.putText(bmp, str(vit), (10, 50), font, 1, (90,90,90), 2)
    dst = cv2.addWeighted(bmp, 0.3,frame, 0.7, 0)
    cv2.imshow('frame',(255-dst))
    if cv2.waitKey(1) == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
