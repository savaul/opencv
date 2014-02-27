import time
import cv2


webcam = cv2.VideoCapture(0)				# Get ready to start getting images from the webcam
webcam.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 320)		# I have found this to be about the highest-
webcam.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 240)	# 	resolution you'll want to attempt on the pi

frontalface = cv2.CascadeClassifier("finger.haarcascade.xml")		# frontal face pattern detection
#profileface = cv2.CascadeClassifier("haarcascade_profileface.xml")		# side face pattern detection

face = [0,0,0,0]	# This will hold the array that OpenCV returns when it finds a face: (makes a rectangle)
Cface = [0,0]		# Center of the face: a point calculated from the above variable
lastface = 0		# int 1-3 used to speed up detection. The script is looking for a right profile face,-


while True:

	faceFound = False	# This variable is set to true if, on THIS loop a face has already been found
				# We search for a face three diffrent ways, and if we have found one already-
				# there is no reason to keep looking.
	
	if not faceFound:
		if lastface == 0 or lastface == 1:
			aframe = webcam.read()[1]	# there seems to be an issue in OpenCV or V4L or my webcam-
			aframe = webcam.read()[1]	# 	driver, I'm not sure which, but if you wait too long,
			aframe = webcam.read()[1]	#	the webcam consistantly gets exactly five frames behind-
			aframe = webcam.read()[1]	#	realtime. So we just grab a frame five times to ensure-
			aframe = webcam.read()[1]
			cv2.imshow('frame',aframe)	#	we have the most up-to-date image.
			fface = frontalface.detectMultiScale(aframe,1.3,4,(cv2.cv.CV_HAAR_DO_CANNY_PRUNING + cv2.cv.CV_HAAR_FIND_BIGGEST_OBJECT + cv2.cv.CV_HAAR_DO_ROUGH_SEARCH),(10,40))
			if fface != ():			# if we found a frontal face...
				lastface = 1		# set lastface 1 (so next loop we will only look for a frontface)
				for f in fface:		# f in fface is an array with a rectangle representing a face
					faceFound = True
					face = f


	if not faceFound:		# if no face was found...-
		lastface = 0		# 	the next loop needs to know
		face = [0,0,0,0]	# so that it doesn't think the face is still where it was last loop
		print "no desht"

	x,y,w,h = face

	if face[0] != 0 or face[1] !=0 or face[2]!=0:		# if the Center of the face is not zero (meaning no face was found)
		print "desht found at " +str (face)
		cv2.rectangle(aframe, (x, y),((x+w), (y+h)), (0,255,0), 1)
		cv2.imshow('frame', aframe)
        k = cv2.waitKey(5)
        if k == ord('q'):
            break

cv2.destroyAllWindows()

