#PARTICPANTS
#ARAVIND REGONDA
#SWAPANDEEP SANDHU
#JAINIL DESAI

# import packages
from collections import deque
#import numpy
import argparse
import imutils
import cv2

# create arguements and parse them
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video")
ap.add_argument("-b", "--buffer", type=int, default=24)
args = vars(ap.parse_args())

# Upper and lower limit
greenLower = (29, 86, 6)
greenUpper = (64, 255, 255)
pts = deque(maxlen=args["buffer"])

# Start webcam
camera = cv2.VideoCapture(0)

while True:
	# grab frame
	(grabbed, frame) = camera.read()

	# if no frame grabbed, break loop
	if args.get("v") and not grabbed:
		break

	# resize, blur and convert to HSV
	frame = imutils.resize(frame, width=500)
	
	#blurred = cv2.GaussianBlur(frame, (11, 11), 0)
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

	# Generate mask for color and remove distractions
	mask = cv2.inRange(hsv, greenLower, greenUpper)
	mask = cv2.erode(mask, None, iterations=1)
	mask = cv2.dilate(mask, None, iterations=1)

	# Find contours and initialize center
	cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)[-2]
	center = None

	if len(cnts) > 0:
	
		c = max(cnts, key=cv2.contourArea)
		((x, y), radius) = cv2.minEnclosingCircle(c)
		M = cv2.moments(c)
		center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

		if radius > 2:
			# draw the circle and centroids
			cv2.circle(frame, (int(x), int(y)), int(radius),
				(0, 255, 255), 2)
			cv2.circle(frame, center, 5, (0, 0, 255), -1)
	# Updating track points
	pts.appendleft(center)

	# loop for all points
	for i in range(1, len(pts)):
		if pts[i - 1] is None or pts[i] is None:
			continue

		thickness = int(5)
		cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)

	# Show image
	frame2= cv2.flip(frame,1)
	cv2.imshow("Frame", frame)
	cv2.imshow("Flip", frame2)
	cv2.imshow("HSV", hsv)
	cv2.imshow("Mask", mask)
	
	key = cv2.waitKey(1) & 0xFF
	
	# if the 'q' key is pressed, stop the loop
	if key == ord("q"):
		break

# close camera
camera.release()
cv2.destroyAllWindows()
