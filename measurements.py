#!/usr/bin/python

'''
open picture(s) (a specified picture, or all pictures)
click down to start measurement, release to end
measure distance between these points
show picture with distance annotated on it
y to accept, r retry
'''

# to resize via sips:


# import the necessary packages
print('importing things')
import argparse
import cv2
from scipy.spatial import distance as dist
import glob
import numpy as np
print('done importing')

def main():
	fileList = getImageFileList()
	for imName in fileList:
		D = measurePicture(imName)
        print(imName + ',' + str(D))

def measurePicture(imName):
	# grab references to the global variables
	global D, image, refPt, drawing, nameText # not sure these need to be global

	refPt = []
	cropping = False

	# load the image, clone it, and setup the mouse callback function
	image = cv2.imread(imName)

	# resize image
	imWidth = np.shape(image)[1]
	maxWidth = 1024 # in pixels
	resizeMultiplier = float(maxWidth)/imWidth # or just 0.25
	image = cv2.resize(image, (0,0), fx=resizeMultiplier, fy=resizeMultiplier)

	nameText = "Record points: 'y' when done; 'r' to start over"
	clone = image.copy()
	cv2.namedWindow(nameText)
	cv2.setMouseCallback(nameText, clickDrag)

	# keep looping until the 'y' key is pressed
	while True:
		# display the image and wait for a keypress
		cv2.imshow(nameText, image)
		key = cv2.waitKey(1) & 0xFF

		# if the 'r' key is pressed, reset the line
		if key == ord("r"):
			image = clone.copy()

		# if the 'y' or 'q' key is pressed, break from the loop
		elif key == ord("y") or key == ord("q"):
			break

	# close all open windows
	cv2.destroyAllWindows()
	return D

def clickDrag(event, x, y, flags, param):
	# grab references to the global variables
	global D, image, refPt, drawing, nameText

	# if the left mouse button was clicked, record the starting
	# (x, y) coordinates and draw line
	if event == cv2.EVENT_LBUTTONDOWN:
		refPt = [(x, y)]
		drawing = True

	# check to see if the left mouse button was released
	elif event == cv2.EVENT_LBUTTONUP:
		# record the ending (x, y) coordinates and indicate that
		# the cropping operation is finished
		refPt.append((x, y))
		drawing = False

		# draw a line connecting the points
		cv2.line(image, refPt[0], refPt[1], (0, 255, 0), 2)
		D = dist.euclidean(refPt[0], refPt[1])
		mX = refPt[0][0]
		mY = refPt[0][1]
		cv2.putText(image, "{:.1f} pix".format(D), (int(mX), int(mY - 10)),
			cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,0,0), 2)
		cv2.imshow(nameText, image)

def getImageFileList():
	fileList = glob.glob('*.[jpP][pnN][2gG]')
	#fileList = glob.glob('*.jpeg')
	#fileList = glob.glob('*.png')
	#fileList = glob.glob('*.bmp')
	return fileList

if __name__ == '__main__':
	main()
