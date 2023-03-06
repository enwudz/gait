#!/usr/bin/python

import cv2
import sys
import glob
import math
import numpy as np

'''
modified from measure_things.py (in tardigrade gait)
Masure a single image and save image with labeled measurement on it
'''

def main(image_file, saveImage = 'True'):

    # grab references to the global variables
    global D, image, refPt, drawing, nameText

    # Prompt - start measuring?
    measuring = True

    # load image to measure
    img = cv2.imread(image_file)
    clone = img.copy()

    # let's measure things!
    measuring = True
    while measuring is True:

        # measure the things!
        image = clone.copy()
        D, measure_key = measureImage()

        if measure_key == ord('q') or measure_key == ord('d'):
            measuring = False
            cv2.destroyAllWindows()
            break

    D = np.round(D,decimals=2)

    
    if saveImage:
        
        # save the annotated image
        stuff = image_file.split('.')
        annotated_file = stuff[0] + '_measured.' + stuff[1]
        cv2.imwrite(annotated_file, image)
        
        # save a scale file 
        scale_file = image_file.split('.')[0] + '_scale.txt'
        o = open(scale_file,'w')
        o.write('1 mm='+str(D))
        o.close()
    
    return D


def measureImage():

    # grab references to the global variables
    global D, image, refPt, drawing, nameText
    refPt = []
    
    nameText = "Drag a line to measure, then (d) done or (r) reset"
    clone = image.copy()
    height, width = image.shape[:2]

    # keep looping until the 'd' (or 'q') key is pressed
    while True:
        # display the image and wait for a keypress
        cv2.namedWindow(nameText, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(nameText, width, height)
        cv2.setMouseCallback(nameText, clickDrag)
        cv2.imshow(nameText, image)
        key = cv2.waitKey(1) & 0xFF

        # if the 'r' key is pressed, reset the measurement line
        if key == ord("r"):
            image = clone.copy()

        # if the 'd' or 'q' key is pressed, break from the loop
        elif key == ord("d") or key == ord("q"):
            # close all open windows
            cv2.destroyAllWindows()
            break

    # close all open windows
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    return D, key

def clickDrag(event, x, y, flats, param):
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
		D = math.dist(refPt[0], refPt[1])
		mX = refPt[0][0]
		mY = refPt[0][1]
		cv2.putText(image, "{:.1f} pix".format(D), (int(mX), int(mY - 10)),
			cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,0,0), 2)
		cv2.imshow(nameText, image)

if __name__== "__main__":
    if len(sys.argv) > 1:
            image_file = sys.argv[1]
            print('looking for ' + image_file)
    else:
        image_file = glob.glob('*.png')[0]

    main(image_file)
