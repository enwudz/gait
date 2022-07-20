#!/usr/bin/python
import cv2
import sys
import os
import glob
from gait_analysis import *

'''
Measure things!
    distance of tardigrade travel
    width of tardigrade
    length of tardigrade
    size of field of view
    (and calculate speed from distance and time)

distance is measured in pixels - need to have a reference to convert to real distance
    e.g. a micrometer at same magnification

Superimpose first and last frame
Prompt - measure (w)idth, (l)ength, (d)istance, (f)ield of view, (q)uit
Assign zeros to all before measuring
Open window, measure (c) to close and record, (r) to reset and try again
If multiple measurements of same thing, take average
Upon quit, calculate speed, write all data to mov_data.txt file
'''

def main(data_folder):

    # get data ... which folder should we look in?
    # run this script in a directory that has directories containing data for clips
    if len(data_folder) == 0: 
        dirs = listDirectories()
        data_folder = selectOneFromList(dirs)
    mov_data = os.path.join(data_folder, 'mov_data.txt')
    fileTest(mov_data)

    # parse movie data to get info about movie 
    # (movie name, analyzed length, frame range for speed calculation, movie length)
    movie_info = getMovieInfo(data_folder)

    # if measurements already exist, prompt to quit
    if movie_info['tardigrade_width'] > 0:
        should_i_quit = input('\n It looks like there are already measurements .. (p)roceed or (q)uit?: ')
        if should_i_quit == 'q':
            exit()
        elif should_i_quit == 'p':
            print('I need to think about whether to keep old measurements or discard . . .')

    # if we have information about what frames to use to calculate speed
    # ... save these two frames if we do not already have them
    saveSpeedFrames(data_folder, movie_info)

    # Prompt - measure (w)idth, (l)ength, (d)istance, (f)ield of view, (q)uit
    # Assign zeros to all of these before measuring
    tardigrade_width = []
    tardigrade_length = []
    distance_traveled = []
    field_of_view = []
    tardigrade_speed = 0
    measuring = True

    # load beginning frame and ending frame
    beginning = loadImage(data_folder, 'beginning_speed_frame.png')
    ending = loadImage(data_folder, 'ending_speed_frame.png')

    # superimpose the two images, and clone
    image = cv2.addWeighted(beginning,0.5,ending,0.5,0)
    clone = image.copy()

    # what should we measure first?
    what_to_measure = promptForMeasurement()
    if what_to_measure == 'q':
        measuring = False

    # measure the things! put this in a function that we can call . . . 
    while measuring is True:
        print('I will measure something!')
        
        # call measuring function here

        image_window_text = "Drag a line to measure " + what_to_measure + " ; then (d)one or (r) reset"
        cv2.namedWindow(image_window_text)
        cv2.setMouseCallback(image_window_text, clickDrag)
        cv2.imshow(image_window_text, image)
        key = cv2.waitKey(1)

        # if the 'r' key is pressed, reset the measurement line
        if key == ord("r"):
            image = clone.copy()

        # if the 'd' or 'q' key is pressed, break from the loop
        elif key == ord("d") or key == ord("q"):
            cv2.destroyAllWindows()

        # keep on measuring?
        what_to_measure = promptForMeasurement()
        if what_to_measure == 'q':
            measuring = False

    cv2.destroyAllWindows()


    # update movie_info with averages of measurements!

    # update mov_data.txt
    updateMovieData(data_folder, movie_info)

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

def loadImage(data_folder, image_filename):
    image_file = os.path.join(data_folder, image_filename)
    image = cv2.imread(image_file)
    return image

def promptForMeasurement():
    selection = input('\nMeasure (w)idth, (l)ength, (d)istance, (f)ield of view, (q)uit?: ' ).rstrip()
    print('you pressed ' + selection)
    if selection == 'w':
        what_to_measure = 'width'
    elif selection == 'l':
        what_to_measure = 'length'
    elif selection == 'd':
        what_to_measure = 'distance'
    elif selection == 'f':
        what_to_measure = 'field_of_view'
    else:
        what_to_measure = 'q'
    print(what_to_measure)
    return what_to_measure

if __name__== "__main__":
    if len(sys.argv) > 1:
            data_folder = sys.argv[1]
            print('looking in ' + data_folder)
    else:
        data_folder = ''

    main(data_folder)