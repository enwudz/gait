#!/usr/bin/python
import cv2
import sys
import os
import math
from gait_analysis import *

'''
Measure things!
    distance of tardigrade travel
    width of tardigrade
    length of tardigrade
    size of field of view (consider showing on the image where the horizontal halfway line is?)
    (and calculate speed from distance and time)

distance is measured in pixels - need to have a reference to convert to real distance
    e.g. a micrometer at same magnification
'''

def main(data_folder):

    # grab references to the global variables
    global D, image, refPt, drawing, nameText

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
    what_to_measure = ''

    # let's measure things!
    measuring = True
    while measuring is True:

        if what_to_measure == 'q':
            measuring = False
            cv2.destroyAllWindows()
            break
        
        # measure the things!
        image = clone.copy()
        D, what_to_measure = measureImage()
        print(what_to_measure + ' = ' + str(D))

        # add measurement to appropriate list (only need 1, but can take averages if multiple)
        if what_to_measure == 'width':
            tardigrade_width.append(D)
        elif what_to_measure == 'length':
            tardigrade_length.append(D)
        elif what_to_measure == 'distance':
            distance_traveled.append(D)
        elif what_to_measure == 'field_of_view':
            field_of_view.append(D)
    
    # all done measuring! calculate measurement averages and add to movie_info
    movie_info['tardigrade_width'] = np.around(np.mean(tardigrade_width))
    movie_info['tardigrade_length'] = np.around(np.mean(tardigrade_length))
    movie_info['field_width'] = np.around(np.mean(field_of_view))
    movie_info['distance_traveled'] = np.around(np.mean(distance_traveled))

    # calculate speed!
    elapsed_time = movie_info['speed_end'] - movie_info['speed_start']
    tardigrade_speed = round(movie_info['distance_traveled'] / elapsed_time, 2)
    print('Tardigrade speed is ' + str(tardigrade_speed) + ' pixels/second')
    movie_info['tardigrade_speed'] = tardigrade_speed

    # update mov_data.txt
    updateMovieData(data_folder, movie_info)

def measureImage():

    # grab references to the global variables
    global D, image, refPt, drawing, nameText
    refPt = []

    # what should we measure?
    what_to_measure = promptForMeasurement()
    if what_to_measure == 'q':
        return (0, what_to_measure) 

    nameText = "Drag a line to measure " + what_to_measure + " ; then (d)one or (r) reset"
    clone = image.copy()

    # keep looping until the 'd' (or 'q') key is pressed
    while True:
        # display the image and wait for a keypress
        cv2.namedWindow(nameText)
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
    return round(D,2), what_to_measure

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

def loadImage(data_folder, image_filename):
    image_file = os.path.join(data_folder, image_filename)
    image = cv2.imread(image_file)
    return image

def promptForMeasurement():
    selection = input('\nMeasure (w)idth, (l)ength, (d)istance, (f)ield of view, (q)uit?: ' ).rstrip()
    # print('you pressed ' + selection) # testing
    if selection == 'w':
        what_to_measure = 'width'
        print('  You selected width - measure across body between leg pairs 2 and 3')
    elif selection == 'l':
        what_to_measure = 'length'
        print('  You selected length - measure from nose to space between 4th leg pair')
    elif selection == 'd':
        what_to_measure = 'distance'
        print('  You selected distance - measure nose to nose')
    elif selection == 'f':
        what_to_measure = 'field_of_view'
        print('  You selected field of view - measure across diameter')
    else:
        what_to_measure = 'q'
        print('All done measuring!')
    # print(what_to_measure) # testing
    return what_to_measure

if __name__== "__main__":
    if len(sys.argv) > 1:
            data_folder = sys.argv[1]
            print('looking in ' + data_folder)
    else:
        data_folder = ''

    main(data_folder)