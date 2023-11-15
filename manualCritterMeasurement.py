#!/usr/bin/python
import cv2
import sys
import math
import gaitFunctions

'''
Measure dimensions of a critter!
    width of critter
    length of tardigrade

distance is measured in pixels - need to have a reference to convert to real distance
    e.g. a micrometer at same magnification
'''

def main(movie_file):

    # grab references to the global variables
    global D, image, refPt, drawing, nameText

    # get an image to measure
    # get number of frames
    vid_width, vid_height, vid_fps, vid_frames, vid_length = gaitFunctions.getVideoData(movie_file)
    print('There are ' + str(vid_frames) + ' frames in ' + movie_file)
    
    # select a frame to measure ... first, last, or a specific frame
    selection_list = ['first frame','last frame', 'a specific frame number']
    print('\nWhich movie frame should we measure? ')
    selection = gaitFunctions.selectOneFromList(selection_list)

    if selection == 'first frame' or selection == 'last frame':
        first_frame, last_frame = gaitFunctions.getFirstLastFrames(movie_file)
        if selection == 'first frame':
            measured_frame = first_frame
        else: 
            measured_frame = last_frame
            
    else:
        # we want to choose a specific frame.
        frame_input = input('Enter a specific frame number to measure: ')
        try: 
            frame_number = int(frame_input)
        except:
            badInput(frame_input, movie_file)
        if frame_number < 1 or frame_number > vid_frames:
            badInput(frame_input, movie_file)
        else:
            # grab this specific frame from the movie
            print('We will get ' + str(frame_number) + ' !')
            cap = cv2.VideoCapture(movie_file)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number-1)
            ret, measured_frame = cap.read()

    # got the frame!! Ready to measure things
    
    measuring = True
    what_to_measure = ''
    width = 0
    length = 0

    # let's measure things!
    measuring = True
    while measuring is True:

        if what_to_measure == 'q':
            measuring = False
            cv2.destroyAllWindows()
            break
        
        # measure the things!
        image = measured_frame.copy()
        D, what_to_measure = measureImage()
        if what_to_measure != 'q':
            print(what_to_measure + ' = ' + str(D))

        # add measurement to appropriate list (only need 1, but can take averages if multiple)
        if what_to_measure == 'width':
            width = D
        elif what_to_measure == 'length':
            length = D
    
    return length, width

def badInput(frame_number, movie_file):
    print(frame_number + ' is not a valid frame in ' + movie_file)
    exit()

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
    return D, what_to_measure

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


def promptForMeasurement():
    selection = input('\nMeasure (w)idth or (l)ength or (q)uit?: ' ).rstrip()
    # print('you pressed ' + selection) # testing
    if selection == 'w':
        what_to_measure = 'width'
        print('  You selected width - measure across body between leg pairs 2 and 3')
    elif selection == 'l':
        what_to_measure = 'length'
        print('  You selected length - measure from nose to space between 4th leg pair')
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