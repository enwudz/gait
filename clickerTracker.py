#!/usr/bin/python

'''
Track things that are not working with autoTracker
load video
report # frames
ask how many frames to track (suggest default = every 5th frame)
    be sure to include first and last frame
open each frame
record points
fill in missing points between the recorded frames


'''

import gaitFunctions
import manualCritterMeasurement
import initializeClip
import pandas as pd
import numpy as np
import math
import sys
import cv2

def main(movie_file):

    # grab references to the global variables
    global image, refPt, drawing, nameText    

    # get or make excel file for this clip
    excel_file_exists, excel_filename = gaitFunctions.check_for_excel(movie_file)
    if excel_file_exists == False:
        initializeClip.main(movie_file)    

    # get number of frames in movie from the excel file for this clip
    frameTimes = gaitFunctions.getFrameTimes(movie_file)
    
    # ask how many frames to track (suggest every 5th frame, plus first and last)
    suggested_frame_number = int(len(frameTimes)/5) + 2
    selection = input('\nEnter number of frames to track:  (d) = ' + str(suggested_frame_number) + '  : ')
    if selection == 'd': 
        num_frames_to_track = suggested_frame_number
    else:
        try:
            num_frames_to_track = int(selection)
        except:
            num_frames_to_track = suggested_frame_number
    print('\nWe will track ' + str(num_frames_to_track) + ' frames ...')
        
    # make list of frames to track
    frame_indices = np.linspace(5, len(frameTimes)-5, num_frames_to_track-2)
    frame_indices = np.insert(frame_indices, 0, 0)
    frame_indices = np.append(frame_indices, len(frameTimes)-1)
    frame_indices = [int(x) for x in frame_indices]
    # tracked_frames = [frameTimes[i] for i in frame_indices]
    
    # go through movie
    print('... starting video ' + movie_file)
    vid = cv2.VideoCapture(movie_file)
    fps = vid.get(5)
    print('Video is at ' + str(round(fps)) + ' frames per second')
    frame_number = 0
    
    xcoords = np.zeros(len(frameTimes))
    ycoords = np.zeros(len(frameTimes))
    
    while vid.isOpened():
        
        ret, image = vid.read()
         
        # if frame in list of frames to track ...
        if frame_number in frame_indices: 
            
            # if first frame
            if frame_number == 0:
                print('here is frame # 1')
                # gaitFunctions.displayFrame(frame)
                refPt = getPoint()
                xcoords[frame_number] = refPt[0][0]
                ycoords[frame_number] = refPt[0][1]
        
            else:
                print('here is frame # ' + str(frame_number))
                # gaitFunctions.displayFrame(frame)
                refPt = getPoint()
                xcoords[frame_number] = refPt[0][0]
                ycoords[frame_number] = refPt[0][1]
                
                # quit upon last frame
                if frame_number == frame_indices[-1]:
                    print('This is the last frame')
                    
                    # shut down the video capture object
                    vid.release()
                    cv2.destroyAllWindows()
        
        # update frame_number
        frame_number += 1
        
    # fill in gaps between points
    xcoords = fillGaps(xcoords)
    ycoords = fillGaps(ycoords)
    
    # smooth the coordinates
    # smoothedx = gaitFunctions.smoothFiltfilt(xcoords,3,0.05)
    # smoothedy = gaitFunctions.smoothFiltfilt(ycoords,3,0.05)
    
    # measure the length and width of the critter
    length, width = manualCritterMeasurement.main(movie_file)
    
    # calculate the area ... assume the critter is elliptical ...
    area = math.pi * 2 * length * 2 * width
    
    lengths = [length] * len(xcoords)
    widths = [width] * len(xcoords)
    areas = [area] * len(xcoords)
    
    d = {'times':frameTimes, 'xcoords':xcoords, 'ycoords':ycoords, 
         'areas':areas, 'lengths':lengths, 'widths':widths}
    
    df = pd.DataFrame(d)
    with pd.ExcelWriter(excel_filename, engine='openpyxl', if_sheet_exists='replace', mode='a') as writer: 
        df.to_excel(writer, index=False, sheet_name='pathtracking')
    

def fillGaps(arr):
    iszero = np.concatenate(([0], np.equal(arr, 0).view(np.int8), [0]))
    absdiff = np.abs(np.diff(iszero))
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
    
    for r in ranges:
        startval = arr[r[0]-1]
        endval = arr[r[1]]
        
        numToFill = r[1]-r[0]
        fill = np.linspace(startval,endval,numToFill+2)[1:-1]
        
        arr[r[0]:r[1]] = fill
    
    return arr

def getPoint():

    # grab references to the global variables
    global image, refPt, drawing, nameText
    refPt = []

    nameText = "Click on the center of mass; then (d)one or (r) reset"
    clone = image.copy()

    # keep looping until the 'd' (or 'q') key is pressed
    while True:
        # display the image and wait for a keypress
        cv2.namedWindow(nameText)
        cv2.setMouseCallback(nameText, clickPoint)
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
    return refPt

def clickPoint(event, x, y, flats, param):
    # grab references to the global variables
    global D, image, refPt, drawing, nameText

    # if the left mouse button was clicked, record the (x, y) coordinates
    if event == cv2.EVENT_LBUTTONDOWN:
            refPt = [(x, y)]  
            drawing = True
            cv2.circle(image, refPt[0], 10, (0,255,0), -1)
            cv2.imshow(nameText, image)

if __name__== "__main__":

    if len(sys.argv) > 1:
        movie_file = sys.argv[1]
    else:
        movie_file = gaitFunctions.selectFile(['mp4','mov'])
       
    print('Movie is ' + movie_file)

    main(movie_file)