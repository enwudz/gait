#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 06:46:13 2022

@author: iwoods

From a movie file that has been tracked with autoTracker ...
    show path of centroids along the video (time gradient color)
    show frame times within video (time gradient color)
    show timing of turns (with decreasing alpha each frame) 
        and stops as text on the movie frames. 
    
to make movie from saved frames, run:
    python makeMovieFromImages.py 'searchterm' fps outfile
    ('searchterm' like '*frames*')

    
"""

import sys
import numpy as np
import matplotlib as mpl
import cv2
import gaitFunctions

def main(movie_file, save_frames = False):
    
    if save_frames == 'True':
        print('Saving frames to make a movie ... run ')
        print("python makeMovieFromImages.py 'searchterm' fps outfile")
        print(" ... like ... ")
        print("python makeMovieFromImages.py '*frames*png* 30 out.mp4")
    
    # plotting stuff to adjust
    font = cv2.FONT_HERSHEY_DUPLEX
    marker_size = 10
    text_size = 2
    turn_color = (155, 155, 0)
    stop_color = (15, 0, 100)
    time_x, time_y = [0.05, 0.1] # where should we put the time label?
    turn_x, turn_y = [0.05, 0.95] # where should we put the turn label?
    stop_x, stop_y = [0.05, 0.8] # where should we put the stop label?
    colormap = 'plasma' # plasma, cool, Wistia, autumn, rainbow
    
    # read in times and coordinates
    tracked_df, excel_filename = gaitFunctions.loadTrackedPath(movie_file)
    frametimes = tracked_df.times.values    
    xcoords = tracked_df.xcoords.values
    ycoords = tracked_df.ycoords.values
    stops = tracked_df.stops.values
    turns = tracked_df.turns.values
    
    # get timing of turns and stops
    # these are arrays of frametimes when there are stops and turns
    stop_times = frametimes[np.where(stops==1)]
    turn_times = frametimes[np.where(turns==1)]
    
    # get alphas for labels of turns and stops
    label_buffer = 30 # in frames
    stop_alphas = gaitFunctions.labelTimes(frametimes, stop_times, label_buffer)
    turn_alphas = gaitFunctions.labelTimes(frametimes, turn_times, label_buffer)
    
    # get colors for coordinates and times (coded for time)
    num_frames = getFrameCount(movie_file) 
    dot_colors = makeColorList(colormap, num_frames)
    
    # get video
    vid = cv2.VideoCapture(movie_file)
    frame_number = 0
    filestem = movie_file.split('.')[0]
    
    # checking frame times (from centroid file) vs. what cv2 says . . .
    #print(len(frametimes), frames_in_video) # these are not the same sometimes
    #print(frametimes[-5:])
    
    # set text positions for times, turns, stops
    vid_width  = int(vid.get(3))
    vid_height = int(vid.get(4))
    
    # where should we put the labels for frame time and stop and turn?
    time_stamp_position = (int(time_x * vid_width), int(time_y * vid_height) )
    turn_position =  (int(turn_x * vid_width), int(turn_y * vid_height) )
    stop_position = (int(stop_x * vid_width), int(stop_y * vid_height) )
    
    while vid.isOpened():
        ret, frame = vid.read()
        
        if (ret != True):  # no frame!
            print('... video end!')
            break    
    
        frametime = np.round(frametimes[frame_number], decimals = 3)
        
        # TIMES (color coded)
        frame = cv2.putText(frame, str(frametime).ljust(5,'0'),
                            time_stamp_position, # position
                            font, text_size,
                            dot_colors[frame_number], # color
                            4, cv2.LINE_8)
        
        # COORDINATES (color-coded to time)
        # ==> SINGLE coordinate
        # x = xcoords[frame_number]
        # y = ycoords[frame_number]
        # cv2.circle(frame, (x, y), 5, dot_colors[frame_number], -1)
        # ==> add ALL coordinates so far
        frame  = addCoordinatesToFrame(frame, xcoords[:frame_number+1], ycoords[:frame_number+1], dot_colors, marker_size)
                
        # add text for turns (fade in before and out after by text alpha)
        if turn_alphas[frame_number] == 1:
            cv2.putText(frame, 'Turn', turn_position, font, text_size, turn_color, 4, cv2.LINE_8)
        elif turn_alphas[frame_number] > 0:
            overlay = frame.copy()
            cv2.putText(overlay, 'Turn', turn_position, font, text_size, turn_color, 4, cv2.LINE_8)
            frame = cv2.addWeighted(overlay, turn_alphas[frame_number], frame, 1 - turn_alphas[frame_number], 0) 
        
        # add text for stops
        if stop_alphas[frame_number] == 1:
            cv2.putText(frame, 'Stop', stop_position, font, text_size, stop_color, 4, cv2.LINE_8)
        elif stop_alphas[frame_number] > 0:
            overlay = frame.copy()
            cv2.putText(overlay, 'Stop', stop_position, font, text_size, stop_color, 4, cv2.LINE_8)
            frame = cv2.addWeighted(overlay, stop_alphas[frame_number], frame, 1 - stop_alphas[frame_number], 0) 
        
        # show the frame
        cv2.imshow('press (q) to quit', frame) # frame or binary_frame
        
        # save frame to file
        if save_frames == True:
            saveFrameToFile(filestem, frame_number, frame) 
        
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break      
        
        frame_number += 1
     
    print(frame_number)    
    vid.release()
    cv2.destroyAllWindows()
    
    return

def saveFrameToFile(file_stem, frame_number, frame):
    # to make a movie from frames
    # conda install ffmpeg
    # ffmpeg -f image2 -r 10 -s 1080x1920 -pattern_type glob -i '*.png' -vcodec mpeg4 movie.mp4
    # -r is framerate of movie
    
    file_name = file_stem + '_frames_' + str(frame_number).zfill(8) + '.png'
    cv2.imwrite(file_name, frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        

def addCoordinatesToFrame(frame, xcoords, ycoords, colors, markersize=5):
    '''

    Parameters
    ----------
    frame : open CV image
    coordinates : list of tuples
        a list of tuples of coordinates within frame
    colors : list of colors
        a list of tuples of cv2 formatted colors ... can be longer than coordinates list

    Returns
    -------
    frame with a dot positioned at each coordinate

    '''
    for i, xcoord in enumerate(xcoords):
        cv2.circle(frame, (int(xcoord), int(ycoords[i])), markersize, colors[i], -1)

    return frame

def getFrameCount(videofile):
    """get the number of frames in a movie file"""
    cap = cv2.VideoCapture(videofile)
    num_frames = int(cap.get(cv2. CAP_PROP_FRAME_COUNT))
    print('... number of frames in ' + videofile + ' : ' + str(num_frames) )
    cap.release()
    return num_frames

def makeColorList(cmap_name, N):
     
    # cmap = cm.get_cmap(cmap_name, N)
    cmap = mpl.colormaps.get_cmap(cmap_name)
    cmap = cmap(np.arange(N))[:,0:3]
    cmap = np.fliplr(cmap)
     
     # format for cv2 = 255 is max pixel intensity, colors are BGR     
    cmap = cmap * 255 # for opencv colors
     # convert RGB to BGR ... apparently no need! 
     # cmap = [[color[0], color[1], color[2]] for i, color in enumerate(cmap)]
    
    return [tuple(i) for i in cmap]
    
if __name__== "__main__":

    save_frames = False    
    if len(sys.argv) > 2:
        if sys.argv[2] == 'True':
            save_frames = True
    if len(sys.argv) > 1:
        movie_file = sys.argv[1]
    else:
        movie_file = gaitFunctions.selectFile(['mp4','mov'])

    main(movie_file, save_frames)