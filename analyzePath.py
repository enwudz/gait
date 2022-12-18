#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 20:22:50 2022

@author: iwoods
"""

import sys
import glob
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import cv2
import scipy.signal

'''
WISH LIST
change to only one plot option (currenlty is scatter or line)
change color for smoothed path = time gradient
change color for raw path = light gray(?)

get movie parameters from the *tracked* files instead of the movie
    fps, time ... 
    
think about next step - what is going take these updated track files, and
    combine clips from single tardigrades
    save summary data

'''

def main(tracking_data, plot_style = 'line'): # scatter or line or none
    
    # tracking_data file is from trackCritter.py

    # get height, width, fps from filestem
    filestem = tracking_data.split('_tracked')[0]
    movie_file = filestem + '.mov'
    (vid_width, vid_height, vid_fps, vid_frames, vid_length) = getVideoData(movie_file, False)

    # read in data
    df = pd.read_csv(tracking_data, names = ['frametime','area','x','y'], header=None)
    frametimes = df.frametime.values
    areas = df.area.values
    median_area = np.median(areas)
    
    # get fps, video time from data
    
    # get coordinates
    xcoords = df.x.values
    ycoords = df.y.values

    # smooth the coordinates!
    smoothedx = smoothFiltfilt(xcoords,3,0.05)
    smoothedy = smoothFiltfilt(ycoords,3,0.05)

    # calculate distance from smoothed data
    distance = cumulativeDistance(smoothedx, smoothedy)

    # calculate # of turns, # of speed changes (from smoothed data)
    time_increment = 0.5 # in seconds
    num_stops, discrete_turns, angle_space, stop_times, turn_times = turnsStartsStops(frametimes, smoothedx, smoothedy, vid_fps, time_increment)

    if plot_style != 'none':

        if plot_style == 'line':
            # ==> line plot to compare raw path with smoothed path
            f, a = plotSmoothedPath(filestem, xcoords, ycoords, smoothedx, smoothedy)

        else:
            # ==> scatter plot of centroids along path with colormap that shows time
            f, a = plotPathScatter(filestem, xcoords, ycoords, vid_length)

        # ==> add labels from experiment and show plot:
        a.set_xlabel(getDataLabel(median_area, distance, vid_length, angle_space, discrete_turns, num_stops ))
        a.set_xticks([])
        a.set_yticks([])
        plt.title(filestem)
        plt.show()

    # print out data
    # fileData = filestem.split('_')
    # if len(fileData) == 4:
    #     initials, date, treatment, tardistring = filestem.split('_')
    #     timeRange = ''
    # elif len(fileData) == 5:
    #     initials, date, treatment, tardistring, timeRange = filestem.split('_')
        
    # if 'tardigrade' in tardistring:
    #     tardigrade = tardistring.split('tardigrade')[1].split('-')[0]
    # else:
    #     tardigrade = tardistring
    datastring = filestem + ',' + str(getScale(filestem))
    # datastring += ',' + ','.join([initials,date,treatment,tardigrade,timeRange])
    datastring += ',' + ','.join([str(x) for x in [vid_length, median_area, distance, discrete_turns, angle_space, num_stops]])
    print(datastring)

    return stop_times, turn_times


def getScale(filestem):

    scaleFile = glob.glob('*scale.txt')
    if len(scaleFile) > 0:
        with open(scaleFile[0],'r') as f:
            stuff = f.readlines()
            for thing in stuff:
                if '=' in thing:
                    scale = float(thing.split('=')[1])
                else:
                    scale = float(thing)
    else:
        print('no scale for ' + filestem)
        micrometerFiles = glob.glob('*micrometer*')
        if len(micrometerFiles) > 0:
            import measureImage
            micrometerFile = micrometerFiles[0]
            scale = float(measureImage.main(micrometerFile))

        else:
            print('no micrometer image ... ')
            scale = 1

    #print('Scale is ' + str(scale))
    return scale

def getFirstLastFrames(filestem):
    first_frame_file = filestem + '_first.png'
    last_frame_file = filestem + '_last.png'
    
    try:
        first_frame = cv2.imread(first_frame_file)
    except:
        vidcap = cv2.VideoCapture(filestem + '.mov')
        success, image = vidcap.read()
        if success:
            first_frame = image
        else:
            print('cannot get an image from ' + filestem)
            first_frame = None
    
    try:
        last_frame = cv2.imread(last_frame_file)
    except:
        vidcap = cv2.VideoCapture(filestem + '.mov')
        frame_num = 1
        good_frame = None
        while vidcap.isOpened():
            ret, frame = vidcap.read()
            if ret == False:
                print('Last successful frame = ' + str(frame_num))
                last_frame = good_frame
                vidcap.release()
            else:
                frame_num += 1
                good_frame = frame
    
    return first_frame, last_frame

def superImposedFirstLast(filestem):
    # superimpose first and last frames
    first_frame, last_frame = getFirstLastFrames(filestem)
    combined_frame = cv2.addWeighted(first_frame, 0.3, last_frame, 0.7, 0)
    return combined_frame

def plotPathScatter(filestem, xcoords, ycoords, vid_length):

    combined_frame = superImposedFirstLast(filestem)
    

    f, a = plt.subplots(1, figsize=(14,6))
    a.imshow(combined_frame) # combined_frame or last_frame
    cmap_name = 'plasma'
    cmap = mpl.cm.get_cmap(cmap_name)
    cols = cmap(np.linspace(0,1,len(xcoords)))
    a.scatter(xcoords,ycoords, c = cols, s=10)
    a.set_xticks([])
    a.set_yticks([])
    # add legend for time
    norm = mpl.colors.Normalize(vmin=0, vmax=vid_length)
    plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), label = 'Time (sec)')

    return f, a

def plotSmoothedPath(filestem, xcoords, ycoords, smoothedx, smoothedy):

    combined_frame = superImposedFirstLast(filestem)

    f, a = plt.subplots(1, figsize=(14,6))
    a.imshow(combined_frame) # combined_frame or last_frame
    plt.plot(xcoords,ycoords, linewidth=8, color = 'forestgreen', label = 'raw') # raw coordinates
    plt.plot(smoothedx,smoothedy, linewidth=2, color = 'lightgreen', label = 'smoothed') # smoothed
    plt.legend()
    return f, a

def turnsStartsStops(times, xcoords, ycoords, vid_fps, increment):
    '''
    From x and y coordinates of a path, group into increments of length binsize

    estimate the number of times the tardgirade stops (where speed is < threshold)
    estimate the number of discrete turns in the path (where angle of turn > threshold)
    estimate the total amount of angle space explored along the path
    [old code] estimate the number of times there is a change in speed greater than a specified threshold

    Parameters
    ----------
    times : numpy array
        times of each video frame
    xcoords : numpy array
        x coordinates
    ycoords : numpy array
        y coordinates.
    vid_fps : integer
        frames (i.e. coordinates) per second in video
    increment : integer
        increment duration (in seconds) for path coordinate bins

    Returns
    -------

    num_stops = integer amount of # of stops
    discrete_turns = integer amount of # of turns
    angle_space = floating point number of cumulative changes in path angle
    [old code] speed_changes = integer amount of # of changes in speed

    '''

    # get the number of points per time increment
    points_in_bin = int(vid_fps * increment)

    # get duration of the video in seconds
    video_length = np.around(len(xcoords) / vid_fps, decimals = 2)

    # get distance traveled along path
    path_distance = cumulativeDistance(xcoords, ycoords)

    # get average speed
    average_speed = np.around(path_distance / video_length, decimals = 2)

    # bin the times and coordinates
    binned_time = binList(times, points_in_bin)
    binned_x = binList(xcoords, points_in_bin)
    binned_y = binList(ycoords, points_in_bin)
    speeds = np.zeros(len(binned_x))
    bearings = np.zeros(len(binned_x))

    # calculate speed and angle for each bin
    for i, xbin in enumerate(binned_x): # could probably do list comprehension
        start_coord = np.array([xbin[0], -binned_y[i][0]]) # we do -y because y=0 is the top of the image
        end_coord = np.array([xbin[-1], -binned_y[i][-1]])

        # calculate speed in this increment
        distance_for_bin = np.linalg.norm(start_coord - end_coord)
        time_in_bin = len(xbin) / vid_fps
        speeds[i] = np.around(distance_for_bin / time_in_bin, decimals=2)

        # calculate angle in this increment
        bearings[i] = getBearing(start_coord, end_coord)
        # print(start_coord, end_coord, angles[i])

    # just printing the turn angles to test things out
    # np.set_printoptions(suppress=True)
    # print(bearings)

    # ==> from speeds and angles, FIND speedChanges, discrete_turns, angle_space
    # DEFINE THRESHOLDS for changes in speed or direction
    # for turn, define a discrete turn as a turn that is greater than
    #     X (?) degrees ... when SPEED is above a certain threshold?
    # for stops, define a stop as an interval with speed < X% of average speed
    #     X% (?) of the average speed across the path?
    # [old code] for speed changes, define a change in speed as a change that is greater than
    #     X% (?) of the average speed across the path

    # [old] speed change threshold
    # [old] speed_change_percentage_threshold = 33 # percent of average speed
    # what is the magnitude of a 'real' change in speed?
    # [old] speed_change_threshold = np.around(speed_change_percentage_threshold/100 * average_speed, decimals = 2)
    # print('speed change threshold: ', speed_change_threshold)

    # define stop thresholds
    stop_percentage_threshold = 50 # percent of average speed
    stop_magnitude_threshold = np.around(stop_percentage_threshold/100 * average_speed, decimals = 2)
    #print('threshold for STOP: ', stop_magnitude_threshold)
    moving = True # current state of the movement (False if 'stopped')

    # define discrete turn thresholds
    turn_degree_threshold = 28 # degrees
    #print('threshold (degrees) for discrete turn: ', turn_degree_threshold)

    # set counters to zero
    # speed_changes = 0   # changes in speed that meet the thresholds above
    num_stops = 0       # number of times the speed slows before a defined threshold
    discrete_turns = 0  # changes in bearing that meet the thresholds above
    angle_space = 0     # cumulative total of changes in bearing

    # keep track of which bins have no movement (i.e. stops)
    stop_times = []

    # keep track of which bins have turns
    turn_times = []

    for i, speed in enumerate(speeds[:-1]):

        # [old code for speed change threshold]
        # what was the change in speed?
        # delta_speed = np.abs(speeds[i+1] - speeds[i])
        # # was this a 'discrete' change in speed?
        # if delta_speed >= speed_change_threshold:
        #     #print('change in speed: ', delta_speed)
        #     speed_changes += 1

        # Decide if this bin is a STOP
        if moving: # moving = True, the critter was MOVING before this bin
            # is the critter moving now?
            if speed <= stop_magnitude_threshold:
                moving = False # this critter WAS moving but now it has STOPPED!
                num_stops += 1
                stop_times.append(binned_time[i])
            # the critter was MOVING before, and it is still moving,
            # so we leave moving=True

        else: # moving = False, the critter was STOPPED before this bin
             # is the critter moving now?
             if speed > stop_magnitude_threshold:
                 moving = True # this critter was STOPPED but now it is moving!
             # if it is not moving before now,
             # and it is not moving now,
             # then we just leave moving=FALSE

        # What was the change in bearing between this bin and the previous one?
        # need some care:  if successive bearings cross north (i.e. 0/360) ...
        # both will be near (within ~20 degrees) of 0 or 360
        # and so we need to adjust how we calculate difference in bearing
        if bearings[i] < 20 and bearings[i+1] > 340: # the path crossed North
            delta_bearing = bearings[i] + 360 - bearings[i+1]
        elif bearings[i] > 340 and bearings[i+1] < 20: # the path crossed North
            delta_bearing = 360 - bearings[i] + bearings[i+1]
        else:
            delta_bearing = np.abs(bearings[i+1]-bearings[i])
        angle_space += delta_bearing # cumulative total of changes in bearing

        # Decide if this bin is a 'discrete' change in bearing (i.e. a 'turn')?
        # if moving and delta_bearing >= turn_degree_threshold:
        if delta_bearing >= turn_degree_threshold:
            #print('A TURN!')
            discrete_turns += 1
            turn_times.append(binned_time[i])

    angle_space = np.around(angle_space, decimals=2)
    printMe = False
    if printMe == True:
        #printString = 'Speed changes: ' + str(speed_changes)
        printString = 'Stops: ' + str(num_stops)
        printString += ', Discrete turns: ' + str(discrete_turns)
        printString += ', Explored angles: ' + str(angle_space)
        print(printString)

    return num_stops, discrete_turns, angle_space, stop_times, turn_times

def getBearing(p1, p2):
    '''
    Given two coordinates, calculate the bearing of the direction of a line
    connecting the first point with the second. Bearing is 0 (North) - 360

    Parameters
    ----------
    p1 : tuple
        x and y coordinate of point 1
    p2 : tuple
        x and y coordinate of point 1.

    Returns
    -------
    bearing : floating point decimal
        Bearing between x and y. 0 = North/Up, 90 = East/Right, 180 = South/Down, 270 = West/Left

    '''
    deltaX = p2[0]-p1[0]
    deltaY = p2[1]-p1[1]
    degrees = np.arctan2(deltaX,deltaY) / np.pi * 180
    if degrees < 0:
        degrees = 360 + degrees
    return np.around(degrees, decimals = 2)

def binList(my_list, bin_size):
    '''
    Break up a list into a list of lists, where each list has length of bin_size

    Parameters
    ----------
    my_list : a list
        list of anything.
    bin_size : integer
        number of items in each bin.

    Returns
    -------
    binned_list : list (of lists)
        a list of lists, where each list has length of bin_size.

    '''
    binned_list = [my_list[x:x+bin_size] for x in range(0, len(my_list), bin_size)]
    return binned_list

def getDataLabel(area, distance, vid_length, angle_space = 0, discrete_turns = 0, num_stops = 0):
    # convert from pixels?
    speed = np.around(distance/vid_length, decimals = 2)
    data_label = 'Area : ' + str(area)
    data_label += ', Distance : ' + str(distance)
    data_label += ', Time: ' + str(vid_length)
    data_label += ', Speed: ' + str(speed)
    data_label += ', Stops: ' + str(num_stops)

    # angle space
    if angle_space > 0:
        data_label += ', Angles explored: ' + str(angle_space)
        data_label += ', Turns: ' + str(discrete_turns)

    return data_label

def cumulativeDistance(x,y):
    '''
    Given a list of x and y points, calculate the cumulative distance of a path
    connecting the first x,y with the last x,y

    Parameters
    ----------
    x : list
        list of x coordinates.
    y : list
        list of y coordinates..

    Returns
    -------
    cumulative_distance : floating point decimal
        the distance of the path connecting the first point with the last point.

    '''
    # adapted from https://stackoverflow.com/questions/65134556/compute-cumulative-euclidean-distances-between-subsequent-pairwise-coordinates
    XY = np.array((x, y)).T
    cumulative_distance = np.linalg.norm(XY - np.roll(XY, -1, axis=0), axis=1)[:-1].sum()
    return np.around(cumulative_distance, decimals = 2)

def smoothFiltfilt(x, pole=3, freq=0.1):

    '''
    adapted from https://swharden.com/blog/2020-09-23-signal-filtering-in-python/
    output length is same as input length
    as freq increases, signal is smoothed LESS

    Parameters
    ----------
    x : numpy array
        numpy array of x or y coordinates
    pole : integer
        see documentation.
    freq : floating point decimal between 0 and 1
        see documentation

    Returns
    -------
    filtered: numpy array
        smoothed data

    '''

    b, a = scipy.signal.butter(pole, freq)
    filtered = scipy.signal.filtfilt(b,a,x)
    return filtered

def getVideoData(videoFile, printOut = True):
    if len(glob.glob(videoFile)) == 0:
        exit('Cannot find ' + videoFile)
    else:
        vid = cv2.VideoCapture(videoFile)
        vid_width  = int(vid.get(3))
        vid_height = int(vid.get(4))
        vid_fps = int(np.round(vid.get(5)))
        vid_frames = int(vid.get(7))
        vid.release()
        vid_length = np.around(vid_frames / vid_fps, decimals = 2)
    if printOut == True:
        printString = 'width: ' + str(vid_width)
        printString += ', height: ' + str(vid_height)
        printString += ', fps: ' + str(vid_fps)
        printString += ', #frames: ' + str(vid_frames)
        printString += ', duration: ' + str(vid_length)
        print(printString)
    return (vid_width, vid_height, vid_fps, vid_frames, vid_length)


if __name__ == "__main__":


    if len(sys.argv) > 1:
        tracking_data = sys.argv[1]
    else:
        tracking_list = glob.glob('*tracked*')
        tracking_data = tracking_list[0]

    #print('Getting data from ' + tracking_data)

    main(tracking_data)
