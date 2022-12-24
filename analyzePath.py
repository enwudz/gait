#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 20:22:50 2022

@author: iwoods
"""

import sys
import glob
import numpy as np
import pandas as pd
import scipy.signal
import gaitFunctions

'''
WISH LIST


'''

def main(movie_file, plot_style = 'none'): # plot_style is 'track' or 'time'
    
    # tracking data is from trackCritter.py, and is in an excel file for this clip
    
    # load identity info for this clip
    info = gaitFunctions.loadIdentityInfo(movie_file)
    
    # get scale (conversion between pixels and millimeters)
    scale = getScale(info)
    print('... 1 mm = ' + str(scale) + ' pixels')
        
    # load the tracked data
    tracked_data, excel_filename = gaitFunctions.loadTrackedPath(movie_file)

    # read in data
    frametimes = tracked_data.times.values
    areas = tracked_data.areas.values / scale**2
    xcoords = tracked_data.xcoords.values
    ycoords = tracked_data.ycoords.values 
    
    # smooth the coordinates!
    smoothedx = smoothFiltfilt(xcoords,3,0.05)
    smoothedy = smoothFiltfilt(ycoords,3,0.05)

    # get vectors for  distance, speed, cumulative_distance, bearings, bearing_changes
    distance, speed, cumulative_distance, bearings, bearing_changes = distanceSpeedBearings(frametimes, smoothedx, smoothedy, scale)
    
    # get vectors for stops and turns
    time_increment = 0.5 # in seconds
    stops, turns = stopsTurns(frametimes, speed, bearing_changes, time_increment)
    
    # add all tracking vectors to the excel file, 'pathtracking' tab
    d = {'times':frametimes, 'xcoords':xcoords, 'ycoords':ycoords, 'areas':areas, 
         'smoothed_x':smoothedx, 'smoothed_y':smoothedy, 'distance':distance,
         'speed':speed, 'cumulative_distance':cumulative_distance, 'bearings': bearings,
         'bearing_changes':bearing_changes, 'stops':stops, 'turns':turns}
    df = pd.DataFrame(d)
    with pd.ExcelWriter(excel_filename, engine='openpyxl', if_sheet_exists='replace', mode='a') as writer: 
        df.to_excel(writer, index=False, sheet_name='pathtracking')
    
    # add path tracking summary values to 'path_stats' tab
    # area, distance, average speed, num turns, num stops, bearings, time_increment for turns & stops
    parameters = ['area','clip duration','total distance','average speed','# turns','# stops','cumulative bearings','bin duration']
    median_area = np.median(areas)
    clip_duration = frametimes[-1]
    total_distance = np.sum(distance)
    average_speed = np.mean(speed[:-1])
    num_turns = len(gaitFunctions.one_runs(turns))
    num_stops = len(gaitFunctions.one_runs(stops))
    cumulative_bearings = np.sum(bearing_changes)
    vals = [median_area, clip_duration, total_distance, average_speed, num_turns, num_stops, cumulative_bearings, time_increment]
    
    path_stats = zip(parameters, vals)
    df2 = pd.DataFrame(path_stats)
    df2.columns = ['path parameter', 'value']
    with pd.ExcelWriter(excel_filename, engine='openpyxl', if_sheet_exists='replace', mode='a') as writer: 
        df2.to_excel(writer, index=False, sheet_name='path_stats')
    
    # PLOTS
    if plot_style != 'none':
        # plot!
        import plotPath
        plotPath.main(movie_file, plot_style) 
        
    return df, df2

def change_in_bearing(bearing1, bearing2):
    # need some care:  if successive bearings cross north (i.e. 0/360) ...
    # both will be near (e.g. within ~20 degrees) of 0 or 360
    # and so we need to adjust how we calculate difference in bearing
    
    buffer = 50
    
    if bearing1 > 360-buffer and bearing2 < buffer: # the path crossed North
        delta_bearing = bearing2 + 360 - bearing1
    elif bearing2 > 360-buffer and bearing1 < buffer: # the path crossed North
        delta_bearing = 360 - bearing2 + bearing1
    else:
        delta_bearing = np.abs(bearing1 - bearing2)
    
    return delta_bearing

def stopsTurns(times, speed, bearing_changes, increment):    
    
    '''
    From vectors of speed and bearings ...
    group into bins based on a time increment

    estimate when tardigrade stops (where speed is < threshold)
    estimate when discrete turns in the path (where angle of turn > threshold)

    Parameters
    ----------
    times : numpy array
        times of each video frame
    speed : numpy array
        from distanceSpeedBearings, vector of speed in each video frame
    bearing_changes : numpy array
        from distanceSpeedBearings, vector of bearing change in each video frame
    increment : float
        increment duration (in seconds) to bin video frame bins

    Returns
    -------
    
    stops = binary vector (1 = stopped, 0 = moving)
    turns = binary vector (1 = turning, 0 = not turning)

    '''
    
    # initialize empty arrays
    stops = np.zeros(len(speed))
    turns = np.zeros(len(speed))
    
    # make bins
    current_time = 0
    video_length = times[-1]
    
    # define stop threshold
    # if mean speed for a bin is below this threshold, it is a STOP!
    mean_speed = np.mean(speed)
    stop_threshold = 0.5 * mean_speed 
    
    # define turn threshold
    # if change in bearing in a bin is greater than this threshold, it is a TURN!
    turn_threshold = 28 # in degrees
    
    while current_time + increment <= video_length:
        
        # get bin
        next_time = current_time + increment
        start_bin = np.where(times >= current_time)[0][0]
        end_bin = np.where(times >= next_time)[0][0]
        
        # find STOPs
        # look at AVERAGE SPEED of this bin
        # if below a threshold for speed? = a STOP
        # in STOPS, set all frames of this bin to 1
        if np.mean(speed[start_bin:end_bin]) <= stop_threshold:
            stops[start_bin:end_bin] = 1
                         
        # find TURNS
        # look at total change in bearing from this bin
        # if ABOVE a threshold (eg 28 degrees)? = a TURN
        # in TURNS, set all frames of this bin to 1
        # what if turned a bit one way, then turned back? Hmmmm...
        if np.sum(bearing_changes[start_bin:end_bin]) >= turn_threshold:
            turns[start_bin:end_bin] = 1

        current_time += increment
    
    # deal with last portion of video that is less than time increment in duration
    # or don't . . . leave that off?
    # if current_time < video_length:
    #     start_bin = np.where(times >= current_time)[0][0]
    #     if np.mean(speed[start_bin:-1]) <= stop_threshold:
    #         stops[start_bin:-1] = 1
    #     if np.sum(bearing_changes[start_bin:-1]) >= turn_threshold:
    #         turns[start_bin:-1] = 1

    return stops, turns

def distanceSpeedBearings(times, xcoords, ycoords, scale):
    '''
    for all video frames: 
    make vectors of speed, distance, cumulative distance, bearing

    Parameters
    ----------
    times : numpy array
        times of each video frame
    xcoords : numpy array
        x coordinates (usually smoothed)
    ycoords : numpy array
        y coordinates (usually smoothed)
    scale : float
        number of pixels for 1 mm real distance

    Returns
    -------

    vectors of same length as times:
    distance = distance traveled per video frame
    speed = speed of travel during video frame
    cumulative_distance = distance traveled from beginning through video frame
    bearing = change in bearing during video frame (this will be ZERO if stopped)
    
    stops = binary vector (1 = stopped, 0 = moving)
    turns = binary vector (1 = turning, 0 = not turning)

    '''

    # get vector of distances traveled in every frame
    # could probably do list comprehension
    distance = np.zeros(len(times))
    speed = np.zeros(len(times))
    cumulative_distance = np.zeros(len(times))
    bearings = np.zeros(len(times))
    bearing_changes = np.zeros(len(times))
    
    for i, time in enumerate(times[:-1]):
        current_x = xcoords[i]
        current_y = -ycoords[i] # we do -y because y=0 is the top of the image
        next_x = xcoords[i+1]
        next_y = -ycoords[i+1] # we do -y because y=0 is the top of the image
        
        start_coord = np.array([current_x, current_y])
        end_coord = np.array([next_x, next_y])
        
        time_interval = times[i+1] - times[i]
        
        distance_in_frame = np.linalg.norm(start_coord - end_coord) / scale
        
        distance[i] = distance_in_frame
        speed[i] = distance_in_frame / time_interval
        
        bearings[i] = getBearing(start_coord, end_coord)
        
        if i == 0:
            cumulative_distance[i] = distance_in_frame
            bearing_changes[i] = 0
        else:
            cumulative_distance[i] = cumulative_distance[i-1] +  distance_in_frame
            delta_bearing = change_in_bearing(bearings[i], bearings[i-1])
            bearing_changes[i] = delta_bearing
    
    return distance, speed, cumulative_distance, bearings, bearing_changes

def getScale(info):

    # does info have the scale already?
    if 'scale' in info.keys():    
        scale = float(info['scale'])
    else:

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
            print('no scale for ' + info['file_stem'])
            print('... measure 1mm on the micrometer (see image) ... ')
            micrometerFiles = glob.glob('*micrometer*')
            if len(micrometerFiles) > 0:
                import measureImage
                micrometerFile = micrometerFiles[0]
                scale = float(measureImage.main(micrometerFile))
    
            else:
                print('no micrometer image ... ')
                scale = 1
        
        # update the excel file
        print('... adding scale to excel file for this clip ... ')
        excel_filename = info['file_stem'] + '.xlsx'
        parameters = gaitFunctions.identity_print_order()
        vals = [info[x] for x in parameters]
        parameters.append('scale')
        vals.append(scale)
        
        d = {'Parameter':parameters,'Value':vals}
        df = pd.DataFrame(d)
        with pd.ExcelWriter(excel_filename, engine='openpyxl', if_sheet_exists='replace', mode='a') as writer: 
            df.to_excel(writer, index=False, sheet_name='identity')
        
    #print('Scale is ' + str(scale))
    return scale

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

if __name__== "__main__":

    if len(sys.argv) > 1:
        movie_file = sys.argv[1]
        try:
            plot_style = sys.argv[2]
        except:
            plot_style = 'none'
    else:
        movie_file = gaitFunctions.select_movie_file()
        plot_style = 'none'
       
    print('Movie is ' + movie_file)

    main(movie_file, plot_style)
