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
import gaitFunctions

'''
WISH LIST

if no length, prompt to measure it

'''

def main(movie_file):
    
    # tracking data is from autoTracker.py, and is in an excel file for each clip
    
    # things we can adjust ... see also turn_threshold = 28 in stopsturns function
    time_increment = 0.3 # in seconds ... over what time duration should we look for stops and turns
    turn_threshold = 28 # in degrees ... minimum angle to be called a 'turn' within time increment
    cruise_bout_threshold = 3 # in seconds ... what is minimum time to count as a 'cruise'?
    
    # load tracked path stats for this clip (if available)
    path_stats = gaitFunctions.loadPathStats(movie_file)
    
    # get scale (conversion between pixels and real distance)
    scale, unit = getScale(path_stats, movie_file)

    # load the excel file with tracked data ==> pandas dataframe
    tracked_data, excel_filename = gaitFunctions.loadTrackedPath(movie_file)
    if tracked_data is None:
        print('No path tracking data yet - run trackCritter.py')
        return    
    
    cols = tracked_data.columns.values
    for c in cols:
        if 'uncertainty' in c:
            uncertainty_col = c
            try:
                pixel_threshold = int(c.split('_')[-1])
            except:
                pixel_threshold = 100

    # collect relevant tracking data from dataframe
    frametimes = tracked_data.times.values
    
    try:
        areas = tracked_data.areas.values 
        lengths = tracked_data.lengths.values 
        uncertainties = tracked_data[uncertainty_col].values
        
    except:
        areas = np.zeros(len(frametimes))
        lengths = np.zeros(len(frametimes))
        uncertainties = np.zeros(len(frametimes))
        uncertainty_col = 'uncertainty'
    
    # get coordinates of centroids at every frame    
    xcoords = tracked_data.xcoords.values
    ycoords = tracked_data.ycoords.values
    
    # smooth the coordinates!
    smoothedx = gaitFunctions.smoothFiltfilt(xcoords,3,0.05)
    smoothedy = gaitFunctions.smoothFiltfilt(ycoords,3,0.05)

    # from smoothed data, get vectors for distance, speed, cumulative_distance, bearings, bearing_changes
    distance, speed, cumulative_distance, bearings, bearing_changes = distanceSpeedBearings(frametimes, smoothedx, smoothedy)
    
    # get vectors for stops and turns
    stops, turns = stopsTurns(frametimes, speed, bearing_changes, bearings, time_increment, np.median(lengths), turn_threshold)
    
    # get % cruising
    non_cruising_proportion = np.count_nonzero(stops + turns) / len(stops)
    cruising_proportion = np.round( ( 1 - non_cruising_proportion ) * 100, 2)
    
    # get medians for critter size: area and length
    median_area = np.median(areas) / scale**2
    median_length = np.median(lengths) / scale
    
    ### add all tracking vectors to the excel file, 'pathtracking' tab
    d = {'times':frametimes, 'xcoords':xcoords, 'ycoords':ycoords, 'areas':areas, 'lengths':lengths,
         'smoothed_x':smoothedx, 'smoothed_y':smoothedy, 'distance':distance,
         'speed':speed, 'cumulative_distance':cumulative_distance, 'bearings': bearings,
         'bearing_changes':bearing_changes, 'stops':stops, 'turns':turns, uncertainty_col:uncertainties}
    df = pd.DataFrame(d)
    with pd.ExcelWriter(excel_filename, engine='openpyxl', if_sheet_exists='replace', mode='a') as writer: 
        df.to_excel(writer, index=False, sheet_name='pathtracking')
    
    ### add path tracking summary values to 'path_stats' tab

    # calculate summary values
    clip_duration = frametimes[-1]
    total_distance = np.sum(distance) / scale

    average_speed = np.mean(speed[:-1]) / scale
    num_turns = len(gaitFunctions.one_runs(turns))
    num_stops = len(gaitFunctions.one_runs(stops))
    cumulative_bearings = np.sum(np.abs(bearing_changes))
    try:
        tracking_confidence = gaitFunctions.getTrackingConfidence(uncertainties, pixel_threshold)
    except:
        tracking_confidence = 100
        pixel_threshold = 100
        
    # add cruise bout data
    cruise_bouts = cruiseBouts(turns,stops)
    cruise_bout_timing = []
    cruise_bout_durations = []
    cruise_bout_total_duration = 0
    for bout in cruise_bouts:
        bout_start = frametimes[bout[0]]
        if bout[1] >= len(frametimes):
            bout_end = frametimes[-1]
        else:
            bout_end = frametimes[bout[1]]
        bout_duration = np.round(bout_end-bout_start,3)
        if bout_duration >= cruise_bout_threshold:
            cruise_bout_timing.append(str(bout_start) + '-' + str(bout_end))
            cruise_bout_durations.append(bout_duration)
            cruise_bout_total_duration += bout_duration

    num_cruise_bouts = len(cruise_bout_timing)
    print('\n# cruising bouts: ' + str(num_cruise_bouts) + ', total seconds cruising: ' + str(cruise_bout_total_duration))
    timing_string = ';'.join(cruise_bout_timing)
    print('timing: ', timing_string)
    durations_string = ';'.join([str(x) for x in cruise_bout_durations])
    print('durations: ', durations_string)
            
    parameters = ['scale','unit','area','length','clip duration','total distance','average speed',
                  '# turns','# stops', '% cruising', '# cruise bouts', 'cruise bout durations', 'cruise bout timing','cumulative bearings','bin duration',
                  'pixel threshold','tracking confidence']
    vals = [scale, unit, median_area, median_length, clip_duration, total_distance, 
            average_speed, num_turns, num_stops, cruising_proportion, num_cruise_bouts, durations_string, timing_string, cumulative_bearings, 
            time_increment, pixel_threshold, tracking_confidence]
    
    path_stats = zip(parameters, vals)
    df2 = pd.DataFrame(path_stats)
    df2.columns = ['path parameter', 'value']
    with pd.ExcelWriter(excel_filename, engine='openpyxl', if_sheet_exists='replace', mode='a') as writer: 
        df2.to_excel(writer, index=False, sheet_name='path_stats')
    
        
    return df, df2

def cruiseBouts(turns, stops):
    cruising = turns + stops
    cruise_bouts = gaitFunctions.zero_runs(cruising)
    return cruise_bouts

def stopsTurns(times, speed, bearing_changes, bearings, increment, length, turn_threshold):    
    
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
        increment duration (in seconds) ... window duration to call stops and turns
    length : float
        length of critter in mm

    Returns
    -------
    
    stops = binary vector (1 = stopped, 0 = moving)
    turns = binary vector (1 = turning, 0 = not turning)

    '''
    
    # empty arrays
    stops = np.zeros(len(speed))
    turns = np.zeros(len(speed))
    
    # # make bins
    # current_time = 0
    # video_length = times[-1]
    
    # define speed threshold for stop
    # if mean speed in a time window (increment) is below this threshold, it is a STOP!
    threshold_distance = 0.15 * length # expressed as fraction of length of critter
    stop_threshold = threshold_distance * increment # STOP is below this speed
    
    # bin_number = 0
    
    time_of_last_batch = times[-1] - increment
    start_of_last_batch = np.where(times >= time_of_last_batch)[0][0]
    
    for i, time in enumerate(times[:start_of_last_batch]):
        
        next_time = time + increment
        start_bin = np.where(times >= time)[0][0]
        end_bin = np.where(times >= next_time)[0][0]
        mean_speed_in_bin = np.mean(speed[start_bin:end_bin])
        
        # find STOPs
        # look at AVERAGE SPEED of this bin
        # if below a threshold for speed? = a STOP
        # in STOPS, set all frames of this bin to 1
        # print(mean_speed_in_bin, stop_threshold) # to test!
        if mean_speed_in_bin <= stop_threshold:       
            stops[start_bin:end_bin] = 1
                         
        # find TURNS     
        # look at total change in bearing from this bin
        # # if ABOVE a threshold (eg 28 degrees)? = a TURN
        # # in TURNS, set all frames of this bin to 1
        # # this is a bit wonky if stopped or exploring very slowly
        if np.sum(np.abs(bearing_changes[start_bin:end_bin])) >= turn_threshold:
            turns[start_bin:end_bin] = 1
  
    return stops, turns

def distanceSpeedBearings(times, xcoords, ycoords):
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

    Returns
    -------

    vectors of same length as times:
    distance = distance traveled per video frame (in PIXELS)
    speed = speed of travel during video frame (in PIXELS / second)
    cumulative_distance = distance traveled from beginning through video frame (in PIXELS)
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
        
        distance_in_frame = np.linalg.norm(start_coord - end_coord)
        
        distance[i] = distance_in_frame
        speed[i] = distance_in_frame / time_interval
        
        bearings[i] = getBearing(start_coord, end_coord)
        
        if i == 0:
            cumulative_distance[i] = distance_in_frame
            bearing_changes[i] = 0
        else:
            cumulative_distance[i] = cumulative_distance[i-1] +  distance_in_frame
            delta_bearing = gaitFunctions.change_in_bearing(bearings[i], bearings[i-1])
            bearing_changes[i] = delta_bearing
    
    return distance, speed, cumulative_distance, bearings, bearing_changes

def getScale(path_stats, movie_file):

    # does path_stats have the scale already?
    if 'scale' in path_stats.keys():    
        scale = float(path_stats['scale'])
        if 'unit' in path_stats.keys():
            unit = path_stats['unit']
        else:
            unit = '1 mm'
    
    # is there a file that contains measurement from  micrometer image?
    elif len(glob.glob('*scale.txt')) > 0:
        scaleFile = glob.glob('*scale.txt')[0]
        with open(scaleFile,'r') as f:
            stuff = f.readlines()
            for thing in stuff:
                if '=' in thing:
                    scale = float(thing.split('=')[1])
                    unit = thing.split('=')[0]
                else:
                    scale = float(thing)
                    unit = '1 mm'
                    
    # is there a micrometer image?
    elif len(glob.glob('*micrometer*png')) > 0:

        print('... measure 1mm on the micrometer (see image) ... ')
        micrometerFile = glob.glob('*micrometer*png')[0]
        import measureImage
        scale = float(measureImage.main(micrometerFile, True))
        unit = '1 mm'

    # no micrometer ... measure something on first frame or just ignore scale
    else:
        print(' ... no micrometer image and no scale file ... ')
        
        print('\nOptions: ')
        print('   1. enter a known scale (e.g. 1000 pix per mm)')
        print('   2. measure a distance on the first frame')
        print('   3. just set scale to 1 (i.e. ignore scale')
        print()
        selection = input('Which option? ')
        
        try:
            val = int(selection)
        except:
            sys.exit('Please choose a valid selection!')
        
        if val == 1:
            pixperunit = input('\nEnter number of pixels for a measured distance: ')
            
            try:
                scale = float(pixperunit)
            except:
                sys.exit('Please enter a number!')
            unit = input('\nEnter unit of measurement (e.g. inch or cm or mm): ')

        elif val == 2:
            firstframe = gaitFunctions.getFirstFrame(movie_file)
            imfile = movie_file.split('.')[0] + '_first.png'
            
            import cv2
            import os
            import measureImage
            
            cv2.imwrite(imfile, firstframe, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            pixel_distance = measureImage.main(imfile, False)
            os.remove(imfile)
            
            distance_measured = input('\nEnter numerical value of real distance (e.g. 95): ')
            
            try:
                measurement = float(distance_measured)
            except:
                sys.exit('Please enter a number!')
                
            unit = input('\nEnter unit of measurement (e.g. inch or cm or mm): ')
            
            scale = pixel_distance / measurement
            
        else:
            scale = 1
            unit = 'no units'

    print('Scale is ' + str(np.round(scale,2)) + ' pixels per ' + unit)
    return scale, unit

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


if __name__== "__main__":

    if len(sys.argv) > 1:
        movie_file = sys.argv[1]

    else:
        movie_file = gaitFunctions.selectFile(['mp4','mov'])
       
    print('Movie is ' + movie_file)

    main(movie_file)
