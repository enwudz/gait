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

'''

def main(movie_file):
    
    # tracking data is from autoTracker.py, and is in an excel file for each clip
    
    ''' things we can adjust to determine what is a STOP and what is a TURN '''
    # What time window should we look for stops and turns
    time_increment = 0.3 # in seconds ... 
    
    # define STOP = distance threshold expressed as fraction of body length of critter
    # if distance moved in a time window (increment) is below this threshold, it is a STOP!
    # ... for example, need to move 0.15 body lengths in 0.3 seconds
    stop_threshold = 0.03 # so distance threshold = stop_threshold * body_length
    
    # define threshold for TURNS
    turn_threshold = 28 # in degrees ... minimum angle to be called a 'turn' within time increment
    cruise_bout_threshold = 3 # in seconds ... what is minimum time to count as a 'cruise'?
    
    ''' finished setting parameters, get going on analysis '''
    # load tracked path stats for this clip (if available)
    path_stats = gaitFunctions.loadPathStats(movie_file)
    
    # get scale (conversion between pixels and real distance)
    scale, unit = getScale(path_stats, movie_file)

    # load the excel file with tracked data ==> pandas dataframe
    tracked_data, excel_filename = gaitFunctions.loadTrackedPath(movie_file)
    if tracked_data is None:
        print('No path tracking data yet - run autoTracker.py or clickerTracker.py')
        return
    else:
        print('\nAnalyzing tracked path from ' + movie_file)
    
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
        # units for areas, lengths, widths are in pixels
        areas = tracked_data.areas.values 
        lengths = tracked_data.lengths.values
        widths = tracked_data.widths.values
        uncertainties = tracked_data[uncertainty_col].values
        
    except:
        areas = np.zeros(len(frametimes))
        lengths = np.zeros(len(frametimes))
        widths = np.zeros(len(frametimes))
        uncertainties = np.zeros(len(frametimes))
        uncertainty_col = 'uncertainty'
    
    # get coordinates of centroids at every frame    
    xcoords = tracked_data.xcoords.values
    ycoords = tracked_data.ycoords.values
    
    # smooth the coordinates!
    smoothedx = gaitFunctions.smoothFiltfilt(xcoords,3,0.05)
    smoothedy = gaitFunctions.smoothFiltfilt(ycoords,3,0.05)
    # import matplotlib.pyplot as plt # smoothing quality control
    # plt.plot(ycoords,'r')
    # plt.plot(smoothedy,'k')
    # plt.show()
    # exit()

    # from smoothed data, get vectors for distance, speed, cumulative_distance, bearings, bearing_changes
    distance, speed, cumulative_distance, bearings = distanceSpeedBearings(frametimes, smoothedx, smoothedy)
    
    # get vector for stops
    print('stop threshold is ' + str(stop_threshold) + ' body lengths in ' + str(time_increment) + ' seconds' )
    distance_threshold = np.median(lengths) * stop_threshold # this is in pixels
    stops = getStops(frametimes, smoothedx, smoothedy, time_increment, distance_threshold)
    
    # get vector for turns
    filtered_bearings, bearing_changes, turns = getTurns(frametimes, stops, bearings, time_increment, turn_threshold)
    
    # get % cruising
    non_cruising_proportion = np.count_nonzero(stops + turns) / len(stops)
    cruising_proportion = np.round( ( 1 - non_cruising_proportion ) * 100, 2)
    
    # get medians for critter size: area and length
    median_area_pixels = np.median(areas)
    median_length_pixels = np.median(lengths)
    median_width_pixels = np.median(widths)
    median_area_scaled = np.median(areas) / scale**2
    median_length_scaled = np.median(lengths) / scale
    median_width_scaled = np.median(widths) / scale
    
    ### add all tracking vectors to the excel file, 'pathtracking' tab
    d = {'times':frametimes, 'xcoords':xcoords, 'ycoords':ycoords, 'areas':areas, 'lengths':lengths, 'widths':widths,
         'smoothed_x':smoothedx, 'smoothed_y':smoothedy, 'distance':distance,
         'speed':speed, 'cumulative_distance':cumulative_distance, 'bearings': bearings, 'filtered_bearings': filtered_bearings,
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
    speeds_during_cruising = []
    degrees_during_cruising = []
    cruise_bout_total_duration = 0
    cruise_bout_total_distance = 0
    num_cruising_frames= 0
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
            start_idx = np.min(np.where(frametimes>=bout_start))
            end_idx = np.min(np.where(frametimes>=bout_end))  
            speeds_during_cruising.extend(speed[start_idx:end_idx])
            degrees_during_cruising.extend(bearing_changes[start_idx:end_idx])
            cruise_bout_total_distance += np.sum(distance[start_idx:end_idx])/scale
            num_cruising_frames +=  (end_idx - start_idx)

    # convert cruising speed with scale
    if len(speeds_during_cruising) > 0:
        speeds_during_cruising = np.array(speeds_during_cruising)
        average_cruising_speed_scaled = np.mean(speeds_during_cruising) / scale
        total_cruising_degrees = np.sum(np.abs(degrees_during_cruising))
    else:
        average_cruising_speed_scaled = np.nan
        total_cruising_degrees = np.nan

    num_cruise_bouts = len(cruise_bout_timing)
    print('# cruising bouts: ' + str(num_cruise_bouts) + ', total seconds cruising: ' + str(np.round(cruise_bout_total_duration,1)))
    timing_string = ';'.join(cruise_bout_timing)
    print('timing: ', timing_string)
    durations_string = ';'.join([str(x) for x in cruise_bout_durations])
    print('durations: ', durations_string)
            
    parameters = ['scale','unit','body area (pixels^2)','body length (pixels)', 'body area (scaled)','body length (scaled)',
                  'body width (pixels)', 'body width (scaled)',
                  'clip duration','total distance','average speed',
                  '# turns','# stops', '% cruising', '# cruising frames',
                  '# cruise bouts', 'total duration cruising', 'total distance cruising',
                  'cruise bout durations', 'cruise bout timing', 'average cruising speed',
                  'cumulative bearings','bearings during cruising','bin duration',
                  'pixel threshold','tracking confidence']
    vals = [scale, unit, median_area_pixels, median_length_pixels, median_area_scaled, median_length_scaled, 
            median_width_pixels, median_width_scaled,
            clip_duration, total_distance, average_speed,
            num_turns, num_stops, cruising_proportion, num_cruising_frames,
            num_cruise_bouts, cruise_bout_total_duration, cruise_bout_total_distance, 
            durations_string, timing_string, average_cruising_speed_scaled,
            cumulative_bearings, total_cruising_degrees, time_increment, 
            pixel_threshold, tracking_confidence]
    
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

def getStops(times, xcoords, ycoords, time_increment, distance_threshold):

    '''
    From vectors of times and coordinates ...
    group into bins based on a time increment

    estimate when tardigrade stops (where distance traveled in time window is < threshold)

    Parameters
    ----------
    times : numpy array
        times of each video frame
    xcoords : numpy array
        x coordinate at each frame (usually smoothed)
    ycoords : numpy array 
        y coordinate at each frame (usually smoothed)
    body_length : float
        length of body in mm
    time_increment : float
        increment duration (in seconds) ... window duration to call stops
    distance_threshold : float
        threshold body length moved in the time increment, measured in pixels
        if distance moved in time_increment is less than distance_threshold
        then it is a STOP

    Returns
    -------
    stops = binary vector (1 = stopped, 0 = moving)


    find STOPs
    look at AVERAGE SPEED of each bin (based on time increment)
    if below a threshold for speed? = a STOP
    in STOPS array , set all frames of this bin to 1
    ''' 
    
    time_of_last_batch = times[-1] - time_increment
    start_of_last_batch = np.where(times >= time_of_last_batch)[0][0]

    stops = np.zeros(len(times))
    for i, time in enumerate(times[:start_of_last_batch]):
        
        next_time = time + time_increment
        start_bin = np.where(times >= time)[0][0]
        end_bin = np.where(times >= next_time)[0][0]
        start_coord = np.array([xcoords[start_bin], ycoords[start_bin] ] )
        end_coord = np.array([xcoords[end_bin], ycoords[end_bin]])
        distance_traveled_in_bin = np.linalg.norm(start_coord - end_coord)
    
        if distance_traveled_in_bin <= distance_threshold:  
            status = 'stop'
            stops[start_bin:end_bin] = 1
        else:
            status = 'GO'
        # print(status, times[i], distance_traveled_in_bin, distance_threshold ) # quality control
    
    return stops
    
def getTurns(times, stops, bearings, increment, turn_threshold):
    
    '''
    make bearing constant for a bit at the beginning by
    setting bearings at beginning to average of first values?
    '''
    
    set_first_constant = True
    filtered_bearings = np.copy(bearings)
    
    if set_first_constant:
        num_frames_to_average = 30
        turn_buffer_frames = 10 
        if len(bearings) > num_frames_to_average:
            first_bearings = np.array(bearings[:num_frames_to_average])
            # the first_bearings cross 0, then the average will be inflated
            search_buffer = 45
            if len(np.where(first_bearings>=360-search_buffer)[0] ) > 1: # close to NORTH on left
                if len(np.where(first_bearings<=search_buffer)[0] ) > 1: # close to NORTH on right
                    # add 360 to the ones close to NORTH on the right
                    add_360 = np.zeros(len(first_bearings))
                    add_360[np.where(first_bearings<=search_buffer)[0]] = 360
                    first_bearings = first_bearings + add_360   
            first_avg_bearing = np.mean(first_bearings)
            if first_avg_bearing > 360:
                first_avg_bearing = first_avg_bearing - 360
            new_avg_bearings = np.array([first_avg_bearing] * len(first_bearings))
            after_bearing = bearings[num_frames_to_average]
            bearing_change = gaitFunctions.change_in_bearing(first_avg_bearing, after_bearing)
            after_bearing = first_avg_bearing + bearing_change
            new_avg_bearings = gaitFunctions.fillLastBit(new_avg_bearings, first_avg_bearing, after_bearing, turn_buffer_frames)
            filtered_bearings[:num_frames_to_average] = new_avg_bearings
    
    ''' 
    deal with BEARINGs during each STOP ... 
    want to have bearings during a stop be mostly ZERO
    but also want to be pointing in the right direction after the stop
    '''
    
    num_frames_to_average = 5 # when looking for bearing before and after a stop
    turn_buffer_frames = 10  # for gradually turning from old bearing (before stop) to new bearing (after stop)
    
    # if zero runs (i.e. GO's) in stop_ranges are less
    # than a certain duration ... then do not call it a GO
    minimum_go = 2 * num_frames_to_average
    go_ranges = gaitFunctions.zero_runs(stops)
    for go_range in go_ranges:
        duration = go_range[1]-go_range[0]
        if duration <= minimum_go:
            stops[go_range[0]:go_range[1]] = 1
    
    stop_ranges = gaitFunctions.one_runs(stops)
    search_buffer = 45 # degrees to see if we crossed the NORTH line

    for stop_range in stop_ranges:
        
        # get bearing at BEGINNING of this stop
        # average of a few frames before the stop

        if stop_range[0] > num_frames_to_average:
            bearings_before_stop = bearings[stop_range[0] - num_frames_to_average : stop_range[0] ]         
            # if we crossed NORTH (e.g. from 350 to 10) we need to be careful about taking the average
            # print(stop_range[0], bearings_before_stop) # testing OK
            if len(np.where(bearings_before_stop>=360-search_buffer)[0] ) > 1: # close to NORTH on left
                if len(np.where(bearings_before_stop<=search_buffer)[0] ) > 1: # close to NORTH on right
                    # add 360 to the ones close to NORTH on the right
                    # print('We Crossed NORTH!!!!!') # testing
                    add_360 = np.zeros(len(bearings_before_stop))
                    add_360[np.where(bearings_before_stop<=search_buffer)[0]] = 360
                    bearings_before_stop = bearings_before_stop + add_360 
              
            prior_bearing = np.mean(bearings_before_stop)
        elif stop_range[0] > int(num_frames_to_average/2):
            num_frames_to_average = int(num_frames_to_average/2)
            prior_bearing = np.mean(bearings[stop_range[0] - num_frames_to_average : stop_range[0] ])
        else:
            prior_bearing = bearings[stop_range[0]]                                                     
        
        # set bearings during the stop to the prior bearing
        filtered_bearings[stop_range[0]:stop_range[1]] = prior_bearing
        
        # get bearing at END of this stop
        # average of a few frames after the stop
        if stop_range[1] + num_frames_to_average <= len(bearings):
            # print('got to here', stop_range[0], stop_range[1])
            bearings_after_stop = bearings[stop_range[1]:stop_range[1] + num_frames_to_average]
            # print(bearings_after_stop)
            # if we crossed NORTH (e.g. from 350 to 10) we need to be careful about taking the average
            if len(np.where(bearings_after_stop>=360-search_buffer)[0] ) > 1: # close to NORTH on left
                if len(np.where(bearings_after_stop<=search_buffer)[0] ) > 1: # close to NORTH on right
                    # add 360 to the ones close to NORTH on the right
                    # print('We Crossed NORTH!!!!!') # testing
                    add_360 = np.zeros(len(bearings_after_stop))
                    add_360[np.where(bearings_after_stop<=search_buffer)[0]] = 360
                    bearings_after_stop = bearings_after_stop + add_360 
            
            # print(stop_range[1], bearings_after_stop)
            after_bearing = np.mean(bearings_after_stop)

        elif stop_range[1] + int(num_frames_to_average/2) <= len(bearings): 
            num_frames_to_average = int(num_frames_to_average/2)
            after_bearing = np.mean(bearings[stop_range[1]:stop_range[1] + num_frames_to_average])
        elif stop_range[1] == len(bearings):
            after_bearing = bearings[-1]
        else:
            after_bearing = bearings[stop_range[1]]
        
        # need to find direction of rotation from prior bearing to after bearing
        # for example ... if old bearing is 90 and new bearing is 350
        # want to go LEFT (e.g. to -10) = differnce of 100 degrees
        # and not RIGHT = difference of 260 degrees
        if np.abs(after_bearing - prior_bearing) > 180:
            if prior_bearing < 180:
                after_bearing = after_bearing - 360
            else:
                after_bearing = after_bearing + 360

        # print(stop_range[1], prior_bearing, after_bearing)
        # set bearing changes towards end of the stop to equal steps between before and after
        old_bearings = np.copy(filtered_bearings)[stop_range[1]-turn_buffer_frames:stop_range[1]]
        new_bearings = gaitFunctions.fillLastBit(old_bearings,prior_bearing,after_bearing,turn_buffer_frames)
        # print(new_bearings)
        
        # some of these are negative ... that's not good!
        new_bearings = [x +360 if x<0 else x for x in new_bearings]
        new_bearings = [x -360 if x>360 else x for x in new_bearings]
        filtered_bearings[stop_range[1]-turn_buffer_frames:stop_range[1]] = new_bearings
        
    ''' 
    find bearing_changes from (new) bearings
    '''
    bearing_changes = np.zeros(len(filtered_bearings))
    for i, time in enumerate(times[:-1]):
        if i == 0:
            bearing_changes[i] = 0
        else:
            delta_bearing = gaitFunctions.change_in_bearing(filtered_bearings[i], filtered_bearings[i-1])
            bearing_changes[i] = delta_bearing  
    
    ''' now find TURNS from the bearing changes 
    look at total change in bearing from this bin
    if ABOVE a threshold (eg 28 degrees)? = a TURN
    in the TURNS array , set all frames of this bin to 1
    '''
    
    # empty array for turns
    turns = np.zeros(len(bearings))
    
    time_of_last_batch = times[-1] - increment
    start_of_last_batch = np.where(times >= time_of_last_batch)[0][0]
    
    for i, time in enumerate(times[:start_of_last_batch]):                 

        next_time = time + increment
        start_bin = np.where(times >= time)[0][0]
        end_bin = np.where(times >= next_time)[0][0]
        
        if np.sum(np.abs(bearing_changes[start_bin:end_bin])) >= turn_threshold:
            turns[start_bin:end_bin] = 1
  
    return filtered_bearings, bearing_changes, turns 

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
    bearings = bearing during video frame

    '''

    # get vector of distances traveled in every frame
    # could probably do list comprehension
    distance = np.zeros(len(times))
    speed = np.zeros(len(times))
    cumulative_distance = np.zeros(len(times))
    bearings = np.zeros(len(times))
    
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
        else:
            cumulative_distance[i] = cumulative_distance[i-1] +  distance_in_frame
    
    # set last bearing to 2nd to last position
    bearings[-1] = bearings[-2]

    return distance, speed, cumulative_distance, bearings

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
