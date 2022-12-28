#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 17:01:09 2022

@author: iwoods
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Rectangle
import numpy as np
import cv2
import sys
import gaitFunctions
import pandas as pd
import glob

def main(movie_file, plot_style = 'track'): # track or time

    # load excel file for this clip
    excel_file_exists, excel_filename = gaitFunctions.check_for_excel(movie_file)
    if excel_file_exists:
        tracked_df = pd.read_excel(excel_filename, sheet_name='pathtracking', index_col=None)
        if len(tracked_df.columns) <= 4:
            exit(' \n ==> need to run analyzePath.py first! \n')
        path_stats_df = pd.read_excel(excel_filename, sheet_name='path_stats', index_col=None)
        path_stats = dict(zip(path_stats_df['path parameter'].values, path_stats_df['value'].values))
    else:
        import initializeClip
        initializeClip.main(movie_file)
        exit('\n ==> need to run trackCritter.py and analyzePath.py first! \n')

    # get stuff out of the dataframes
    filestem = movie_file.split('.')[0]
    times = tracked_df.times.values
    
    median_area = round(path_stats['area'],4)
    clip_duration = round(path_stats['clip duration'],2)
    distance = round(path_stats['total distance'],3)
    angle_space = round(path_stats['cumulative bearings'],3)
    discrete_turns = path_stats['# turns']
    num_stops = path_stats['# stops']
    
    if plot_style == 'track':
    
        xcoords = tracked_df.xcoords.values
        ycoords = tracked_df.ycoords.values
        smoothedx = tracked_df.smoothed_x.values
        smoothedy = tracked_df.smoothed_y.values
        
        f, a, a_colorbar = plotPathColor(filestem, xcoords, ycoords, smoothedx, smoothedy, times[-1])
        
        # # ==> add labels from experiment and show plot:
        a.set_xlabel(getDataLabel(median_area, distance, clip_duration, angle_space, discrete_turns, num_stops ))
        a.set_xticks([])
        a.set_yticks([])
        a.set_title(filestem)
        plt.show()
    
    else: # plot time vs. other parameters
        
        f = plt.figure(1, figsize=(8,6))

        # plot time (x) vs. speed (left y axis)
        a1 = f.add_axes([0.1, 0.1, 0.8, 0.6])
        speed = tracked_df.speed.values
        line1 = a1.plot(times[:-1],speed[:-1],color='tab:blue',label='speed')
        a1.set_xlabel('Time (s)')
        a1.set_ylabel('Speed (mm/s)', color = 'tab:blue')
        
        # plot time vs. cumulative distance (right y axis)
        a2 = a1.twinx()
        cumulative_distance = tracked_df.cumulative_distance.values
        line2 = a2.plot(times[:-1],cumulative_distance[:-1],color='tab:red',label='distance')
        a2.set_ylabel('Cumulative distance (mm)', color = 'tab:red')
        
        # add stops as rectangles to the speed graph
        speed_xlim = a1.get_xlim()
        speed_ylim = a1.get_ylim()
        stops = tracked_df.stops.values
        stop_bouts = gaitFunctions.one_runs(stops)
        
        if len(stop_bouts) > 0:
            for bout in stop_bouts:
                start_time = times[bout[0]]
                end_time = times[bout[1]]
                a1.add_patch(Rectangle( xy=(start_time,speed_ylim[0]),
                             width = end_time - start_time,
                             height = speed_ylim[1] - speed_ylim[0],
                             facecolor = 'lightgray', edgecolor=None))
        
        # add bearing changes on a separate axis above (a3)
        bearing_changes = tracked_df.bearing_changes.values
        a3 = f.add_axes([0.1, 0.8, 0.8, 0.1])
        a3.plot(times[1:-1],bearing_changes[1:-1],color='tab:green')
        a3.set_xticks([])
        a3.set_ylabel('Change in\nbearing (˚)')
        a3.set_xlim(speed_xlim)
        a3.spines['top'].set_visible(False)
        a3.spines['right'].set_visible(False)
        a3.spines['bottom'].set_visible(False)
        
        # add turns on the bearing changes axis (a3)
        bearing_ylim = a3.get_ylim()
        turns = tracked_df.turns.values
        turn_bouts = gaitFunctions.one_runs(turns)
        
        if len(turn_bouts) > 0:
            for bout in turn_bouts:
                start_time = times[bout[0]]
                end_time = times[bout[1]]
                a3.add_patch(Rectangle( xy=(start_time,bearing_ylim[0]),
                             width = end_time - start_time,
                             height = bearing_ylim[1] - bearing_ylim[0],
                             facecolor = 'lightgray', edgecolor=None))
        
        # tiny axis to show time?
        a4 = f.add_axes([0.1, 0.74, 0.8, 0.02])
        cmap_name = 'plasma'
        cmap = mpl.cm.get_cmap(cmap_name)
        cols = cmap(np.linspace(0,1,len(times[:-1])))
        a4.scatter(times[:-1],np.ones(len(times[:-1])),c=cols,s=10) # color-coded time!
        a4.set_xlim(speed_xlim)
        a4.axis('off')
        
        # adjust parameters and show plot
        lns = line1+line2
        labs = [l.get_label() for l in lns]
        a1.legend(lns, labs, loc='lower right')
        plt.show()

def plotPathColor(filestem, xcoords, ycoords, smoothedx, smoothedy, vid_length):

    combined_frame = superImposedFirstLast(filestem)

    f = plt.figure(1, figsize=(8,6))
    a = f.add_axes([0.1, 0.1, 0.75, 0.8])
    a_colorbar = f.add_axes([0.9,0.2,0.02,0.6])
    a.imshow(combined_frame) # combined_frame or last_frame
    
    # plot path of raw coordinates (i.e. not smoothed)
    # a.plot(xcoords,ycoords, linewidth=8, color = 'gray') # raw coordinates
    
    cmap_name = 'plasma'
    cmap = mpl.cm.get_cmap(cmap_name)
    cols = cmap(np.linspace(0,1,len(xcoords)))
    a.scatter(xcoords,ycoords, s=50, c = 'k', alpha = 0.2) # raw coordinates
    a.scatter(smoothedx, smoothedy, c = cols, s=5) # smothed data
    
    a.set_xticks([])
    a.set_yticks([])
    # add legend for time
    norm = mpl.colors.Normalize(vmin=0, vmax=vid_length)
    plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), label = 'Time (sec)', cax = a_colorbar)

    return f, a, a_colorbar

def plotSmoothedPath(filestem, xcoords, ycoords, smoothedx, smoothedy):

    combined_frame = superImposedFirstLast(filestem)

    f, a = plt.subplots(1, figsize=(14,6))
    a.imshow(combined_frame) # combined_frame or last_frame
    plt.plot(xcoords,ycoords, linewidth=8, color = 'forestgreen', label = 'raw') # raw coordinates
    plt.plot(smoothedx,smoothedy, linewidth=2, color = 'lightgreen', label = 'smoothed') # smoothed
    plt.legend()
    return f, a

def getDataLabel(area, distance, vid_length, angle_space = 0, discrete_turns = 0, num_stops = 0):
    # convert from pixels?
    speed = np.around(distance/vid_length, decimals = 2)
    data_label = 'Size : ' + str(area)
    data_label += ', Distance : ' + str(distance)
    data_label += ', Time: ' + str(vid_length)
    data_label += ', Speed: ' + str(speed)
    data_label += ', Stops: ' + str(int(num_stops))

    # angle space
    if angle_space > 0:
        data_label += ', Angles explored: ' + str(angle_space)
        data_label += ', Turns: ' + str(int(discrete_turns))

    return data_label

def superImposedFirstLast(filestem):
    # superimpose first and last frames
    first_frame, last_frame = gaitFunctions.getFirstLastFrames(filestem)
    combined_frame = cv2.addWeighted(first_frame, 0.3, last_frame, 0.7, 0)
    return combined_frame

if __name__== "__main__":

    if len(sys.argv) > 1:
        movie_file = sys.argv[1]
        try:
            plot_style = sys.argv[2]
            print('hi!')
        except:
            plot_style = 'none'
    else:
        movie_file = gaitFunctions.select_movie_file()
        plot_style = 'none'

    print('Plot style is ' + plot_style)
    main(movie_file, plot_style)