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
# import glob

def main(movie_file, plot_style = ''): # track or speed or steps

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
     
    # collect data for path_stats
    # median_area = round(path_stats['area'],4)
    median_length = round(path_stats['length'],4)
    clip_duration = round(path_stats['clip duration'],2)
    distance = round(path_stats['total distance'],3)
    angle_space = round(path_stats['cumulative bearings'],3)
    discrete_turns = path_stats['# turns']
    num_stops = path_stats['# stops']
    
    if len(plot_style) == 0: # choose a type of plot
        plot_style = selectPlotStyle()
    
    if plot_style == 'track': # show critter path and smoothed path
    
        print('Here is a plot of the path taken by the critter - close the plot window to proceed')
        
        xcoords = tracked_df.xcoords.values
        ycoords = tracked_df.ycoords.values
        smoothedx = tracked_df.smoothed_x.values
        smoothedy = tracked_df.smoothed_y.values
        
        f, a, a_colorbar = plotPathColor(filestem, xcoords, ycoords, smoothedx, smoothedy, times[-1])
        
        # # ==> add labels from experiment and show plot:
        a.set_xlabel(getDataLabel(median_length, distance, clip_duration, angle_space, discrete_turns, num_stops ))
        a.set_xticks([])
        a.set_yticks([])
        a.set_title(filestem)
        plt.show()
    
    elif plot_style == 'speed': # plot time vs. other parameters
    
        print('Here is a plot the speed and distance and turns of the critter - close the plot window to proceed')
        
        f = plt.figure(1, figsize=(8,6))

        # plot time (x) vs. speed (left y axis)
        speedax = f.add_axes([0.1, 0.1, 0.63, 0.6])      
        speedax, distax = speedDistancePlot(speedax, tracked_df)
        speed_xlim = speedax.get_xlim()
                
        # plot bearing changes on a separate axis above (a3)
        bearingax = f.add_axes([0.1, 0.8, 0.63, 0.1])
        bearingax = bearingChangePlot(bearingax, tracked_df)
        bearingax.set_xlim(speed_xlim)
        
        # tiny axis to show time
        timeribbonax = f.add_axes([0.1, 0.74, 0.63, 0.02])
        timeribbonax = timeRibbonPlot(timeribbonax, tracked_df)
        timeribbonax.set_xlim(speed_xlim)
        timeribbonax.axis('off')
        
        # add 'cruising' percentage plot
        cruisingax = f.add_axes([0.9, 0.1, 0.05, 0.8])
        cruisingProportionPlot(cruisingax, tracked_df)
        
        # adjust parameters and show plot
        plt.show()
    
    elif plot_style == 'steps': # show all steps, with speed and turns and gait styles (need frameStepper)
        
        print('Here is a plot of all kinds of information about the path of the critter - close the plot window to proceed')
    
        f = plt.figure(1, figsize=(12,8))
        
        # plot time (x) vs. speed (left y axis)
        speedax = f.add_axes([0.1, 0.55, 0.65, 0.3]) 
        speedax, distax = speedDistancePlot(speedax, tracked_df)
        speed_xlim = speedax.get_xlim()
        speedax.set_xlabel('')
        
        # plot bearing changes on a separate axis above (a3)
        bearingax = f.add_axes([0.1, 0.9, 0.65, 0.05])
        bearingax = bearingChangePlot(bearingax, tracked_df)
        bearingax.set_xlim(speed_xlim)
        
        # add 'cruising' percentage plot
        cruisingax = f.add_axes([0.88, 0.55, 0.02, 0.4])
        cruisingProportionPlot(cruisingax, tracked_df)
    
        # plot the steps for the lateral legs
        steps = f.add_axes([0.1, 0.1, 0.65, 0.15])
        lateral_legs = gaitFunctions.get_leg_combos()['legs_lateral']
        steps = gaitFunctions.plotLegSet(steps, movie_file, lateral_legs)
        steps.set_xlim(speed_xlim)
        
        # plot the gait styles for lateral legs
        gaits_ax = f.add_axes([0.1, 0.26, 0.65, 0.04])
        gaits_ax = gaitFunctions.plotGaits(gaits_ax, movie_file, 'lateral')
        gaits_ax.set_xlim(speed_xlim)
        
        # proportions and legend for gait styles: lateral
        lateral_gait_proportions_ax = f.add_axes([0.83, 0.1, 0.02, 0.18])
        lateral_gait_proportions_ax = gaitFunctions.gaitStyleProportionsPlot(lateral_gait_proportions_ax, 
                                                                              [movie_file],
                                                                              'lateral')
        
        # plot the gait styles for rear legs
        reargaits_ax = f.add_axes([0.1, 0.44, 0.65, 0.04])
        reargaits_ax = gaitFunctions.plotGaits(reargaits_ax, movie_file, 'rear')
        reargaits_ax.set_xlim(speed_xlim)
        
        # plot the steps for the rear legs
        rear_steps = f.add_axes([0.1, 0.36, 0.65, 0.055])
        rear_legs = gaitFunctions.get_leg_combos()['legs_4']
        rear_steps = gaitFunctions.plotLegSet(rear_steps, movie_file, rear_legs)
        rear_steps.set_xlim(speed_xlim)
        rear_steps.set_xlabel('')
        # rear_steps.set_xticks([])
        
        # proportions and legend for gait styles: rear
        rear_gait_proportions_ax = f.add_axes([0.83, 0.33, 0.02, 0.18])
        rear_gait_proportions_ax = gaitFunctions.gaitStyleProportionsPlot(rear_gait_proportions_ax, 
                                                                              [movie_file],
                                                                              'rear')

        plt.show()
        
        
    elif plot_style == 'legs': # show steps for a particular set of legs (need frameStepper) 
        
        # choose legs to plot
        leg_combos = gaitFunctions.get_leg_combos()
        for k in sorted(leg_combos.keys()):
            print(k)
            
        print('Here is a plot of the steps of the selected legs - close the plot window to proceed')
        
        # set up an axis for the steps
        f = plt.figure(1, figsize=(12,8))
        ax = f.add_axes([0.1, 0.1, 0.85, 0.85])
        ax = gaitFunctions.plotLegSet(ax, movie_file, 'all')
        plt.show()


def cruisingProportionPlot(ax, tracked_df):
    
    stops = tracked_df.stops.values
    turns = tracked_df.turns.values
    
    non_cruising_proportion = np.count_nonzero(stops + turns) / len(stops)
    cruising_proportion = 1 - non_cruising_proportion
    
    cruising_color = 'lightcoral'
    
    ax.set_ylabel('Proportion Cruising', color = cruising_color)
    ax.bar(1, cruising_proportion, bottom = 0, 
           color = cruising_color, edgecolor = 'white', width = 0.5)
    ax.bar(1, non_cruising_proportion, bottom = cruising_proportion,
           color = 'lightgray', edgecolor = 'white', width = 0.5)
    
    ax.set_xticks([])
    ax.set_ylim([0,1])
    
    return ax
    

def timeRibbonPlot(a4, tracked_df):
    cmap_name = 'plasma'
    times = tracked_df.times.values
    cmap = mpl.cm.get_cmap(cmap_name)
    cols = cmap(np.linspace(0,1,len(times[:-1])))
    a4.scatter(times[:-1],np.ones(len(times[:-1])),c=cols,s=10) # color-coded time!
    return a4

def bearingChangePlot(a3, tracked_df):
    bearing_changes = tracked_df.bearing_changes.values
    times = tracked_df.times.values
    a3.plot(times[1:-1],bearing_changes[1:-1],color='tab:green')
    a3.set_xticks([])
    a3.set_ylabel('Change in\nbearing (˚)')
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
    return a3

def speedDistancePlot(a1, tracked_df):
    
    times = tracked_df.times.values
    speed = tracked_df.speed.values
    line1 = a1.plot(times[:-1],speed[:-1],color='tab:blue',label='speed')
    a1.set_xlabel('Time (s)')
    a1.set_ylabel('Speed (mm/s)', color = 'tab:blue')
    a1.set_xlim([0, times[-1]])
    
    # plot time vs. cumulative distance (right y axis)
    a2 = a1.twinx()
    cumulative_distance = tracked_df.cumulative_distance.values
    line2 = a2.plot(times[:-1],cumulative_distance[:-1],color='tab:red',label='distance')
    a2.set_ylabel('Cumulative distance (mm)', color = 'tab:red')
    
    # add legend
    lns = line1+line2
    labs = [l.get_label() for l in lns]
    a1.legend(lns, labs, loc='lower right')
    
    # add stops as rectangles to the speed graph
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
    return a1, a2
    

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

def getDataLabel(length, distance, vid_length, angle_space = 0, discrete_turns = 0, num_stops = 0):
    # convert from pixels?
    speed = np.around(distance/vid_length, decimals = 2)
    data_label = 'Length : ' + str(length)
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

def selectPlotStyle():
    print('\nPlot options: \n')
    print('   1. track = show critter path on image')
    print('   2. speed = show speed, distance, and turns')
    print('   3. steps = show all steps, with speed and turns (need frameStepper)')
    print('   4. legs  = show steps for a particular set of legs (need frameStepper)')
    selection = input('\nChoose one: ')
    if selection == '1':
        plot_style = 'track'
    elif selection == '2':
        plot_style = 'speed'
    elif selection == '3':
        plot_style = 'steps'
    elif selection == '4':
        plot_style = 'legs'
    else:
        print('\ninvalid selection, choosing "track"')
        plot_style = 'track'
    return plot_style

if __name__== "__main__":

    if len(sys.argv) > 1:
        movie_file = sys.argv[1]
        try:
            plot_style = sys.argv[2]
            print('hi!')
        except:
            plot_style = ''
    else:
        movie_file = gaitFunctions.select_movie_file()
        plot_style = ''

    print('Plot style is ' + plot_style)
    main(movie_file, plot_style)