#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 17:01:09 2022

@author: iwoods
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib as mpl
import numpy as np
# import cv2
import sys
import gaitFunctions
import pandas as pd
# import glob

def main(movie_file, plot_style = ''): # track or speed or steps

    print('\nPreparing plots for ' + movie_file)

    # load excel file for this clip and get tracked_df
    excel_file_exists, excel_filename = gaitFunctions.check_for_excel(movie_file)
    if excel_file_exists:
        tracked_df = pd.read_excel(excel_filename, sheet_name='pathtracking', index_col=None)
        if len(tracked_df.columns) <= 4:
            exit(' \n ==> need to run analyzeTrack.py first! \n')
        path_stats = gaitFunctions.loadPathStats(movie_file)
        
    else:
        import initializeClip
        initializeClip.main(movie_file)
        exit('\n ==> need to run trackCritter.py and analyzeTrack.py first! \n')
     
    # some (older) data does not have units of scale ... ask to rerun analyzeTrack if so
    try: 
        unit = path_stats['unit']
    except:
        import analyzeTrack
        analyzeTrack.main(movie_file)
        path_stats = gaitFunctions.loadPathStats(movie_file)

    # collect data for path_stats
    # median_area = round(path_stats['area'],4)
    scale = float(path_stats['scale'])
    median_length = round(path_stats['body length (scaled)'],4)
    clip_duration = round(path_stats['clip duration'],2)
    distance = round(path_stats['total distance'],3)
    angle_space = round(path_stats['cumulative bearings'],3)
    discrete_turns = path_stats['# turns']
    num_stops = path_stats['# stops']
    
    # get step data if available
    stepdata_df = gaitFunctions.loadStepData(movie_file)
    have_steps = True
    if stepdata_df is None:
        print(' ... no step data available yet - run frameStepper.py')
        have_steps = False
    
    # select plot style if none provided
    if len(plot_style) == 0: # plot style not provided, choose a type of plot
        plot_style = selectPlotStyle(have_steps)
        style_specified = False
    else:
        style_specified = True
    
    # start plotting - continue to offer options until finished
    plotting = True
    while plotting:
    
        if plot_style == 'track': # show critter path and smoothed path
        
            print('Here is a plot of the path taken by the critter - close the plot window to proceed')
            
            f = plt.figure(1, figsize=(8,6))
            ax = f.add_axes([0.1, 0.1, 0.75, 0.8])
            ax_colorbar = f.add_axes([0.9,0.2,0.02,0.6])     
            ax, ax_colorbar = gaitFunctions.plotTrack(ax, ax_colorbar, movie_file, tracked_df)
            
            # ==> add labels from experiment and show plot:
            ax.set_xlabel(getDataLabel(unit, median_length, distance, clip_duration, angle_space, discrete_turns, num_stops ))
            plt.show()
            
            # prompted to keep plotting
            plot_style = keepPlotting(style_specified, have_steps)
        
        elif plot_style == 'speed': # plot time vs. other parameters
        
            print('Here is a plot of speed and distance and turns - close the plot window to proceed')
            
            f = plt.figure(1, figsize=(8,6))
    
            # plot time (x) vs. speed (left y axis)
            speedax = f.add_axes([0.1, 0.1, 0.63, 0.6])      
            speedax, distax = speedDistancePlot(speedax, tracked_df, scale, unit)
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
            
            plot_style = keepPlotting(style_specified, have_steps)
        
        elif plot_style == 'steps': # show all steps, with speed and turns and gait styles (need frameStepper)
            
            print('Here is a plot of all kinds of information about the path of the critter - close the plot window to proceed')
        
            f = plt.figure(1, figsize=(12,8))
            
            # plot time (x) vs. speed (left y axis)
            speedax = f.add_axes([0.1, 0.55, 0.65, 0.3]) 
            speedax, distax = speedDistancePlot(speedax, tracked_df, scale, unit)
            speed_xlim = speedax.get_xlim()
            speedax.set_xlabel('')
            
            # plot bearing changes on a separate axis above (a3)
            bearingax = f.add_axes([0.1, 0.9, 0.65, 0.05])
            bearingax = bearingChangePlot(bearingax, tracked_df)
            bearingax.set_xlim(speed_xlim)
            
            # 'cruising' percentage plot
            cruisingax = f.add_axes([0.88, 0.55, 0.02, 0.4])
            cruisingProportionPlot(cruisingax, tracked_df)
    
            # time ribbon plot
            timeribbonax = f.add_axes([0.1, 0.865, 0.65, 0.02])
            timeribbonax = timeRibbonPlot(timeribbonax, tracked_df)
            timeribbonax.set_xlim(speed_xlim)
            timeribbonax.axis('off')
        
            # plot the steps for the lateral legs
            steps = f.add_axes([0.1, 0.1, 0.65, 0.15])
            lateral_legs = gaitFunctions.get_leg_combos()[0]['legs_lateral']
            steps = gaitFunctions.plotLegSet(steps, movie_file, lateral_legs)
            steps.set_xlim(speed_xlim)
            
            # plot the gait styles for lateral legs
            gaits_ax = f.add_axes([0.1, 0.26, 0.65, 0.04])
            gaits_ax = gaitFunctions.plotGaits(gaits_ax, excel_filename, 'lateral')
            gaits_ax.set_xlim(speed_xlim)
            
            # proportions and legend for gait styles: lateral
            lateral_gait_proportions_ax = f.add_axes([0.83, 0.1, 0.02, 0.18])
            lateral_gait_proportions_ax = gaitFunctions.gaitStyleProportionsPlot(lateral_gait_proportions_ax, 
                                                                                  [excel_filename],
                                                                                  'lateral')
            
            # plot the gait styles for rear legs
            reargaits_ax = f.add_axes([0.1, 0.44, 0.65, 0.04])
            reargaits_ax = gaitFunctions.plotGaits(reargaits_ax, excel_filename, 'rear')
            reargaits_ax.set_xlim(speed_xlim)
            
            # plot the steps for the rear legs
            rear_steps = f.add_axes([0.1, 0.36, 0.65, 0.055])
            rear_legs = gaitFunctions.get_leg_combos()[0]['legs_4']
            rear_steps = gaitFunctions.plotLegSet(rear_steps, movie_file, rear_legs)
            rear_steps.set_xlim(speed_xlim)
            rear_steps.set_xlabel('')
            # rear_steps.set_xticks([])
            
            # proportions and legend for gait styles: rear
            rear_gait_proportions_ax = f.add_axes([0.83, 0.33, 0.02, 0.18])
            rear_gait_proportions_ax = gaitFunctions.gaitStyleProportionsPlot(rear_gait_proportions_ax, 
                                                                                  [excel_filename],
                                                                                  'rear')
    
            plt.show()
            
            plot_style = keepPlotting(style_specified, have_steps)
            
        elif plot_style == 'legs': # show steps for a particular set of legs (need frameStepper) 
            
            # choose legs to plot
            leg_combos, combo_order = gaitFunctions.get_leg_combos()
            print('Which legs should we show?')
            leg_choice = gaitFunctions.selectOneFromList(combo_order)
            legs = leg_combos[leg_choice]
                
            print('Here is a plot of the steps of the selected legs - close the plot window to proceed')
            
            # set up an axis for the steps
            fig_height = len(legs)
            f = plt.figure(1, figsize=(12,fig_height))
            ax = f.add_axes([0.1, 0.1, 0.85, 0.85])
            ax = gaitFunctions.plotLegSet(ax, movie_file, legs)
            plt.show()
            
            plot_style = keepPlotting(style_specified, have_steps)         
        
        elif plot_style == 'step parameters': # show step parameters from step_timing sheet
        
            print('Here is a plot of step parameters - close the plot window to proceed')
            
            f, axes = plt.subplots(1,5, figsize=(14,3), constrained_layout=True)
            f = gaitFunctions.stepParameterPlot(f, stepdata_df)
            plt.show()
            
            plot_style = keepPlotting(style_specified, have_steps) 
        
        elif plot_style == 'left vs. right': # step parameters for lateral legs on left and right
             
            print('Here is a plot of step parameters - comparing left lateral legs with right lateral legs')
            print(' ... close the plot window to proceed')
            # set up an axis for the step parameters
            f, axes = plt.subplots(1,5, figsize = (14,3), constrained_layout=True)
            f = gaitFunctions.stepParameterLeftRightPlot(f, stepdata_df)
            plt.show()
            
            plot_style = keepPlotting(style_specified, have_steps) 
            
        elif plot_style == 'speed vs. steps': # scatter plot of speed vs. step parameters
           
            print('Here is a plot of speed vs. step parameters, for lateral legs')
            print(' ... close the plot window to proceed')
            f, axes = plt.subplots(1,5, figsize = (14,3), constrained_layout=True)
            f = gaitFunctions.speedStepParameterPlot(f, stepdata_df)
            plt.show()      
                
            plot_style = keepPlotting(style_specified, have_steps) 
            
        elif plot_style == 'swing offsets':
              
            print('Here is a plot of swing-swing offsets for lateral legs')
            print(' ... close the plot window to proceed')
            
            # anterior-swing offsets, opposite-swing offsets (lateral), opposite-swing offsets(rear)
            # and normalized to gait cycle
            
            f, axes = plt.subplots(2,3, figsize = (10,6), constrained_layout=True)
            f = gaitFunctions.swingOffsetPlot(f, stepdata_df)
            plt.show()  
            
            plot_style = keepPlotting(style_specified, have_steps) 
        
        elif plot_style == 'metachronal lag':
            print('Here is a plot of the metachronal lag')
            print('This is the amount of time elapsed between the swing of the third leg')
            print('and the swing of the first leg')
            print(' ... close the plot window to proceed')

            f, axes = plt.subplots(1,2, figsize = (8,3), constrained_layout=True)
            f = gaitFunctions.metachronalLagLRPlot(f, stepdata_df)
            plt.show()
            
            plot_style = keepPlotting(style_specified, have_steps)
        
        elif plot_style == 'finished':
            plotting = False
            break
            
        elif plotting == False:
            print('done plotting!')
            break
        
        else:
            plot_style = selectPlotStyle(have_steps)
            
def keepPlotting(style_specified, have_steps=False):
    if style_specified:
        plot_style = 'finished'
    else:
        plot_style = selectPlotStyle(have_steps)
    return plot_style

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
    cmap = mpl.colormaps.get_cmap(cmap_name)
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

def speedDistancePlot(a1, tracked_df, scale, unit):
    
    if unit == 'inch':
        unit = 'in'
    
    times = tracked_df.times.values
    speed = tracked_df.speed.values / scale
    line1 = a1.plot(times[:-1],speed[:-1],color='tab:blue',label='speed')
    a1.set_xlabel('Time (s)')
    a1.set_ylabel('Speed (' + unit + '/s)', color = 'tab:blue')
    a1.set_xlim([0, times[-1]])
    
    # plot time vs. cumulative distance (right y axis)
    a2 = a1.twinx()
    cumulative_distance = tracked_df.cumulative_distance.values / scale
    line2 = a2.plot(times[:-1],cumulative_distance[:-1],color='tab:red',label='distance')
    a2.set_ylabel('Cumulative distance (' + unit + ')', color = 'tab:red')
    
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
    

def getDataLabel(unit, length, distance, vid_length, angle_space = 0, discrete_turns = 0, num_stops = 0):
    
    if unit == 'inch':
        unit = 'in'
    
    data_label = ''
    
    # convert from pixels?
    speed = np.around(distance/vid_length, decimals = 3)
    if length > 0:
        data_label += 'Length: ' + str(length) + ' ' + unit
    data_label += ', Distance: ' + str(distance) + ' ' + unit
    data_label += ', Time: ' + str(vid_length) + ' sec'
    data_label += ', Speed: ' + str(speed) + ' ' + unit + '/sec'
    data_label += '\nStops: ' + str(int(num_stops))

    # angle space
    if angle_space > 0:
        data_label += ', Angles explored: ' + str(angle_space)
        data_label += ', Turns: ' + str(int(discrete_turns))

    return data_label


def selectPlotStyle(have_steps=False):
    
    plotStyles = ['track',
                  'speed',
                  'steps',
                  'legs',
                  'step parameters',
                  'left vs. right',
                  'speed vs. steps',
                  'swing offsets',
                  'metachronal lag']
    
    plotDescriptions = ['show critter path on background', # track
                        'show speed, distance, and turns', # speed
                        'show all steps, with speed and turns', # steps
                        'show steps for a particular set of legs', # legs
                        'show step parameters (stance, swing, duty factor, cycle, distance)', # step parameters
                        'show step parameters comparing left vs. right lateral legs', # left vs. right
                        'show scatter plot of speed vs step parameters (for lateral legs)', # speed vs. steps
                        'show swing-swing timing offsets', # offsets
                        'show elapsed time between 3rd leg swing and 1st leg swing'
                        ]
    print('\nPlot options: \n')
    print('  0. finished = quit plotting')
    
    if have_steps:
        last_ind = len(plotStyles)
    else:
        last_ind = 2 # how many plot choices do not require step timing data?
    
    for i in np.arange(last_ind):
        print('  ' + str(i+1) + '. ' + plotStyles[i] + ' = ' + plotDescriptions[i])
    
    selection = input('\nChoose one: ')
    
    try:
        ind = int(selection) - 1
        if ind == -1:
            plot_style = 'finished'
            print('... Finished Plotting!\n')
        else:
            plot_style = plotStyles[ind]
            print('You chose ' + plot_style)
    except:
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
        movie_file = gaitFunctions.selectFile(['mp4','mov'])
        plot_style = ''

    if len(plot_style) > 0:
        print('Plot style is ' + plot_style)
    main(movie_file, plot_style)
