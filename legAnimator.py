#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 11:25:57 2023

@author: iwoods
    
Make animation of leg state, along with a movie of walking critter

NEED - a folder of rotated, cropped images from a movie, created by critterZoomer.py


    put boxwalker in too?
        label gait style below boxwalker?
        
    put rear legs above movie, lateral legs below?


    Get whole axis of leg plots
    set xlims based on current time, and plot window
    (set axis background to a background-y color)

"""

# import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import sys
import gaitFunctions
import glob
import os
import matplotlib.animation as animation
import matplotlib.image as img

def main(movie_file):
    
    ''' make sure we have everything we need '''
    
    ## Do we have an excel file with gait style data?
    base_name = movie_file.split('.')[0]
    excel_file = base_name + '.xlsx'
    times, lateral_gaits = gaitFunctions.getGaitStyleVec(excel_file, 'lateral')
    times, rear_gaits = gaitFunctions.getGaitStyleVec(excel_file, 'rear')
    if lateral_gaits is None or rear_gaits is None:
        print('Need to have an excel file for ' + movie_file + ' with gait data from frameStepper.py')
        return
    
    ## Do we have a folder of cropped, rotated images?
    cropped_folder = base_name + '_rotated'
    if len(glob.glob(cropped_folder)) < 1:
        print('Need to run critterZoomer.py first for ' + movie_file)
        return
   
    ## Do we actually have image files in that folder?
    im_files = sorted(glob.glob(os.path.join(cropped_folder, '*.png')))
    if len(im_files) == 0:
        print('No image files found in ' + cropped_folder)
        return
    
    ## Do we have step data for this movie?
    frames_swinging = gaitFunctions.frameSwings(movie_file)
    if frames_swinging is None: # nothing here
        print(' ... need data from frameStepper.py!')
        return
    
    ## We have all the things we need!
    print('Good to go for ' + movie_file)
    print(frames_swinging)
    
    ## define the time window that we want to show for the step plots
    time_window = 5 # in seconds
    left_xlim = 0 - time_window
    right_xlim = 0
    
    rear_legs = gaitFunctions.get_leg_combos()[0]['rear']
    lateral_legs = gaitFunctions.get_leg_combos()[0]['lateral']    
    stance_color, swing_color = gaitFunctions.stanceSwingColors()
    
    print('Setting up a figure . . . ')
    f = plt.figure(figsize = (12,8))
    rear_steps_ax = f.add_axes([0.8, 0.1, 0.1, 0.8])
    lateral_steps_ax = f.add_axes([0.1, 0.1, 0.1, 0.8])
    tardi_ax = f.add_axes([0.3, 0.2, 0.3, 0.6])
    tardi_ax.axis('off')
    
    # (ax, legs_to_plot, frame_times, time_min, time_max, frames_swinging, swing_color, stance_color)
    
    rear_steps_ax = plotLegSet(rear_steps_ax, rear_legs, times, 0, 5, frames_swinging, swing_color, stance_color)
    
    ims = []
    print('Making the animation . . . can take awhile')
    for i, im_file in enumerate(im_files[:30]):
 
        frame_time = getTimeFromFilename(im_file)
        print(frame_time)
        
        # get and display the tardigrade image
        tardi_pic = img.imread(im_file)
        tardi_im = tardi_ax.imshow(tardi_pic, animated=True)
        
        # Step Plots: I think I need to plt these every time to make the animation work . . .
        
        
        
        # Box Walker
            
        
        # Time Text
        
        
        # Lateral Gait Text
        
        
        # Rear Gait Text
        
        
        # Lateral Gait Ribbon
        
        
        # Rear Gait Ribbon
        
        if i == 0:
            first_tardi_im = img.imread(im_files[i])
            tardi_ax.imshow(first_tardi_im)  # show an initial one first
        
        ims.append([tardi_im])
        
        
    # rear_steps_ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    # lateral_steps_ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    
    ani = animation.ArtistAnimation(f, ims, interval=33, blit=True, repeat = False) # repeat_delay=1000
    # ani.save(base_name + "_leganimator.mp4")
    plt.show()


def plotLegSet(ax, legs_to_plot, frame_times, time_min, time_max, frames_swinging, swing_color, stance_color):
    
    for i, leg in enumerate(legs_to_plot):
        
        # make white bars up to the time where we actually have data
        
        # add stances and swings 
        for j, frame_time in enumerate(frame_times[:-1]): # define time window to plot here
            
            bar_height = frame_times[i+1] - frame_times[i]
            if leg in frames_swinging[frame_time]:
                bar_color = swing_color
            else:
                bar_color = stance_color
                
            stepbars = ax.bar(i+1, bar_height, width=1, bottom = j*bar_height, color = bar_color)
    
    # ax.invert_yaxis()
    ax.set_xlim([0.5, len(legs_to_plot)+0.5])
    ax.set_ylabel('Time (sec)')
    ax.set_xticks(np.arange(len(legs_to_plot))+1)
    ax.set_xticklabels(legs_to_plot)
    ax.set_xlabel('legs')
    ax.set_frame_on(False)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    
    return ax, stepbars
    
def getTimeFromFilename(im_file):
    # sample filename: iw_6Feb_caffeine_tardigrade1_004-015_000030.png
    filestem = im_file.split('.')[0]
    msec = int(filestem.split('_')[-1])
    sec = msec / 1000
    return sec
    

if __name__== "__main__":

    if len(sys.argv) > 1:
        movie_file = sys.argv[1]
    else:
        movie_file = gaitFunctions.selectFile(['mp4','mov'])

    main(movie_file)
    
    
