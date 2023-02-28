#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 11:25:57 2023

@author: iwoods
    
Make animation of leg states and gait styles, along with a movie of walking critter

I can animate saved images ... or 2D functions 
... but I cannot figure out how to animate the stacked bar charts for steps and gaits

SO ... klugey solution here:
make folders full of images of what I want ... and animate them ... 

(reminder of how to make a movie with ffmpeg, within a folder of images)
ffmpeg -f image2 -r 30 -pattern_type glob -i '*.png' -pix_fmt yuv420p -crf 20 test_movie.mp4

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
    
    ## define the time window that we want to show for the step plots
    time_window = 5 # in seconds
    
    ## Do we have an excel file with gait style data?
    base_name = movie_file.split('.')[0]
    excel_file = base_name + '.xlsx'
    times, lateral_gaits = gaitFunctions.getGaitStyleVec(excel_file, 'lateral')
    times, rear_gaits = gaitFunctions.getGaitStyleVec(excel_file, 'rear')
    if lateral_gaits is None or rear_gaits is None:
        print('Need to have an excel file for ' + movie_file + ' with gait data from frameStepper.py')
        return
    
    ## Do we have a folder of cropped, rotated images?
    cropped_folder = base_name + '_rotacrop'
    if len(glob.glob(cropped_folder)) < 1:
        print('Need rotated images from ' + movie_file + ' ... run critterZoomer.py')
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
    
    ## Do we have box walker images? [ask: make or exit]
    ## ************************************
    
    ## Do we have images of lateral leg steps and gait styles [ask: make or exit]
    lateral_stepfolder = base_name + '_lateralsteps'
    if len(glob.glob(lateral_stepfolder)) < 1:
        print('Making lateral step images for ' + movie_file)
        os.mkdir(lateral_stepfolder)
        makeStepImages(lateral_stepfolder, base_name, frames_swinging, time_window, 'lateral')
    else:
        print('\nHave lateral step images for ' + movie_file + '!')
    lateral_step_files = sorted(glob.glob(os.path.join(lateral_stepfolder,'*.png')))
    
    ## Do we have images of rear leg steps and gait styles [ask: make or exit]
    rear_stepfolder = base_name + '_rearsteps'
    if len(glob.glob(rear_stepfolder)) < 1:
        print('Making rear step images for ' + movie_file)
        os.mkdir(rear_stepfolder)
        makeStepImages(rear_stepfolder, base_name, frames_swinging, time_window, 'rear')
    else:
        print('\nHave rear step images for ' + movie_file + '!')
    rear_step_files = sorted(glob.glob(os.path.join(rear_stepfolder,'*.png')))
    
    ## We have all the things we need!
    print('\nGood to go for ' + movie_file)
    
    # ## Do the animated figure!
    print('\nSetting up an animated figure . . . ')
    f = plt.figure(figsize = (10,8))
    
    lateral_steps_ax = f.add_axes([0.1, 0.18, 0.296, 0.83]) # size is 3.7, 8
    tardi_ax = f.add_axes([0.4, 0.18, 0.3, 0.77])
    rear_steps_ax =    f.add_axes([0.7, 0.18, 0.2,   0.83]) # size is 2.5, 8
    
    tardi_ax.axis('off')
    lateral_steps_ax.axis('off')
    rear_steps_ax.axis('off')
    
    lateral_gait_legend_ax = f.add_axes([0.15, 0.03, 0.23, 0.15])
    lateral_gait_legend_ax = gaitFunctions.gaitStyleLegend(lateral_gait_legend_ax, 'lateral')
    
    rear_gait_legend_ax = f.add_axes([0.75, 0.1, 0.13, 0.07])
    rear_gait_legend_ax = gaitFunctions.gaitStyleLegend(rear_gait_legend_ax, 'rear')
    
    ims = []
    print('Making the animation . . . can take awhile')
    for i, im_file in enumerate(im_files):
 
        # frame_time = getTimeFromFilename(im_file)
        # print(frame_time)
        
        # get and display the tardigrade image
        tardi_pic = img.imread(im_file)
        tardi_im = tardi_ax.imshow(tardi_pic, animated=True)
        
        lateral_pic = img.imread(lateral_step_files[i])
        lateral_im = lateral_steps_ax.imshow(lateral_pic, animated=True)
        
        rear_pic = img.imread(rear_step_files[i])
        rear_im = rear_steps_ax.imshow(rear_pic, animated=True)
        
        # if i == 0: # show an initial one first
        #     tardi_ax.imshow(img.imread(im_files[i])) 
        #     lateral_steps_ax.imshow(img.imread(lateral_step_files[i]))
        #     rear_steps_ax.imshow(img.imread(rear_step_files[i]))
        
        ims.append([tardi_im, rear_im, lateral_im])
    
    ani = animation.ArtistAnimation(f, ims, interval=33, blit=True, repeat = False) # repeat_delay=1000
    ani.save(base_name + "_leganimator.mp4")
    plt.show()

    exit()
    
    
    return
    
def makeStepImages(folder, base_name, frames_swinging, time_window, leg_set='lateral'):
    
    stance_color, swing_color = gaitFunctions.stanceSwingColors()
    excel_file = base_name + '.xlsx'
    
    if leg_set == 'lateral':
        ## lateral step figures
        lateral_legs = ['L3','L2','L1','R1','R2','R3']
        times, lateral_gait_styles = gaitFunctions.getGaitStyleVec(excel_file, 'lateral')
        lateral_combos, lateral_combo_colors = gaitFunctions.get_gait_combo_colors('lateral')
        
        print('Saving lateral step figures in ' + folder)
        print(' ... this takes awhile - sit tight!')
 
        # frame by frame adjustment of y axis time
        for frame_time in times:
            
            fname = base_name + '_' + str(int(frame_time*1000)).zfill(6) + '.png'
            f = plt.figure(figsize=(3.7,8))
            step_ax = f.add_axes([0.15, 0.05, 0.6, 0.88])
            gait_ax = f.add_axes([0.8, 0.05, 0.15, 0.88])
            step_ax = plotLegSetAtTime(step_ax, lateral_legs, times, frames_swinging, time_window, 
                                                frame_time, swing_color, stance_color)
                  
            gait_ax = plotGaitStyleAtTime(gait_ax, 'lateral', times, lateral_gait_styles, time_window, frame_time, lateral_combo_colors)
            
            step_ax.set_ylim([frame_time-time_window, frame_time])
            gait_ax.set_ylim([frame_time-time_window, frame_time])
            
            plt.savefig(os.path.join(folder,fname))
            plt.close(f)
    
    
    else:
        ## rear step figures
        rear_legs = ['L4','R4']
        rear_combos, rear_combo_colors = gaitFunctions.get_gait_combo_colors('rear')
        times, rear_gait_styles = gaitFunctions.getGaitStyleVec(excel_file, 'rear')
        
        print('Saving rear step figures in ' + folder)
        print(' ... this takes awhile - sit tight!')
        
        for frame_time in times:
            
            fname = base_name + '_' + str(int(frame_time*1000)).zfill(6) + '.png'
            f = plt.figure(figsize=(2.5,8))
            step_ax = f.add_axes([0.25, 0.05, 0.45, 0.88])
            gait_ax = f.add_axes([0.75, 0.05, 0.2, 0.88])
            step_ax = plotLegSetAtTime(step_ax, rear_legs, times, frames_swinging, time_window, 
                                                frame_time, swing_color, stance_color)
            
            gait_ax = plotGaitStyleAtTime(gait_ax, 'rear', times, rear_gait_styles, time_window, frame_time, rear_combo_colors)
            
            step_ax.set_ylim([frame_time-time_window, frame_time])
            gait_ax.set_ylim([frame_time-time_window, frame_time])
            
            plt.savefig(os.path.join(folder,fname))
            plt.close(f)
    
    plt.close("all")


def plotGaitStyleAtTime(ax, leg_set, frame_times, gait_styles, time_window, current_time, combo_colors): # combo colors
            
    # make white bars up to the time where we actually have data
    min_time = current_time - time_window
    ax.bar(1, np.abs(min_time), width = 1, bottom = min_time, color = 'white')
    
    # add stances and swings 
    max_time_index = np.argmax(frame_times >= current_time)
    
    previous_time = 0
    for i, style in enumerate(gait_styles[:max_time_index]):
        bar_height = frame_times[i] - previous_time
        ax.bar(1, bar_height, width=1, bottom=previous_time, color = combo_colors[style])
        previous_time += bar_height
    
    # ax.invert_yaxis()
    ax.set_xlabel('gait\n' + leg_set)

    ax.set_yticks([])
    ax.set_xticks([])
    ax.xaxis.set_label_position('top') 
    ax.set_frame_on(False)
    # ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    
    return ax


def plotLegSetAtTime(ax, legs_to_plot, frame_times, frames_swinging, time_window, current_time, swing_color, stance_color):
    
    for i, leg in enumerate(legs_to_plot):
        
        # make white bars up to the time where we actually have data
        min_time = current_time - time_window
        stepbars = ax.bar(i+1, np.abs(min_time), width = 1, bottom = min_time, color = 'white')
        
        # add stances and swings 
        max_time_index = np.argmax(frame_times >= current_time)
        
        bottom_val = 0
        for j, frame_time in enumerate(frame_times[:max_time_index]): # define time window to plot here
            
            bar_height = frame_times[j+1] - frame_times[j]
            if leg in frames_swinging[frame_time]:
                bar_color = swing_color
            else:
                bar_color = stance_color
                
            ax.bar(i+1, bar_height, width=1, bottom = bottom_val, color = bar_color)
            bottom_val += bar_height
    
    # ax.invert_yaxis()
    ax.set_xlim([0.5, len(legs_to_plot)+0.5])
    ax.set_ylabel('Time (sec)')
    ax.set_xticks(np.arange(len(legs_to_plot))+1)
    ax.set_xticklabels(legs_to_plot)
    ax.set_xlabel('legs')
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top') 
    ax.set_frame_on(False)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    
    return ax
    
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
    
    
