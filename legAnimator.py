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

WISHLIST

if filmed from the top, select leg sets to show


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
    
    ## define the time window that we want to show for the step plots
    time_window = 5 # in seconds
    
    ## Figure out what species we are dealing with
    identity_info = gaitFunctions.loadIdentityInfo(movie_file)
    if 'species' in identity_info.keys():
        species = identity_info['species']
        num_legs = identity_info['num_legs']
    else:
        species = 'tardigrade'
        num_legs = 8
    print('This is a ' + species)
    
    ## if species is tardigrade, ask if we want to do lateral legs or rear legs
    if species == 'tardigrade':
        leg_set = gaitFunctions.choose_lateral_rear()
    else:
        leg_set = species
        
    # if movie duration is less then selected time window, set time window to movie duration
    movie_duration = float(identity_info['duration'])
    if movie_duration < time_window:
        print('Setting time window for step plots to ' + str(movie_duration) + ' seconds')
        time_window = movie_duration
    
    ''' make sure we have everything we need '''
    
    ## Do we have a folder of cropped, rotated images to animate into a movie?
    base_name = movie_file.split('.')[0]
    excel_file = base_name + '.xlsx'
    cropped_folder = base_name + '_rotacrop'
    if len(glob.glob(cropped_folder)) < 1:
        print('Need rotated images from ' + movie_file + ' ... run rotaZoomer.py')
        return
   
    ## Do we actually have image files in that folder?
    im_files = sorted(glob.glob(os.path.join(cropped_folder, '*.png')))
    if len(im_files) == 0:
        print('No image files found in ' + cropped_folder + ' ... save images with rotaZoomer.py')
        return
    
    ## Do we have step data for this movie?
    frames_swinging = gaitFunctions.frameSwings(movie_file)
    if frames_swinging is None: # nothing here
        print(' ... need step data from frameStepper.py!')
        return
    times, gaits = gaitFunctions.getGaitStyleVec(excel_file, leg_set)    
    if gaits is None:
        print('Need to have an excel file for ' + movie_file + ' with gait data from frameStepper.py for ' + leg_set + ' legs')
        return
    else:
        print('Found gait data for ' + movie_file)
        
    ## Do we have images of leg steps and gait styles for this leg set?
    stepfolder = base_name + '_' + leg_set + 'steps'
    if len(glob.glob(stepfolder)) < 1:
        print('Making ' + leg_set + ' step images for ' + movie_file)
        os.mkdir(stepfolder)
        makeStepImages(stepfolder, base_name, frames_swinging, time_window, leg_set)
    else:
        print('\nHave ' + leg_set + ' step images for ' + movie_file + '!')
    step_files = sorted(glob.glob(os.path.join(stepfolder,'*.png')))
        
    ## Do we have boxwalker images for this movie
    boxwalker_folder = base_name + '_boxwalker_' + leg_set
    if len(glob.glob(boxwalker_folder)) < 1:
        print('Making boxwalker images for ' + movie_file + ', ' + leg_set + ' legs')
        import boxWalker
        boxWalker.main(movie_file, leg_set)
    else:
        print('\nHave ' + leg_set + ' boxwalker images for ' + movie_file + '!')
    boxwalker_files = sorted(glob.glob(os.path.join(boxwalker_folder,'*.png')))
        
    ## We have all the things we need!
    print('\nGood to go for ' + movie_file)
    
    # ## Do the animated figure!
    print('\nSetting up an animated figure . . . ')
    
    if species == 'tardigrade':
        if leg_set == 'rear':
            f = plt.figure(figsize = (10,8))
            steps_ax = f.add_axes([0.05, 0.18, 0.25, 0.8]) # width:height is 2.5:8 for rear
            critter_ax = f.add_axes([0.3, 0.18, 0.3, 0.75]) # width:height is 1:2 for tardi
            box_ax = f.add_axes([0.6, 0.18, 0.4, 0.75]) # width:height is 1:2 for box
            legend_ax = f.add_axes([0.13, 0.03, 0.12, 0.15])
            legend_ax = gaitFunctions.gaitStyleLegend(legend_ax, 'rear')
        else:
            f = plt.figure(figsize = (12,8))
            steps_ax = f.add_axes([0.05, 0.18, 0.37, 0.8]) # width:height is 3.7:8 for lateral
            critter_ax = f.add_axes([0.35, 0.18, 0.3, 0.75]) # width:height is 1:2 for tardi
            box_ax = f.add_axes([0.58, 0.18, 0.4, 0.75]) # width:height is 1:2 for box
            legend_ax = f.add_axes([0.15, 0.03, 0.2, 0.15])
            legend_ax = gaitFunctions.gaitStyleLegend(legend_ax, 'lateral')
    else:
        # set up a plot for a non-tardigrade critter
        f = plt.figure(figsize = (14,8))
        step_ax_width = 0.8
        step_ax_height = 0.5 # step_ax_width / (10/(num_legs/2))
        steps_ax = f.add_axes([0.1, 0.05, step_ax_width, step_ax_height]) # width:height is 10:number_of_legs/2 for non-tardi critter
        critter_ax = f.add_axes([0.45, 0.51, 0.5, 0.4])  # width:height is 4:3 for non-tardi critter
        box_ax = f.add_axes([0.05, 0.51, 0.5, 0.4]) # width:height is number_of_legs / 2 + 1 : number_of_legs for non-tardi critter
        
    # exit()
    critter_ax.axis('off')
    steps_ax.axis('off')
    box_ax.axis('off')
    
    ims = []
    print('Making the animation . . . can take awhile')
    for i, im_file in enumerate(im_files):
 
        # frame_time = getTimeFromFilename(im_file)
        # print(frame_time)
        
        # get and display the tardigrade image
        critter_pic = img.imread(im_file)
        critter_im = critter_ax.imshow(critter_pic, animated=True)
        
        step_pic = img.imread(step_files[i])
        step_im = steps_ax.imshow(step_pic, animated=True)
        
        box_pic = img.imread(boxwalker_files[i])
        box_im = box_ax.imshow(box_pic, animated=True)
        
        # if i == 0: # show an initial one first
        #     tardi_ax.imshow(img.imread(im_files[i])) 
        #     lateral_steps_ax.imshow(img.imread(lateral_step_files[i]))
        #     rear_steps_ax.imshow(img.imread(rear_step_files[i]))
        
        # ims.append([critter_im, step_im])
        ims.append([critter_im, step_im, box_im])
    
    
    # save at multiple fps?
    fps_list = [5,10,30]
    # fps_list= [30]
    
    for fps in fps_list:
        frame_interval = int(1000/fps)
        
        ani = animation.ArtistAnimation(f, ims, interval=frame_interval, blit=True, repeat = False) # repeat_delay=1000
        
        ani_file = base_name + "_" + str(fps) + "fps_" + leg_set + "_leganimator.mp4"
        print(str(fps) + 'fps animation finished. Saving the animation as ' + ani_file)
        ani.save(ani_file)
    
        # plt.show()

    exit()
    
    
    return
    
def makeStepImages(folder, base_name, frames_swinging, time_window, leg_set='lateral'):
    
    stance_color, swing_color = gaitFunctions.stanceSwingColors()
    excel_file = base_name + '.xlsx'
    
    if leg_set == 'lateral':
        ## lateral step figures
        lateral_legs = gaitFunctions.get_leg_list(6,'stepplot')
        times, lateral_gait_styles = gaitFunctions.getGaitStyleVec(excel_file, 'lateral')
        lateral_combos, lateral_combo_colors = gaitFunctions.get_gait_combo_colors('lateral')
        
        print('Saving lateral step figures in ' + folder)
        print(' ... this takes awhile - sit tight!')
 
        # frame by frame adjustment of y axis time
        for frame_time in times:
            
            fname = base_name + '_' + str(int(frame_time*1000)).zfill(6) + '.png'
            fig_size = (3.7,8)
            f = plt.figure(figsize=fig_size)
            step_ax = f.add_axes([0.15, 0.05, 0.6, 0.88])
            gait_ax = f.add_axes([0.8, 0.05, 0.15, 0.88])
            step_ax = plotLegSetAtTime(step_ax, 'vertical', lateral_legs, times, frames_swinging, time_window, 
                                                frame_time, swing_color, stance_color)
                  
            gait_ax = plotGaitStyleAtTime(gait_ax, leg_set, times, lateral_gait_styles, time_window, frame_time, lateral_combo_colors)
            
            step_ax.set_ylim([frame_time-time_window, frame_time])
            gait_ax.set_ylim([frame_time-time_window, frame_time])
            
            plt.savefig(os.path.join(folder,fname))
            plt.close(f)
    
    elif leg_set == 'rear':
        ## rear step figures
        rear_legs = ['L4','R4']
        rear_combos, rear_combo_colors = gaitFunctions.get_gait_combo_colors('rear')
        times, rear_gait_styles = gaitFunctions.getGaitStyleVec(excel_file, 'rear')
        
        print('Saving rear step figures in ' + folder)
        print(' ... this takes awhile - sit tight!')
        
        for frame_time in times:
            
            fname = base_name + '_' + str(int(frame_time*1000)).zfill(6) + '.png'
            fig_size = (2.5,8)
            f = plt.figure(figsize=fig_size)
            step_ax = f.add_axes([0.25, 0.05, 0.45, 0.88])
            gait_ax = f.add_axes([0.75, 0.05, 0.2, 0.88])
            step_ax = plotLegSetAtTime(step_ax, 'vertical', rear_legs, times, frames_swinging, time_window, 
                                                frame_time, swing_color, stance_color)
            gait_ax = plotGaitStyleAtTime(gait_ax, leg_set, times, rear_gait_styles, time_window, frame_time, rear_combo_colors)
            
            step_ax.set_ylim([frame_time-time_window, frame_time])
            gait_ax.set_ylim([frame_time-time_window, frame_time])
            
            plt.savefig(os.path.join(folder,fname))
            plt.close(f)
            
    else:
        
        if leg_set in ['cat','dog','tetrapod','four']:
            legs = gaitFunctions.get_leg_list(4,'stepplot')
        elif leg_set in ['human','two']:
            legs = gaitFunctions.get_leg_list(2,'stepplot')
        else:
            sys.exit("I don't know what to do with " + leg_set)
        
        combos, combo_colors = gaitFunctions.get_gait_combo_colors(leg_set)
        times, gaits = gaitFunctions.getGaitStyleVec(excel_file, leg_set)

        print('Saving step figures in ' + folder)
        print(' ... this takes awhile - sit tight!')
        
        for frame_time in times:
            fname = base_name + '_' + str(int(frame_time*1000)).zfill(6) + '.png'
            fig_size = (10, len(legs)/2)
            f = plt.figure(figsize=fig_size)
            step_ax = f.add_axes([0.1, 0.15,  0.85, 0.6])
            gait_ax = f.add_axes([0.1, 0.8,   0.85, 0.1])
            step_ax = plotLegSetAtTime(step_ax, 'horizontal', legs, times, frames_swinging, time_window,
                                       frame_time, swing_color, stance_color)
            gait_ax = plotGaitStyleAtTime(gait_ax, leg_set, times, gaits, time_window, frame_time, combo_colors)
            
            step_ax.set_xlim([frame_time-time_window, frame_time])
            gait_ax.set_xlim([frame_time-time_window, frame_time])
            
            plt.savefig(os.path.join(folder,fname))
            # plt.show()
            plt.close(f)
    
    plt.close("all")
    return fig_size


def plotGaitStyleAtTime(ax, leg_set, frame_times, gait_styles, time_window, current_time, combo_colors): # combo colors
    
    if leg_set in ['lateral','rear']:
        plot_orientation = 'vertical'
    else:
        plot_orientation = 'horizontal'
    
    # make white bars up to the time where we actually have data
    min_time = current_time - time_window
    
    if plot_orientation == 'vertical':
        ax.bar(1, np.abs(min_time), width = 1, bottom = min_time, color = 'white')
    else:
        ax.barh(1, np.abs(min_time), height = 1, left = min_time, color = 'white')
    
    # add stances and swings 
    max_time_index = np.argmax(frame_times >= current_time)
    
    previous_time = 0
    for i, style in enumerate(gait_styles[:max_time_index]):
        bar_size = frame_times[i] - previous_time
        if plot_orientation == 'vertical':
            ax.bar(1, bar_size, width=1, bottom=previous_time, color = combo_colors[style])
        else:
            ax.barh(1, bar_size, height=1, left=previous_time, color = combo_colors[style])
        previous_time += bar_size
    
    # ax.invert_yaxis()
    if plot_orientation == 'vertical':
        ax.set_xlabel('gait\n' + leg_set)
        ax.xaxis.set_label_position('top') 
        # ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    else:
        ax.set_ylabel('gait\n')
        ax.yaxis.set_label_position('right') 

    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_frame_on(False)
    
    return ax


def plotLegSetAtTime(ax, plot_orientation, legs_to_plot, frame_times, frames_swinging, time_window, current_time, swing_color, stance_color):
    
    for i, leg in enumerate(legs_to_plot):
        
        # make white bars up to the time where we actually have data
        min_time = current_time - time_window
        if plot_orientation == 'vertical':
            stepbars = ax.bar(i+1, np.abs(min_time), width = 1, bottom = min_time, color = 'white')
        else:
            stepbars = ax.barh(i+1, np.abs(min_time), height = 1, left = min_time, color = 'white')
        
        # add stances and swings 
        max_time_index = np.argmax(frame_times >= current_time)
        
        bottom_val = 0
        for j, frame_time in enumerate(frame_times[:max_time_index]): # define time window to plot here
            
            bar_size = frame_times[j+1] - frame_times[j]
            if leg in frames_swinging[frame_time]:
                bar_color = swing_color
            else:
                bar_color = stance_color
            
            if plot_orientation == 'vertical':
                ax.bar(i+1, bar_size, width=1, bottom = bottom_val, color = bar_color)
            else: 
                ax.barh(i+1, bar_size, height=1, left = bottom_val, color = bar_color)
            bottom_val += bar_size
    
    if plot_orientation == 'vertical':
        # ax.invert_yaxis()
        ax.set_xlim([0.5, len(legs_to_plot)+0.5])
        ax.set_ylabel('Time (sec)')
        ax.set_xticks(np.arange(len(legs_to_plot))+1)
        ax.set_xticklabels(legs_to_plot)
        ax.set_xlabel('legs')
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top') 
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    else:
        # ax.invert_xaxis()
        ax.set_ylim([0.5, len(legs_to_plot)+0.5])
        ax.set_xlabel('Time (sec)')
        ax.set_yticks(np.arange(len(legs_to_plot))+1)
        ax.set_yticklabels(legs_to_plot)
        ax.set_ylabel('legs')
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position('right') 
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    ax.set_frame_on(False)
    
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
    
    
