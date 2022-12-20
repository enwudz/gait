#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 17:01:09 2022

@author: iwoods
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import cv2
import sys
import gait_analysis
import pandas as pd
import glob

def main(movie_file, plot_style = 'track'): # track or time

    # load excel file for this clip
    excel_file_exists, excel_filename = gait_analysis.check_for_excel(movie_file)
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
        pass
        

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

def getFirstLastFrames(filestem):
    first_frame_file = filestem + '_first.png'
    last_frame_file = filestem + '_last.png'
    
    if len(glob.glob(first_frame_file)) > 0:
        first_frame = cv2.imread(first_frame_file)
    else:
        print('... getting first frame ...')
        vidcap = cv2.VideoCapture(filestem + '.mov')
        success, image = vidcap.read()
        if success:
            first_frame = image
        else:
            print('cannot get an image from ' + filestem)
            first_frame = None
    
    if len(glob.glob(last_frame_file)) > 0:
        last_frame = cv2.imread(last_frame_file)
    else:
        print('... getting last frame ...')
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

if __name__== "__main__":

    if len(sys.argv) > 1:
        movie_file = sys.argv[1]
        try:
            plot_style = sys.argv[2]
        except:
            plot_style = 'none'
    else:
        movie_file = gait_analysis.select_movie_file()
        plot_style = 'none'

    main(movie_file, plot_style)