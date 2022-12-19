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

def main(movie_file, plot_style = 'track'): # track or time
    # if plot_style == 'line':
    #     # ==> line plot to compare raw path with smoothed path
    #     f, a = plotSmoothedPath(filestem, xcoords, ycoords, smoothedx, smoothedy)

    # else:
    #     # ==> scatter plot of centroids along path with colormap that shows time
    #     f, a = plotPathScatter(filestem, xcoords, ycoords, vid_length)

    # # ==> add labels from experiment and show plot:
    # a.set_xlabel(getDataLabel(median_area, distance, vid_length, angle_space, discrete_turns, num_stops ))
    # a.set_xticks([])
    # a.set_yticks([])
    # plt.title(filestem)
    # plt.show()
    pass

def getDataLabel(area, distance, vid_length, angle_space = 0, discrete_turns = 0, num_stops = 0):
    # convert from pixels?
    speed = np.around(distance/vid_length, decimals = 2)
    data_label = 'Area : ' + str(area)
    data_label += ', Distance : ' + str(distance)
    data_label += ', Time: ' + str(vid_length)
    data_label += ', Speed: ' + str(speed)
    data_label += ', Stops: ' + str(num_stops)

    # angle space
    if angle_space > 0:
        data_label += ', Angles explored: ' + str(angle_space)
        data_label += ', Turns: ' + str(discrete_turns)

    return data_label

def getFirstLastFrames(filestem):
    first_frame_file = filestem + '_first.png'
    last_frame_file = filestem + '_last.png'
    
    try:
        first_frame = cv2.imread(first_frame_file)
    except:
        vidcap = cv2.VideoCapture(filestem + '.mov')
        success, image = vidcap.read()
        if success:
            first_frame = image
        else:
            print('cannot get an image from ' + filestem)
            first_frame = None
    
    try:
        last_frame = cv2.imread(last_frame_file)
    except:
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

def plotPathScatter(filestem, xcoords, ycoords, vid_length):

    combined_frame = superImposedFirstLast(filestem)
    

    f, a = plt.subplots(1, figsize=(14,6))
    a.imshow(combined_frame) # combined_frame or last_frame
    cmap_name = 'plasma'
    cmap = mpl.cm.get_cmap(cmap_name)
    cols = cmap(np.linspace(0,1,len(xcoords)))
    a.scatter(xcoords,ycoords, c = cols, s=10)
    a.set_xticks([])
    a.set_yticks([])
    # add legend for time
    norm = mpl.colors.Normalize(vmin=0, vmax=vid_length)
    plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), label = 'Time (sec)')

    return f, a

def plotSmoothedPath(filestem, xcoords, ycoords, smoothedx, smoothedy):

    combined_frame = superImposedFirstLast(filestem)

    f, a = plt.subplots(1, figsize=(14,6))
    a.imshow(combined_frame) # combined_frame or last_frame
    plt.plot(xcoords,ycoords, linewidth=8, color = 'forestgreen', label = 'raw') # raw coordinates
    plt.plot(smoothedx,smoothedy, linewidth=2, color = 'lightgreen', label = 'smoothed') # smoothed
    plt.legend()
    return f, a